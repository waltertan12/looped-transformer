#!/usr/bin/env python
"""H19 (D-013): does each loop pass handle one level of brackets?

For every sub-expression of a test problem, at the token right after it ends (where its value is
first fully determined), train a linear probe on the hidden state after each pass (looped) or each
layer (deep, shallow) to read out the last digit of its value. If passes handle levels, height-1
sub-expressions become readable early, height-2 later, and so on.

    scripts/probe_levels.py --ckpts long/c2k8-s0,long/c16k1-s0,long/c2k1-s0 --out results/toy/probe/levels.json

Runs on CPU by default (small models; keeps the GPU for training).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ckpt import load_checkpoint, train_k  # noqa: E402
from looptf.toy.data import decode, encode  # noqa: E402
from looptf.toy.dataset import slice_paths  # noqa: E402
from looptf.toy.model import apply_rope, place_ids  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def _height(n) -> int:
    return 1 + max(_height(n.left), _height(n.right)) if isinstance(n, ast.BinOp) else 0


def subexpr_nodes(prompt: str) -> list[tuple[int, int, int]]:
    """(position, height, value) for every operator node of `prompt` (which ends with '=').
    Position = index of the character right after the node's text: a ')' , the next operator or
    '='. That is the first token at which the node's value is fully determined."""
    expr = prompt[:-1]
    out = []
    for n in ast.walk(ast.parse(expr, mode="eval").body):
        if isinstance(n, ast.BinOp):
            v = eval(compile(ast.Expression(n), "<n>", "eval"))
            out.append((n.end_col_offset, _height(n), int(v)))
    return out


@torch.no_grad()
def _attn_weights(blk, u, cos, sin) -> torch.Tensor:
    a = blk.attn
    B, T, D = u.shape
    q, k, _ = a.qkv(blk.norm1(u)).view(B, T, 3, a.n_heads, a.head_dim).permute(2, 0, 3, 1, 4)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = (q @ k.transpose(-1, -2)) / a.head_dim ** 0.5
    mask = torch.ones(T, T, dtype=torch.bool, device=u.device).tril()
    return F.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)  # [B, heads, T, T]


@torch.no_grad()
def pass_states(model, x: torch.Tensor, n_loops: int, per_layer: bool):
    """Hidden states entering the core and after each pass (per_layer=False) or after each core
    block (per_layer=True), plus attention weights [n_core, B, heads, T, T] per pass."""
    cfg = model.cfg
    T = x.shape[1]
    cos, sin = model.rope_cos[:T], model.rope_sin[:T]
    e = model.embed(x)
    if model.place is not None:
        pid = place_ids(x, cfg.max_places)
        if getattr(model, "place_ans", None) is None:
            e = e + model.place(pid)
        else:
            from looptf.toy.model import answer_mask
            after = answer_mask(x)
            z = torch.zeros_like(pid)
            e = e + model.place(torch.where(after, z, pid)) + model.place_ans(torch.where(after, pid, z))
    for blk in model.prelude:
        e = blk(e, cos, sin)
    states, attn, h = [e], [], e
    for _ in range(n_loops):
        if cfg.inject == "concat":
            u = model.adapter(torch.cat([h, e], dim=-1))
        elif cfg.inject == "add":
            u = h + e
        else:
            u = h
        ws = []
        for blk in model.core:
            ws.append(_attn_weights(blk, u, cos, sin))
            u = blk(u, cos, sin)
            if per_layer:
                states.append(u)
        attn.append(torch.stack(ws))
        h = u
        if not per_layer:
            states.append(h)
    return states, attn


def load_problems(n_per_depth: int, depths=range(2, 9), seed: int = 0):
    root = REPO / "data/nested_bal"
    cols = json.loads((root / "meta.json").read_text())["columns"]
    rng = np.random.default_rng(seed)
    probs = []
    for d in depths:
        tp, mp = slice_paths(root, "test", d)
        tok, lab = np.load(tp), np.load(mp)
        idx = np.flatnonzero(lab[:, cols.index("op_digits")] == 1)
        idx = rng.choice(idx, min(n_per_depth, len(idx)), replace=False)
        for i in idx:
            probs.append((decode(tok[i, :lab[i, cols.index("length")]]), d))
    return probs


def collect(model, k: int, per_layer: bool, probs, bs: int = 256):
    """Features [n_stages, n_nodes, d] (float16), heights, last digits, problem ids."""
    feats, H, Y, P = None, [], [], []
    for s in range(0, len(probs), bs):
        chunk = probs[s:s + bs]
        T = max(len(p) for p, _ in chunk)
        x = torch.zeros(len(chunk), T, dtype=torch.long)
        for i, (p, _) in enumerate(chunk):
            x[i, :len(p)] = torch.tensor(encode(p))
        states, _ = pass_states(model, x, k, per_layer)
        st = torch.stack(states)  # [S, B, T, d]
        rows = []
        for i, (p, _) in enumerate(chunk):
            for pos, h, v in subexpr_nodes(p):
                rows.append((i, pos)); H.append(h); Y.append(abs(v) % 10); P.append(s + i)
        bi = torch.tensor([r[0] for r in rows]); pi = torch.tensor([r[1] for r in rows])
        f = st[:, bi, pi].half().numpy()
        feats = f if feats is None else np.concatenate([feats, f], axis=1)
    return feats, np.array(H), np.array(Y), np.array(P)


def probe_curve(feats, H, Y, P, heights, test_frac: float = 0.2):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    test = (P * 2654435761 % 1000) < test_frac * 1000  # split by problem, deterministic
    out = {}
    for h in heights:
        m = H == h
        tr, te = m & ~test, m & test
        if te.sum() < 200:
            continue
        maj = np.bincount(Y[tr], minlength=10).argmax()
        accs = []
        for sidx in range(feats.shape[0]):
            X = feats[sidx].astype(np.float32)
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[tr]), Y[tr])
            accs.append(float((clf.predict(sc.transform(X[te])) == Y[te]).mean()))
        out[int(h)] = {"acc": accs, "majority": float((Y[te] == maj).mean()), "n_test": int(te.sum())}
    return out


def ready_stage(curve: dict, frac: float = 0.9) -> dict:
    return {h: int(np.argmax(np.array(c["acc"]) >= frac * max(c["acc"]))) for h, c in curve.items()}


def verdict(ready: dict, hs=(1, 2, 3, 4, 5)) -> dict:
    from scipy.stats import spearmanr
    r = [ready[h] for h in hs if h in ready]
    mono = all(b >= a for a, b in zip(r, r[1:]))
    rho = float(spearmanr(range(len(r)), r).correlation) if len(set(r)) > 1 else 0.0
    return {"ready": r, "monotone": mono, "spearman": rho, "supported": bool(mono and rho >= 0.8)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", default="long/c2k8-s0,long/c2k8-s1,long/c16k1-s0,long/c16k1-s1,long/c2k1-s0")
    ap.add_argument("--n_per_depth", type=int, default=800)
    ap.add_argument("--out", default="results/toy/probe/levels.json")
    ap.add_argument("--threads", type=int, default=6, help="CPU threads for torch and the probe solver")
    a = ap.parse_args()
    import warnings
    from threadpoolctl import threadpool_limits
    warnings.filterwarnings("ignore")
    threadpool_limits(limits=a.threads)  # caps OpenBLAS/MKL/OpenMP used by sklearn and numpy
    torch.set_num_threads(a.threads)
    probs = load_problems(a.n_per_depth)
    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {"n_problems": len(probs), "models": {}}
    for name in a.ckpts.split(","):
        if name in res["models"]:
            print(f"{name}: already done, skipping", flush=True)
            continue
        t0 = time.time()
        model, args = load_checkpoint(REPO / f"checkpoints/toy/{name}.pt", "cpu")
        k = train_k(args)
        per_layer = k == 1
        feats, H, Y, P = collect(model, k, per_layer, probs)
        curve = probe_curve(feats, H, Y, P, heights=range(1, 8))
        rd = ready_stage(curve)
        res["models"][name] = {"loops": k, "stages": feats.shape[0], "stage_unit": "layer" if per_layer else "pass",
                               "curve": curve, "ready": rd, "verdict": verdict(rd)}
        print(f"{name}: stages {feats.shape[0]}  ready {rd}  verdict {res['models'][name]['verdict']}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        for h, c in curve.items():
            print(f"   h{h}: " + " ".join(f"{v:.2f}" for v in c["acc"]) + f"   (majority {c['majority']:.2f}, n {c['n_test']})",
                  flush=True)
        out.write_text(json.dumps(res, indent=1))  # save after every model
    print("->", out)


if __name__ == "__main__":
    main()
