"""Train and evaluate a tiny (looped) transformer on mod-10 PEMDAS arithmetic.

The model reads `expr=` and must write the answer's digits then the stop token '#'
(e.g. `(8-3)*(4-6)+7=-3#`), with no written-out working. Exact match on the whole answer. Difficulty is tree depth (serial steps). Train on shallow depths, test
on deeper ones, at several loop counts.

    python -m looptf.toy.train --name looped --n_core 2 --train_loops 1-8
    python -m looptf.toy.train --name shallow --n_core 2 --train_loops 1
    python -m looptf.toy.train --name deep16 --n_core 16 --train_loops 1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import numpy as np

from looptf.toy.data import make_batch, VOCAB
from looptf.toy.dataset import FrozenData
from looptf.toy.model import ToyConfig, ToyLM

ROOT = Path(__file__).resolve().parents[3]


def parse_range(s: str) -> tuple[int, int]:
    lo, _, hi = s.partition("-")
    return int(lo), int(hi or lo)


def parse_list(s: str) -> list[int]:
    return [int(v) for v in s.split(",")]


@torch.no_grad()
def build_eval_sets(depths, n_per_depth, n_ops_range, device, seed=1234):
    """Fixed eval sets, identical across runs: one batch per depth."""
    sets = {}
    for d in depths:
        rng = random.Random(seed + d)
        lo = max(d, n_ops_range[0])
        hi = min(n_ops_range[1], 2**d - 1)
        sets[d] = make_batch(n_per_depth, rng, (lo, hi), (d, d), device=device)
    return sets


def answer_loss(logits, b):
    """Cross-entropy over the answer tokens (digits + EOS) only, and per-example exact match.

    Logits at position t predict token t+1; the answer is predicted from positions
    pos .. pos + n_tgt - 1 (pos = the '=' / '?' token). Exact match under teacher forcing
    equals exact match under greedy decoding.
    """
    B, T, V = logits.shape
    t = torch.arange(T - 1, device=logits.device)
    mask = (t[None] >= b["pos"][:, None]) & (t[None] < (b["pos"] + b["n_tgt"])[:, None])
    tgt = b["x"][:, 1:]
    lg = logits[:, :-1][mask].float()
    loss = F.cross_entropy(lg, tgt[mask])
    right = torch.ones(B, T - 1, dtype=torch.bool, device=logits.device)
    right[mask] = lg.argmax(-1) == tgt[mask]
    return loss, right.all(dim=1)


@torch.no_grad()
def dump_per_example(model, eval_sets, loops_list, path):
    """Save per-example correctness with labels, so accuracy can be sliced later
    (e.g. by answer length, to separate 'too many steps' from 'numbers too big')."""
    model.eval()
    rows = {}
    for k in loops_list:
        for d, b in eval_sets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["x"], n_loops=k)
            rows.setdefault("K", []).append(torch.full_like(b["pos"], k).cpu())
            rows.setdefault("correct", []).append(answer_loss(logits, b)[1].cpu())
            for c in ("depth", "chain_len", "n_ops", "n_ops_total", "n_tgt", "answer", "length"):
                if c in b:
                    rows.setdefault(c, []).append(b[c].cpu())
    model.train()
    np.savez_compressed(path, **{c: torch.cat(v).numpy() for c, v in rows.items()})


@torch.no_grad()
def evaluate(model, eval_sets, loops_list):
    model.eval()
    out = {}
    for k in loops_list:
        out[k] = {}
        for d, b in eval_sets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["x"], n_loops=k)
            out[k][d] = answer_loss(logits, b)[1].float().mean().item()
    model.train()
    return out


def fmt_eval(ev, depths):
    head = "K\\depth " + " ".join(f"{d:>4}" for d in depths)
    rows = [head] + [f"{k:>7} " + " ".join(f"{ev[k][d]:4.2f}" for d in depths) for k in ev]
    return "\n".join(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--n_prelude", type=int, default=1)
    p.add_argument("--n_core", type=int, default=2)
    p.add_argument("--n_coda", type=int, default=1)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--inject", default="concat", choices=["concat", "add", "none"])
    p.add_argument("--train_loops", default="1", help="'4' fixed, or '1-8' sampled uniformly per step")
    p.add_argument("--train_depth", default="1-8")
    p.add_argument("--n_ops", default="1-16")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_depths", default="1-16")
    p.add_argument("--eval_loops", default="1,2,4,8,12,16")
    p.add_argument("--eval_n", type=int, default=1000)
    p.add_argument("--curriculum", action="store_true",
                   help="start at depth 1..2, add a level when the current top depth is mastered")
    p.add_argument("--cur_start", type=int, default=2)
    p.add_argument("--cur_every", type=int, default=250)
    p.add_argument("--cur_threshold", type=float, default=0.9)
    p.add_argument("--data", default=None,
                   help="frozen dataset dir (e.g. data/nested, data/chain); default: generate on the fly. "
                        "With --data, 'depth' flags refer to the dataset's level (depth or chain length)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    dev = "cuda"
    out_dir = ROOT / "results" / "toy" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT / "checkpoints" / "toy"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fd = None
    if a.data:
        fd = FrozenData(Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data, device=dev)
        np_rng = np.random.default_rng(a.seed)
    max_len = max(128, fd.meta["width"]) if fd else 128
    cfg = ToyConfig(vocab_size=len(VOCAB), max_len=max_len, d_model=a.d_model, n_heads=a.n_heads, d_ff=a.d_ff,
                    n_prelude=a.n_prelude, n_core=a.n_core, n_coda=a.n_coda, inject=a.inject)
    model = ToyLM(cfg).to(dev)  # fp32 weights; bf16 compute via autocast (exp3 lesson)
    fwd = torch.compile(model) if a.compile else model
    loops_lo, loops_hi = parse_range(a.train_loops)
    depth_rng, nops_rng = parse_range(a.train_depth), parse_range(a.n_ops)
    d_lo, d_hi = parse_range(a.eval_depths)
    eval_depths = list(range(d_lo, d_hi + 1))
    # Non-looped baselines (trained at K=1) are scored at K=1 only; looped models at every K.
    eval_loops = [1] if loops_hi == 1 else sorted(set(parse_list(a.eval_loops)) | {loops_hi})
    if fd:
        eval_sets = fd.test_sets(eval_depths, a.eval_n)
        eval_depths = sorted(eval_sets)
    else:
        eval_sets = build_eval_sets(eval_depths, a.eval_n, nops_rng, dev)
    # Curriculum: train on depths [lo, cur]; when accuracy at depth `cur` (at the largest
    # training loop count) reaches the threshold, raise cur by one, up to the configured max.
    cur = min(a.cur_start, depth_rng[1]) if a.curriculum else depth_rng[1]
    if fd:  # curriculum checks use held-in training rows, never the test set
        cur_sets = {lv: fd.batch(np.random.default_rng(999 + lv), 256, (lv, lv))
                    for lv in range(depth_rng[0], depth_rng[1] + 1)}
    else:
        cur_sets = build_eval_sets(range(depth_rng[0], depth_rng[1] + 1), 256, nops_rng, dev, seed=999)
    mastered_at = {}

    decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
    no_decay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        t = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    meta = {"args": vars(a), "params": model.num_params(), "eval_loops": eval_loops}
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"[{a.name}] unique params {model.num_params()/1e6:.2f}M  train loops {a.train_loops}  "
          f"train depth {a.train_depth}  eval loops {eval_loops}", flush=True)

    log = open(out_dir / "log.jsonl", "w")
    t0, run_loss, run_acc = time.time(), 0.0, 0.0
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, eval_sets, eval_loops)
            log.write(json.dumps({"step": step, "eval": ev}) + "\n"); log.flush()
            print(f"--- step {step} eval (acc; train depth {a.train_depth}) ---\n"
                  + fmt_eval(ev, eval_depths), flush=True)
            if step == a.steps:
                break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        k = rng.randint(loops_lo, loops_hi)
        if a.curriculum and step % a.cur_every == 0 and step > 0 and cur not in mastered_at:
            acc_top = evaluate(model, {cur: cur_sets[cur]}, [loops_hi])[loops_hi][cur]
            if acc_top >= a.cur_threshold:
                mastered_at[cur] = step
                log.write(json.dumps({"step": step, "mastered_depth": cur, "acc": acc_top}) + "\n")
                print(f"*** step {step}: mastered depth {cur} (acc {acc_top:.2f})", flush=True)
                if cur < depth_rng[1]:
                    cur += 1
        if fd:
            b = fd.batch(np_rng, a.bs, (depth_rng[0], cur))
        else:
            b = make_batch(a.bs, rng, nops_rng, (depth_rng[0], cur), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(b["x"], n_loops=k)
        loss, exact = answer_loss(logits, b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()
        run_acc += exact.float().mean().item()
        if (step + 1) % 200 == 0:
            el = time.time() - t0
            rec = {"step": step + 1, "cur_depth": cur, "loss": run_loss / 200, "train_acc": run_acc / 200,
                   "lr": lr_at(step), "elapsed_s": round(el, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step+1:6d}  loss {rec['loss']:.4f}  acc {rec['train_acc']:.3f}  "
                  f"lr {rec['lr']:.2e}  depth<={cur}  {el:.0f}s", flush=True)
            run_loss = run_acc = 0.0

    dump_per_example(model, eval_sets, eval_loops, out_dir / "per_example.npz")
    meta["mastered_at"] = mastered_at
    (out_dir / "config.json").write_text(json.dumps(meta, indent=2))
    ckpt_path = ckpt_dir / f"{a.name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)  # run names may contain '/'
    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "args": vars(a)}, ckpt_path)
    print(f"[{a.name}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
