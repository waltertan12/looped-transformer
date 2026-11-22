"""Train and evaluate a tiny (looped) transformer on mod-10 PEMDAS arithmetic.

The model reads `expr=` and must predict the answer digit at the '=' position, with no
written-out working. Difficulty is tree depth (serial steps). Train on shallow depths, test
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

from looptf.toy.data import make_batch, VOCAB
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


@torch.no_grad()
def evaluate(model, eval_sets, loops_list):
    model.eval()
    out = {}
    for k in loops_list:
        out[k] = {}
        for d, b in eval_sets.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["x"], n_loops=k)
            pred = logits[torch.arange(len(b["pos"])), b["pos"]].argmax(-1)
            out[k][d] = (pred == b["y"]).float().mean().item()
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
    p.add_argument("--eval_n", type=int, default=512)
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

    cfg = ToyConfig(vocab_size=len(VOCAB), d_model=a.d_model, n_heads=a.n_heads, d_ff=a.d_ff,
                    n_prelude=a.n_prelude, n_core=a.n_core, n_coda=a.n_coda, inject=a.inject)
    model = ToyLM(cfg).to(dev)  # fp32 weights; bf16 compute via autocast (exp3 lesson)
    fwd = torch.compile(model) if a.compile else model
    loops_lo, loops_hi = parse_range(a.train_loops)
    depth_rng, nops_rng = parse_range(a.train_depth), parse_range(a.n_ops)
    d_lo, d_hi = parse_range(a.eval_depths)
    eval_depths = list(range(d_lo, d_hi + 1))
    # Non-looped baselines (trained at K=1) are scored at K=1 only; looped models at every K.
    eval_loops = [1] if loops_hi == 1 else sorted(set(parse_list(a.eval_loops)) | {loops_hi})
    eval_sets = build_eval_sets(eval_depths, a.eval_n, nops_rng, dev)

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
        b = make_batch(a.bs, rng, nops_rng, depth_rng, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(b["x"], n_loops=k)
        logits = logits[torch.arange(a.bs, device=dev), b["pos"]].float()
        loss = F.cross_entropy(logits, b["y"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += loss.item()
        run_acc += (logits.argmax(-1) == b["y"]).float().mean().item()
        if (step + 1) % 200 == 0:
            el = time.time() - t0
            rec = {"step": step + 1, "loss": run_loss / 200, "train_acc": run_acc / 200,
                   "lr": lr_at(step), "elapsed_s": round(el, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(f"step {step+1:6d}  loss {rec['loss']:.4f}  acc {rec['train_acc']:.3f}  "
                  f"lr {rec['lr']:.2e}  {el:.0f}s", flush=True)
            run_loss = run_acc = 0.0

    torch.save({"cfg": cfg.__dict__, "state_dict": model.state_dict(), "args": vars(a)},
               ckpt_dir / f"{a.name}.pt")
    print(f"[{a.name}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
