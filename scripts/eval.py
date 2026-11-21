#!/usr/bin/env python
"""Run a benchmark on a base or looped Qwen model.

Examples:
  python scripts/eval.py --model Qwen/Qwen3-0.6B-Base --tasks arc_easy --limit 200
  python scripts/eval.py --model Qwen/Qwen3-0.6B-Base --tasks arc_easy,hellaswag --loop 4:16:2:16
  python scripts/eval.py --checkpoint checkpoints/run/step2000.pt --tasks gsm8k
"""
import argparse

from looptf.evaluate import run_eval
from looptf.model import LoopSpec

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
p.add_argument("--tasks", default="arc_easy", help="comma-separated lm-eval task names")
p.add_argument("--loop", default=None, help="start:end:n_loops[:coda_start]")
p.add_argument("--gated", action="store_true", help="gated extra passes (start as no-ops)")
p.add_argument("--n-loops", type=int, default=None, help="override loop passes on a checkpoint")
p.add_argument("--checkpoint", default=None, help="looped checkpoint from scripts/train.py")
p.add_argument("--limit", type=int, default=None, help="max examples per task (for quick runs)")
p.add_argument("--num-fewshot", type=int, default=None)
p.add_argument("--batch-size", default="auto")
p.add_argument("--chat", action="store_true", help="apply chat template (instruct models)")
p.add_argument("--out", default="results")
a = p.parse_args()

bs = a.batch_size if a.batch_size == "auto" else int(a.batch_size)
run_eval(
    a.model,
    a.tasks.split(","),
    loop=LoopSpec.parse(a.loop, gated=a.gated) if a.loop else None,
    checkpoint=a.checkpoint,
    limit=a.limit,
    batch_size=bs,
    num_fewshot=a.num_fewshot,
    out_dir=a.out,
    apply_chat_template=a.chat,
    n_loops=a.n_loops,
)
