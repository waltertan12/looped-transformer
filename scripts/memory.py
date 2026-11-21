#!/usr/bin/env python
"""Measure memory and speed at a fixed workload, so numbers are comparable across models.

Workload: batch 1, a PROMPT-token prompt, generate NEW tokens greedily.
Reports weight memory, KV-cache memory (theoretical), peak VRAM, and tokens/s.

  python scripts/memory.py --model Qwen/Qwen3-1.7B-Base
  python scripts/memory.py --checkpoint checkpoints/x/step1000.pt --n-loops 4
"""
import argparse
import json
import time
from pathlib import Path

import torch

from looptf.model import load_looped, load_model, weight_bytes

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
p.add_argument("--checkpoint", default=None)
p.add_argument("--n-loops", type=int, default=None)
p.add_argument("--prompt", type=int, default=2048)
p.add_argument("--new", type=int, default=256)
p.add_argument("--out", default="results/memory.jsonl")
a = p.parse_args()

if a.checkpoint:
    model, tok = load_looped(a.checkpoint, n_loops=a.n_loops)
else:
    model, tok = load_model(a.model)
model.eval()
cfg = model.config

torch.manual_seed(0)
x = torch.randint(1000, 20000, (1, a.prompt), device="cuda")
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
base_alloc = torch.cuda.memory_allocated()
t0 = time.time()
with torch.no_grad():
    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=a.new, min_new_tokens=a.new, do_sample=False)
torch.cuda.synchronize()
dt = time.time() - t0

kv_bytes = (
    2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.head_dim * (a.prompt + a.new) * 2
)  # K and V, bf16
row = {
    "model": a.checkpoint or a.model,
    "n_loops": (cfg.loop_spec or {}).get("n_loops") if getattr(cfg, "loop_spec", None) else None,
    "depth": cfg.num_hidden_layers,
    "weights_gb": round(weight_bytes(model) / 2**30, 3),
    "kv_cache_gb": round(kv_bytes / 2**30, 3),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    "peak_over_weights_gb": round((torch.cuda.max_memory_allocated() - base_alloc) / 2**30, 3),
    "gen_tok_per_s": round(a.new / dt, 1),
    "prompt": a.prompt,
    "new": a.new,
}
print(json.dumps(row))
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "a") as f:
    f.write(json.dumps(row) + "\n")
