"""Benchmark harness: wraps a (possibly looped) HF model for lm-evaluation-harness."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM

from .model import LoopSpec, apply_loop, describe, load_looped, load_model, weight_bytes


def run_eval(
    model_name: str,
    tasks: list[str],
    loop: LoopSpec | None = None,
    checkpoint: str | None = None,
    limit: int | None = None,
    batch_size: int | str = "auto",
    num_fewshot: int | None = None,
    out_dir: str = "results",
    apply_chat_template: bool = False,
    n_loops: int | None = None,
) -> dict:
    if checkpoint:
        model, tok = load_looped(checkpoint, n_loops=n_loops)
    else:
        model, tok = load_model(model_name)
        if loop is not None:
            apply_loop(model, loop)
    info = describe(model)
    print(info)
    weight_gb = weight_bytes(model) / 2**30
    torch.cuda.reset_peak_memory_stats()

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=batch_size)
    t0 = time.time()
    res = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        num_fewshot=num_fewshot,
        apply_chat_template=apply_chat_template,
        log_samples=False,
    )
    elapsed = time.time() - t0
    memory = {
        "weights_gb": round(weight_gb, 3),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
    }
    print(f"memory: {memory}")

    summary = {
        "model": model_name,
        "checkpoint": checkpoint,
        "loop": getattr(model.config, "loop_spec", None),
        "model_info": info,
        "tasks": tasks,
        "limit": limit,
        "elapsed_s": round(elapsed, 1),
        "memory": memory,
        "results": res["results"],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        tag = f"{Path(checkpoint).parent.name}-{Path(checkpoint).stem}-K{model.config.loop_spec['n_loops'] if model.config.loop_spec else 'na'}"
    else:
        tag = f"{Path(model_name).name}_{'loop-' + loop.tag() if loop else 'base'}"
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(lm_eval.utils.make_table(res))
    print(f"saved {path}")
    return summary
