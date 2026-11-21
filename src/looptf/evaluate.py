"""Benchmark harness: wraps a (possibly looped) HF model for lm-evaluation-harness."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lm_eval
from lm_eval.models.huggingface import HFLM

from .model import LoopSpec, apply_loop, describe, load_looped, load_model


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
) -> dict:
    if checkpoint:
        model, tok = load_looped(checkpoint)
    else:
        model, tok = load_model(model_name)
        if loop is not None:
            apply_loop(model, loop)
    info = describe(model)
    print(info)

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

    summary = {
        "model": model_name,
        "checkpoint": checkpoint,
        "loop": loop.__dict__ if loop else None,
        "model_info": info,
        "tasks": tasks,
        "limit": limit,
        "elapsed_s": round(elapsed, 1),
        "results": res["results"],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(checkpoint or model_name).name}_{'loop-' + '-'.join(map(str, loop.__dict__.values())) if loop else 'base'}"
    path = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    print(lm_eval.utils.make_table(res))
    print(f"saved {path}")
    return summary
