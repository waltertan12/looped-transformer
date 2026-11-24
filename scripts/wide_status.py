#!/usr/bin/env python
"""H16 progress: latest step per run and held-out accuracy at matching eval steps.

    scripts/wide_status.py
Accuracy = exact answers on held-out problems, mean over depths 1–8 (depth 8 in brackets), at the
run's training loop count, from log.jsonl evals (every 6k steps). Width-512 runs fall back to
their crashed first attempt (results/toy/wide-crash1) until the rerun passes it.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS = [("shallow", "c2k1", "1"), ("looped", "c2k8", "8"), ("deep", "c16k1", "1")]
WIDTHS = [256, 384, 512]
STEPS = [6000, 12000, 18000, 24000, 30000, 36000]


def read(run: str):
    p = REPO / "results/toy" / run / "log.jsonl"
    if not p.exists():
        return None, {}
    last, ev = 0, {}
    for line in p.read_text().splitlines():
        r = json.loads(line)
        last = max(last, r.get("step", 0))
        if "eval" in r:
            ev[r["step"]] = r["eval"]
    return last, ev


def main() -> None:
    print("run                 step    " + "  ".join(f"{s // 1000:>2}k        " for s in STEPS))
    for name, tag, k in MODELS:
        for w in WIDTHS:
            run = f"wide/{tag}-w{w}-s0"
            last, ev = read(run)
            if w == 512:
                l2, e2 = read(f"wide-crash1/{tag}-w{w}-s0")
                if l2 and (not last or l2 > last):
                    last, ev, run = l2, e2, run + "*"
            paused = (REPO / f"checkpoints/toy/wide/{tag}-w{w}-s0@40k.pt").exists()
            cells = []
            for s in STEPS:
                e = ev.get(s, {}).get(k)
                cells.append(f"{sum(e[str(d)] for d in range(1, 9)) / 8:.2f} ({e['8']:.2f})" if e else "    -      ")
            state = "paused@40k" if paused else (f"{last:>6}" if last else "     -")
            print(f"{name:7s} w{w:<4d}  {state:>10s}  " + "  ".join(cells))
    print("* = crashed first attempt (valid to 40k); the rerun replaces it when it gets further")


if __name__ == "__main__":
    main()
