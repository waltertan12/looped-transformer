#!/usr/bin/env python
"""H16 / D-015 scoring at a checkpoint: does width help numbers, and do loops still add steps?

    scripts/wide_report.py 40k        # uses per_example@40k.npz of every wide run
    scripts/wide_report.py 120k       # uses the final per_example.npz
Measures (exact accuracy on the full held-out set, at each model's training loop count):
  multi-digit inputs: op_digits >= 2, depths 1–3
  big answers: 1-digit inputs, answers 3–5 digits, depths 2–6
  depth 8: 1-digit inputs, depth 8
Rules (D-015, each against width 256 at the same step):
  1. at 512, looped multi-digit >= looped 256 + 0.10
  2. at every width, looped - shallow at depth 8 >= +0.10
  3. (deep - looped) on multi-digit is smaller at 512 than at 256
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODELS = {"shallow": ("c2k1", 1), "looped": ("c2k8", 8), "deep": ("c16k1", 1)}
WIDTHS = (256, 384, 512)


def measures(tag: str, w: int, K: int, at: str) -> dict | None:
    f = REPO / f"results/toy/wide/{tag}-w{w}-s0" / ("per_example.npz" if at == "120k" else f"per_example@{at}.npz")
    if not f.exists():
        return None
    d = np.load(f)
    k = d["K"] == K
    c = d["correct"]
    return {"multi": float(c[k & (d["op_digits"] >= 2) & (d["depth"] <= 3)].mean()),
            "big": float(c[k & (d["op_digits"] == 1) & (d["ans_digits"] >= 3) & (d["ans_digits"] <= 5)
                           & (d["depth"] >= 2) & (d["depth"] <= 6)].mean()),
            "d8": float(c[k & (d["op_digits"] == 1) & (d["depth"] == 8)].mean())}


def main() -> None:
    at = sys.argv[1] if len(sys.argv) > 1 else "40k"
    M = {(m, w): measures(tag, w, K, at) for m, (tag, K) in MODELS.items() for w in WIDTHS}
    print(f"H16 at {at}. Exact accuracy: multi-digit inputs / big answers / depth 8")
    print("model     " + "".join(f"   width {w:<14d}" for w in WIDTHS))
    for m in MODELS:
        cells = [f"{x['multi']:.2f} / {x['big']:.2f} / {x['d8']:.2f}" if (x := M[(m, w)]) else "      missing     " for w in WIDTHS]
        print(f"{m:8s}  " + "   ".join(f"{c:20s}" for c in cells))
    if any(v is None for v in M.values()):
        print("\nNot all runs have this checkpoint yet; rules not scored.")
        return
    lp, dp, sp = (lambda w, m=m: M[(m, w)] for m in ("looped", "deep", "shallow"))
    r1 = lp(512)["multi"] - lp(256)["multi"]
    r2 = {w: lp(w)["d8"] - sp(w)["d8"] for w in WIDTHS}
    g256, g512 = dp(256)["multi"] - lp(256)["multi"], dp(512)["multi"] - lp(512)["multi"]
    print("\nD-015 rules:")
    print(f"  1. looped multi-digit, 512 minus 256: {r1:+.2f} (needs +0.10) -> {'PASS' if r1 >= 0.10 else 'FAIL'}")
    print("  2. looped minus shallow at depth 8: " + ", ".join(f"w{w} {v:+.2f}" for w, v in r2.items())
          + f" (needs +0.10 each) -> {'PASS' if all(v >= 0.10 for v in r2.values()) else 'FAIL'}")
    print(f"  3. deep's lead over looped on multi-digit: 256 {g256:+.2f}, 512 {g512:+.2f} -> "
          f"{'PASS' if g512 < g256 else 'FAIL'} (needs smaller at 512)")
    deep1 = dp(512)["multi"] - dp(256)["multi"]
    print(f"  stop check (40k only): width helps numbers for looped {r1:+.2f}, deep {deep1:+.2f}; "
          f"stop if both are below +0.10 -> {'STOP' if r1 < 0.10 and deep1 < 0.10 else 'continue'}")


if __name__ == "__main__":
    main()
