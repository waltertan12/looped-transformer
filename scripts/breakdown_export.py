#!/usr/bin/env python
"""Test-set accuracy breakdowns at checkpoints, for docs/visual/training-curves.html.

For every per-example dump (H16 runs at @40k / @80k / final, old 120k and 40k runs at their end),
exact accuracy at the run's training loop count, broken down by depth, input digits, answer
digits, share of operators that are × (from the prompt), and, where scored, the add-only and
subtract-only test sets by depth (1-digit inputs, problems not in training; scripts/wide_ops.sh).

    scripts/breakdown_export.py       # -> docs/visual/curves/breakdown.json (local only, gitignored)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops_labels import labels_for_run  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODEL = {"c2k1": ("shallow", 1), "c2k8": ("looped", 8), "c16k1": ("deep", 1)}
MUL = ["no ×", "under 1/3 ×", "1/3 to 2/3 ×", "over 2/3 ×"]
MIN_N = 50


def mul_class(lab: dict) -> np.ndarray:
    f = lab["mul_frac"]
    return np.where(lab["n_mul"] == 0, 0, np.where(f <= 1 / 3, 1, np.where(f <= 2 / 3, 2, 3)))


def pure_ops(path: Path, K: int):
    if not path.exists():
        return None
    d = np.load(path)
    sel = (d["K"] == K) & (d["in_digits"] == 1) & (d["in_train"] == 0)
    return cut(d["correct"][sel], d["depth"][sel], range(1, 16))


def cut(correct, key, values):
    out = []
    for v in values:
        sel = key == v
        n = int(sel.sum())
        out.append([float(correct[sel].mean()) if n >= MIN_N else None, n])
    return out


def dumps():
    for d in sorted((REPO / "results/toy/wide").glob("*-s0")):
        tag = d.name.split("-")[0]
        w = int(d.name.split("-w")[1].split("-")[0])
        for f, at in ((d / "per_example@40k.npz", "40k"), (d / "per_example@80k.npz", "80k"), (d / "per_example.npz", "120k")):
            if f.exists():
                ops = REPO / f"results/toy/wide-ops/{d.name}@{at}"
                yield f"H16 width {w}", at, tag, w, f, ops
    for g, label in (("long", "Main runs, 120k, tags"), ("bal", "Short runs, 40k, no tags")):
        for d in sorted((REPO / "results/toy" / g).glob("*-s0")):
            tag = d.name.split("-")[0]
            if "rev" in d.name or tag not in MODEL or not (d / "per_example.npz").exists():
                continue
            yield label, "end", tag, 256, d / "per_example.npz", REPO / f"results/toy/ops/{d.name}" if g == "long" else None


def main() -> None:
    cache: dict[int, np.ndarray] = {}
    out = []
    for group, at, tag, w, f, ops_root in dumps():
        model, K = MODEL[tag]
        pe = dict(np.load(f))
        n = len(pe["answer"])
        if n not in cache:
            cache[n] = mul_class(labels_for_run(REPO / "data/nested_bal", pe))
        ops = cache[n]
        k = pe["K"] == K
        c, depth, ind, ansd, op = (pe["correct"][k], pe["depth"][k], pe["op_digits"][k], pe["ans_digits"][k], ops[k])
        one = ind == 1
        out.append({"group": group, "at": at, "model": model, "width": w,
                    "depth": cut(c[one], depth[one], range(1, 17)),
                    "in_digits": cut(c[depth <= 3], ind[depth <= 3], [1, 2, 3]),
                    "ans_digits": cut(c[one & (depth >= 2) & (depth <= 8)], ansd[one & (depth >= 2) & (depth <= 8)], range(1, 9)),
                    "mul": cut(c[one & (depth >= 2) & (depth <= 8)], op[one & (depth >= 2) & (depth <= 8)], range(len(MUL))),
                    "add_only": pure_ops(Path(f"{ops_root}-add/per_example.npz"), K) if ops_root else None,
                    "sub_only": pure_ops(Path(f"{ops_root}-sub/per_example.npz"), K) if ops_root else None})
        print(f"{group:26s} {at:5s} {model:8s} overall {float(c.mean()):.2f}")
    p = REPO / "docs/visual/curves/breakdown.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"mul": MUL, "min_n": MIN_N, "rows": out}, separators=(",", ":")))
    print(f"{len(out)} breakdowns -> {p}")


if __name__ == "__main__":
    main()
