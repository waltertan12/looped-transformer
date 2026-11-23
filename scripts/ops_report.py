#!/usr/bin/env python
"""H13 / H14 report from the D-012 checkpoints.

    scripts/ops_report.py --out docs/results-ops.md

H13: accuracy on add-only / sub-only test sets (results/toy/ops/<run>-<kind>/per_example.npz).
H14: accuracy vs share of '*' within cells matched on depth, answer digits, input digits
(results/toy/long/<run>/per_example.npz joined with ops_labels).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ops_labels import labels_for_run  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
MODELS = {"shallow": ("c2k1", 1), "looped": ("c2k8", 8), "deep": ("c16k1", 1)}
SEEDS = (0, 1)


def mul_tercile_gap(correct: np.ndarray, mul_frac: np.ndarray, cell_ids: np.ndarray,
                    min_rows: int = 60) -> tuple[float, int]:
    """Row-weighted mean over cells of acc(lowest third) - acc(highest third), ranking rows by
    mul_frac within each cell (ties broken by a fixed random draw). Cells with < min_rows skipped."""
    tie = np.random.default_rng(0).random(len(mul_frac)) * 1e-6
    num, used = 0.0, 0
    for c in np.unique(cell_ids):
        idx = np.flatnonzero(cell_ids == c)
        if len(idx) < min_rows:
            continue
        order = idx[np.argsort(mul_frac[idx] + tie[idx], kind="stable")]
        t = len(order) // 3
        lo, hi = order[:t], order[-t:]
        num += (correct[lo].mean() - correct[hi].mean()) * 2 * t
        used += 2 * t
    return (num / used if used else float("nan")), used


def h13_verdict(gap_add: list[float], gap_sub: list[float]) -> str:
    g = list(gap_add) + list(gap_sub)
    if any(x >= 0.05 for x in g):
        return "rejected"
    if all(x <= 0.03 for x in g):
        return "supported"
    return "unclear"


def region_acc(d: dict, depth_max: int = 3, digits: tuple[int, ...] = (2, 3)) -> float:
    """Accuracy on unseen prompts (in_train == 0) at depths <= depth_max with the given input sizes:
    the in-distribution, off-the-floor region where training had multi-digit inputs."""
    m = (d["depth"] <= depth_max) & np.isin(d["in_digits"], digits) & (d["in_train"] == 0)
    return float(d["correct"][m].mean())


def _run(model: str, seed: int, rev: bool) -> str:
    return f"{MODELS[model][0]}{'-rev' if rev else ''}-s{seed}"


def _ops(run: str, kind: str) -> dict | None:
    p = REPO / f"results/toy/ops/{run}-{kind}/per_example.npz"
    return dict(np.load(p)) if p.exists() else None


def _cell_mean(d: dict, mask: np.ndarray) -> float:
    """Equal weight per (depth, in_digits) cell."""
    cells = {(int(a), int(b)) for a, b in zip(d["depth"][mask], d["in_digits"][mask])}
    return float(np.mean([d["correct"][mask & (d["depth"] == a) & (d["in_digits"] == b)].mean() for a, b in cells]))


def h13_section(rev: bool) -> tuple[list[str], dict]:
    L, gaps = [], {"add": [], "sub": []}
    for kind in ("add", "sub"):
        meta = json.loads((REPO / f"data/nested_{kind}/meta.json").read_text())
        D = {(m, s): _ops(_run(m, s, rev), kind) for m in MODELS for s in SEEDS}
        if any(v is None for v in D.values()):
            L.append(f"_{kind}: results missing_\n"); continue
        L.append(f"### {kind}-only ({'reversed' if rev else 'normal'} order)\n")
        L.append("Accuracy by input digits (equal weight over depths), seed 0 / seed 1:\n")
        L.append("| inputs | shallow | looped | deep | deep − looped | %Gap |")
        L.append("|---|---|---|---|---|---|")
        for k in (1, 2, 3, 4):
            acc = {ms: _cell_mean(d, d["in_digits"] == k) for ms, d in D.items()}
            f = lambda m: " / ".join(f"{acc[(m, s)]:.2f}" for s in SEEDS)  # noqa: E731
            dl = " / ".join(f"{acc[('deep', s)] - acc[('looped', s)]:+.2f}" for s in SEEDS)
            pg = " / ".join(f"{(acc[('looped', s)] - acc[('shallow', s)]) / (acc[('deep', s)] - acc[('shallow', s)]):.2f}"
                            if abs(acc[('deep', s)] - acc[('shallow', s)]) > 0.02 else "–" for s in SEEDS)
            L.append(f"| {k} digit | {f('shallow')} | {f('looped')} | {f('deep')} | {dl} | {pg} |")
        for s in SEEDS:
            m24 = lambda d: (d["in_digits"] >= 2)  # noqa: E731
            gaps[kind].append(_cell_mean(D[("deep", s)], m24(D[("deep", s)])) -
                              _cell_mean(D[("looped", s)], m24(D[("looped", s)])))
        L.append(f"\n**Rule metric** (deep − looped, inputs 2–4 digits, all depths): "
                 + " / ".join(f"{g:+.3f}" for g in gaps[kind]) + "\n")
        L.append("Off-the-floor view (exploratory, after the rule): unseen prompts only, depths 1–3, 2–3-digit inputs,\n"
                 "where training had multi-digit inputs. seed 0 / seed 1:\n")
        L.append("| shallow | looped | deep | deep − looped |")
        L.append("|---|---|---|---|")
        ra = {ms: region_acc(d) for ms, d in D.items()}
        g = lambda m: " / ".join(f"{ra[(m, s)]:.2f}" for s in SEEDS)  # noqa: E731
        L.append(f"| {g('shallow')} | {g('looped')} | {g('deep')} | "
                 + " / ".join(f"{ra[('deep', s)] - ra[('looped', s)]:+.2f}" for s in SEEDS) + " |\n")
        L.append("Accuracy by depth, 2–4-digit inputs, seed 0 / seed 1 (overlap = share of these 2–4-digit prompts also in training):\n")
        L.append("| depth | overlap | shallow | looped | deep |")
        L.append("|---|---|---|---|---|")
        for lv in sorted(int(x) for x in meta["slices"]):
            sl = meta["slices"][str(lv)]
            n24 = sum(sl["cells"].get(f"in{k}", 0) for k in (2, 3, 4))
            ov = (sum(sl["overlap_by_cell"][f"in{k}"] * sl["cells"].get(f"in{k}", 0) for k in (2, 3, 4)) / n24
                  if "overlap_by_cell" in sl and n24 else float("nan"))
            cells = []
            for m in MODELS:
                vals = []
                for s in SEEDS:
                    d = D[(m, s)]
                    msk = (d["depth"] == lv) & (d["in_digits"] >= 2)
                    vals.append(f"{d['correct'][msk].mean():.2f}" if msk.any() else "–")
                cells.append(" / ".join(vals))
            L.append(f"| {lv} | {ov:.2f} | " + " | ".join(cells) + " |")
        L.append("")
    return L, gaps


def mixed_gap(rev: bool) -> list[float]:
    """deep − looped on nested_bal multi-digit inputs, depths 1–3 (the 'arithmetic' gap)."""
    out = []
    for s in SEEDS:
        acc = {}
        for m in ("looped", "deep"):
            d = np.load(REPO / f"results/toy/long/{_run(m, s, rev)}/per_example.npz")
            msk = (d["K"] == MODELS[m][1]) & (d["op_digits"] >= 2) & (d["depth"] <= 3)
            acc[m] = d["correct"][msk].mean()
        out.append(acc["deep"] - acc["looped"])
    return out


def h14_section() -> tuple[list[str], dict]:
    L = ["Within cells matched on (depth, answer digits, input digits); low − high third of `*` share.",
         "Cells with < 60 rows skipped.\n",
         "| model | seed 0 gap (rows) | seed 1 gap (rows) |", "|---|---|---|"]
    res, sec = {}, {}
    for m, (tag, K) in MODELS.items():
        cells = []
        for s in SEEDS:
            pe = dict(np.load(REPO / f"results/toy/long/{tag}-s{s}/per_example.npz"))
            lab = labels_for_run(REPO / "data/nested_bal", pe)
            k = pe["K"] == K
            cid = (pe["depth"] * 100 + pe["ans_digits"] * 10 + pe["op_digits"])[k]
            g, n = mul_tercile_gap(pe["correct"][k].astype(float), lab["mul_frac"][k], cid)
            res[(m, s)] = g
            cells.append(f"{g:+.3f} ({n:,})")
            sec[(m, s)] = (pe["correct"][k], lab["mul_frac"][k])
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "Secondary: accuracy by overall `*`-share third (not matched), seed 0 / seed 1:\n",
          "| `*` share third | shallow | looped | deep | deep − looped |", "|---|---|---|---|---|"]
    edges = np.quantile(sec[("looped", 0)][1], [1 / 3, 2 / 3])
    for i, name in enumerate(("low", "mid", "high")):
        acc = {}
        for key, (c, mf) in sec.items():
            t = np.digitize(mf, edges)
            acc[key] = c[t == i].mean()
        f = lambda m: " / ".join(f"{acc[(m, s)]:.2f}" for s in SEEDS)  # noqa: E731
        dl = " / ".join(f"{acc[('deep', s)] - acc[('looped', s)]:+.2f}" for s in SEEDS)
        L.append(f"| {name} | {f('shallow')} | {f('looped')} | {f('deep')} | {dl} |")
    return L, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results-ops.md")
    a = ap.parse_args()
    L = ["# H13 / H14 results (D-012 checkpoints, 120k steps, tags on)", "",
         "Generated by `scripts/ops_report.py`. Design: `docs/superpowers/specs/2026-11-23-h13-h14-eval-design.md`.",
         "Each run scored at its training loop count (shallow/deep K=1, looped K=8). Training data has",
         "multi-digit inputs only at depths ≤ 3, so deeper rows with 2–4-digit inputs are out of distribution.", ""]
    h13, gaps = h13_section(rev=False)
    mg = mixed_gap(rev=False)
    v = h13_verdict(gaps["add"], gaps["sub"]) if gaps["add"] and gaps["sub"] else "missing"
    L += ["## H13: do loops match depth on pure addition / subtraction?", "",
          f"**Verdict: {v}.** Rule: supported if deep − looped ≤ 0.03 on add and sub (inputs 2–4 digits, both seeds); "
          f"rejected if any ≥ 0.05. Mixed data (nested_bal, multi-digit inputs, depths 1–3): deep − looped "
          + " / ".join(f"{g:+.3f}" for g in mg) + ".", ""] + h13
    h14, res = h14_section()
    ok = {m: all(res[(m, s)] >= 0.05 for s in SEEDS) for m in MODELS}
    v14 = "supported" if all(ok.values()) else ("partly supported (" + ", ".join(m for m, o in ok.items() if o) + ")"
                                                 if any(ok.values()) else "not supported")
    L += ["## H14: does the operator mix predict errors beyond depth and number size?", "",
          f"**Verdict: {v14}.** Rule: accuracy falls ≥ 0.05 from lowest to highest `*`-share third within "
          "matched cells, both seeds (checked per model).", ""] + h14
    h13r, gr = h13_section(rev=True)
    L += ["", "## Appendix: reversed-order arm (H13 tables)", ""] + h13r
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out.write_text("\n".join(L) + "\n")
    print(f"H13 {v} (add {gaps['add']}, sub {gaps['sub']}, mixed {mg}); H14 {v14} {res}")


if __name__ == "__main__":
    main()
