#!/usr/bin/env python
"""Export training curves for docs/visual/training-curves.html.

Reads results/toy/<group>/<run>/log.jsonl for the main run groups and writes one JSON file with,
per run: training loss and learning rate every 200 steps, and held-out exact accuracy at each
eval (mean over depths 1–8, and depth 8 alone) at the run's training loop count.

    scripts/curves_export.py            # -> docs/visual/curves/curves.json, prints a convergence summary
Re-run while H16 trains to refresh its (partial) curves.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GROUPS = {
    "long": "120k steps, tags, 2 seeds (D-012)",
    "bal": "40k steps, no tags (D-010)",
    "place": "40k steps, tags (D-011)",
    "tags": "40k steps, tag fixes (D-014)",
    "wide": "H16: widths 256 / 384 / 512, no tags, 120k schedule (D-015)",
    "wide-crash1": "H16 width-512 first attempt (crashed at the 40k pause, curves valid to 40k)",
}
MODEL = {"c2k1": "shallow", "c2k8": "looped", "c16k1": "deep"}


def read_run(d: Path) -> dict | None:
    log = d / "log.jsonl"
    if not log.exists():
        return None
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    args = cfg.get("args", {})
    k = str(int(str(args.get("train_loops", "1")).split("-")[-1]))
    train, ev = [], []
    for line in log.read_text().splitlines():
        r = json.loads(line)
        if "loss" in r:
            train.append([r["step"], round(r["loss"], 4), round(r["train_acc"], 4), r["lr"]])
        elif "eval" in r:
            e = r["eval"].get(k) or next(iter(r["eval"].values()))
            accs = [e[str(dd)] for dd in range(1, 9) if str(dd) in e]
            ev.append([r["step"], round(sum(accs) / len(accs), 4), e.get("8")])
    name = d.name
    return {"run": f"{d.parent.name}/{name}", "model": MODEL.get(name.split("-")[0], name),
            "width": int(args.get("d_model", 256)), "steps": int(args.get("steps", 0)),
            "seed": int(args.get("seed", 0)), "variant": "-".join(name.split("-")[1:-1]) or "normal",
            "train": train, "eval": ev}


def main() -> None:
    runs = []
    for g in GROUPS:
        for d in sorted((REPO / "results/toy" / g).glob("*")):
            if d.is_dir() and (r := read_run(d)):
                r["group"] = g
                runs.append(r)
    out = REPO / "docs/visual/curves/curves.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"groups": GROUPS, "runs": runs}, separators=(",", ":")))
    print(f"{len(runs)} runs -> {out} ({out.stat().st_size / 1e3:.0f} KB)")
    print("\nConvergence check on the 120k runs (normal order): loss and eval accuracy, 80k vs 120k")
    for r in runs:
        if r["group"] != "long" or "rev" in r["run"]:
            continue
        L = {s: l for s, l, *_ in r["train"]}
        E = {s: a for s, a, _ in r["eval"]}
        l80 = sum(L[s] for s in L if 78000 < s <= 80000) / 10
        l120 = sum(L[s] for s in L if 118000 < s <= 120000) / 10
        e80 = E.get(78000) or E.get(84000)
        print(f"  {r['run']:16s} loss {l80:.3f} -> {l120:.3f}   eval d1-8 {e80:.3f} -> {E[120000]:.3f}")


if __name__ == "__main__":
    main()
