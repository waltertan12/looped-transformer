#!/usr/bin/env python
"""Write docs/results-h19.md from results/toy/probe/levels.json (scripts/probe_levels.py)."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAMES = {"long/c2k8-s0": "Looped, run 1", "long/c2k8-s1": "Looped, run 2", "long/c16k1-s0": "Deep, run 1",
         "long/c16k1-s1": "Deep, run 2", "long/c2k1-s0": "Shallow, run 1"}


def main() -> None:
    res = json.loads((REPO / "results/toy/probe/levels.json").read_text())
    L = ["# H19 results: does each loop pass handle one level of brackets?", "",
         "**Question.** The looped model runs the same layers 8 times. Does each pass work through one",
         "level of a nested problem, innermost brackets first?", "",
         "**Method.** For every bracketed piece of a test problem (its *level* is how deeply it nests:",
         "`8-3` is level 1, `(8-3)*(4-6)` is level 2), a probe (a tiny linear classifier) tries to read",
         "the last digit of the piece's value from the model's internal state, at the token right after",
         "the piece ends. We train one probe per pass (looped) or per layer (deep, shallow). If passes",
         "handle levels, level 1 should become readable early, level 2 later, and so on: a staircase.",
         f"{res['n_problems']:,} test problems with 1-digit inputs, depths 2–8; 20% held out to score the probes.",
         "Decision card D-013; code `scripts/probe_levels.py`.", "",
         "## Result", "",
         "- **No staircase of one level per pass.** Levels 2 and 3 become readable together and gradually.",
         "- **Only levels 1–3 can be read at all.** Levels 4 and up stay at guessing level in every model,",
         "  even though looped answers depth-4 problems correctly 96% of the time. The intermediate",
         "  values aren't stored as readable digits where each piece ends.",
         "- **Level 1 is done before the loop starts** (by the single layer in front of it) and fades over",
         "  the passes.",
         "- **Deep shows the same order over its layers**, so this isn't specific to looping.",
         "- By the pre-set rule H19 is *not supported*, but the rule counted unreadable levels 4–5, so the",
         "  honest summary is *not shown*: this probe can't see past level 3.", "",
         "Possible next probes: read every piece's value at the final `=`, use a small non-linear probe,",
         "or study the per-pass attention maps saved in `results/toy/probe/attn_*.json`.", "",
         "## Probe accuracy by level and pass", "",
         "Each cell is the share of held-out pieces whose last digit the probe gets right. *Guess* is",
         "always picking the most common digit. Column 0 is before the first pass (or first core layer).", ""]
    for key, name in NAMES.items():
        m = res["models"].get(key)
        if not m:
            continue
        unit = m["stage_unit"]
        n = m["stages"]
        cols = list(range(n)) if n <= 9 else list(range(0, n, 2))
        L += [f"### {name} ({'8 passes' if unit == 'pass' else f'{n - 1} core layers'})", "",
              "| level | " + " | ".join(f"{unit} {c}" for c in cols) + " | best | guess |",
              "|---|" + "---|" * (len(cols) + 2)]
        for h in sorted(m["curve"], key=int):
            if int(h) > 5:
                continue
            c = m["curve"][h]
            acc = c["acc"]
            best = max(acc)
            cells = [f"**{acc[i]:.2f}**" if acc[i] == best else f"{acc[i]:.2f}" for i in cols]
            L.append(f"| {h} | " + " | ".join(cells) + f" | {best:.2f} | {c['majority']:.2f} |")
        L.append("")
    out = REPO / "docs/results-h19.md"
    out.write_text("\n".join(L) + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
