# Housekeeping: repo structure and tech debt

Cross-project to-do list for the repo itself: structure, paths, tooling, the harness. Research
state does not go here. It goes in each project's `HANDOFF.md`.

Add an item when you notice debt. Remove it when it is fixed, and say so in the commit message.

## Open

- **Report scripts write to the old flat docs paths** (since the 2026-11-24 docs reorganization).
  `scripts/arith_report.py`, `scripts/ops_report.py`, `scripts/tags_report.py` default to
  `docs/results-*.md`. `scripts/probe_report.py` writes `docs/results-h19.md` and has no `--out` flag.
  Fix: change the defaults to `docs/looped-math/results/<name>.md`. Until then, pass `--out`.
  Docstrings in `scripts/` also mention the old paths.
- **`docs/visual/` belongs to looped-math** but sits at the top of `docs/`. `scripts/curves_export.py`,
  `scripts/breakdown_export.py` and `scripts/attn_export.py` write into it. Fix: move it to
  `docs/looped-math/visual/` and update the 3 scripts and the links. Do this when no chain runs.
- **`scripts/arith_wide_chain.sh` wait loop can match itself.** `pgrep -f "pause_at 40000"` matches
  any shell whose command line contains that text. It hung the chain on 2026-11-24. Fix it after
  the chain exits. Use the same care in future chain scripts.
- **Code is not split by project.** `src/looptf/` holds both the stopped Qwen retrofit code
  (`model.py`, `train.py`, `evaluate.py`) and the active `toy/` package. `scripts/` and `tests/` mix
  both too. Decide later whether to split. Do not move code while trainings run.
- **Public site URLs changed** with the reorganization. Old links to moved pages are broken.
  Optional fix: redirect stubs at the old paths.
- **The two skills assume looped-math.** `status-update` and `decision-brief` write to
  `docs/looped-math/`. Make them take a project when a second project becomes active.
- **Visual pages on the public site lack a doctype.** They are written for the claude.ai artifact
  viewer, which adds the skeleton. GitHub Pages serves them as-is, so browsers use quirks mode.
  `p0-amino-acids.html` adds its own charset and viewport tags; the three looped-math pages do not.
  The training-curves page also fetches `curves/*.json`, which is gitignored, so it shows no charts on
  the public site.
