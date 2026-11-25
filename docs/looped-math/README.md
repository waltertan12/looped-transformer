# Looped math: do looped transformers add reasoning depth?

**Intent.** Find out, with evidence, whether running the same transformer layers several times
(a looped transformer) adds reasoning depth. Goal: a mini paper with an evidence-backed opinion.

**How.** Tiny transformers trained from scratch on synthetic nested PEMDAS arithmetic, where the
number of reasoning steps (depth) and the number size are separate, exact dials. Three models:
shallow 2×1 (4.4M), looped 2×8 (4.4M, 8 passes), deep 16×1 (19M, same compute as looped).

**Status.** Active. Current question: H16, do wider blocks fix number size (D-015)? What is
running now: `HANDOFF.md`. Main results so far: `hypotheses.md` (Done table).

## Files

| file | holds |
|---|---|
| `HANDOFF.md` | start here: what is running, how to resume, next actions |
| `hypotheses.md` | every hypothesis, its test and verdict |
| `plan.md` | next steps with decision rules written before the results |
| `decisions.md` | decision cards D-001 onward |
| `ideas.md` | idea backlog |
| `research-log.md` | reasoning and results in time order, from 22 Nov 2026 |
| `status.md` | dated snapshots, newest first |
| `results/` | `bal.md` (D-007/D-010), `grid.md` (first grid), `ops.md` (H13/H14), `tags.md` (H18), `h19.md` (H19 probes) |
| `plans/specs/`, `plans/impl/` | experiment designs (human) and build plans (agent), from the superpowers skills |
| `prior-art/related-work.md` | the closest papers, compared with our results |
| `history/` | how the direction changed; `qwen-retrofit/` is the first, stopped approach |

Visual pages live in `docs/visual/` (results page, attention viewer, training curves) and are
published as claude.ai artifacts. Links are in `HANDOFF.md`.

## Code and data (not under docs/)

Code `src/looptf/toy/`, scripts `scripts/arith_*.sh` and friends, tests `tests/test_toy_*.py`.
Run outputs `results/toy/`, checkpoints `checkpoints/toy/`, data `data/` (all gitignored).

## Known path debt (docs moved on 2026-11-24, scripts not yet updated)

Some report scripts still write to the old flat paths by default. When you run them, pass the
new path:
- `scripts/arith_report.py --out docs/looped-math/results/grid.md`
- `scripts/ops_report.py --out docs/looped-math/results/ops.md`
- `scripts/tags_report.py --out docs/looped-math/results/tags.md`
- `scripts/probe_report.py` has no `--out` flag. It writes `docs/results-h19.md`. Move the output to
  `docs/looped-math/results/h19.md`.

`docs/visual/` stays where it is, because `scripts/curves_export.py`, `scripts/breakdown_export.py`
and `scripts/attn_export.py` write into it.
