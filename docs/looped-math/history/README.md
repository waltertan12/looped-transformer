# How the project's direction changed

Short record of each shift and why. Details are in `../research-log.md` (from 22 Nov) and
`qwen-retrofit/log-2026-11-21.md` (before). Decisions are in `../decisions.md`.

## Phase 1 (2026-11-20 to 11-22): loop a pretrained Qwen model (stopped)

**Hypothesis:** cut a pretrained Qwen3-0.6B down and re-run a block of its layers in a loop
(shared weights) to recover or beat the original's benchmark scores with fewer unique parameters.

**What happened:** looping without training collapsed accuracy (arc_easy 0.617 → 0.357, near the
0.25 chance level). Distillation partly repaired it (exp1). With learned on/off dials, training
switched the loop off by itself (exp2, exp3); forced on, more passes hurt (GSM8K 0.39 → 0.27 from
1 to 4 passes). No recipe beat the no-loop control. Likely causes: no input injection, too little
training (~33M tokens vs the billions used by working looped models), and web-text data that
doesn't reward extra thinking.

**Status:** stopped, not a goal any more (Walter, 2026-11-24). Files: `qwen-retrofit/`.

**Why we moved on:** a pretrained model mixes too many effects to see whether looping adds
*reasoning depth*. We needed a task where the number of reasoning steps is controlled exactly.

The code is still in the repo (`src/looptf/model.py`, `evaluate.py`, `train.py`, `configs/`,
`scripts/eval.py`, `scripts/train.py`). Usage:

```bash
python scripts/eval.py --model Qwen/Qwen3-0.6B-Base --tasks arc_easy,hellaswag --limit 300
python scripts/eval.py --tasks arc_easy,hellaswag --limit 300 --loop 4:16:2:28   # looped, untrained
python scripts/train.py configs/qwen3-0.6b-half-loop2.yaml                       # distillation
```

Loop spec `--loop start:end:n_loops[:coda_start]`: prelude `[0,start)`, loop `[start,end)` × n,
coda `[coda_start, L)`; `0:28:1` is the identity.

## Phase 2 (2026-11-22 onward): tiny models from scratch on arithmetic

**Hypothesis:** with depth (sequential steps) and number size separated, a looped model should
beat a same-weights shallow model on many-step problems and match a deeper model at equal compute.

Shifts within this phase:
- **Mod 7 → real answers.** Modular answers had shortcuts and felt artificial; switched to exact,
  unbounded integers.
- **Curriculum → mixed depths with balanced answer sizes.** The depth curriculum stalled at depth 3,
  and the cause was number size, not steps (D-006, D-008).
- **40k → 120k steps.** All 40k models were still improving; the loop gap survived the longer
  training (D-012).
- **Rules written before results** for every experiment, after a missed shallow baseline (D-010).

Current state and next steps: `../HANDOFF.md`, `../hypotheses.md`, `../plan.md`.
