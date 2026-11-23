# Plan

The forward-looking plan, with a decision rule for each step written **before** its results
come in. Reasoning lives in `research-log.md`, choices in `decisions.md`, state in `status.md`.
Revise this file when a step finishes or the plan changes, and say why.

_Last revised 2026-11-22 17:15._

## The question

Does looping add reasoning depth? Concretely: with the same weights, do more loop passes solve
problems that need more sequential steps, and does a looped model match a deeper one with fewer
weights? Arithmetic is the test bed because difficulty is controllable and answers are
checkable.

**Current obstacle:** errors are dominated by big-number arithmetic (multi-digit answers and
inputs), not by step count. Until that is handled, the depth effect is hidden. Steps 1–2 fix
the arithmetic; steps 3–5 measure looping.

## Short term (today / tomorrow)

### 1. Digit order (D-007) — running
Probe set (looped 2×8, shallow 2×1, deep 16×1) × normal / reversed on `nested_bal`.

**Primary metric** (fixed now, before the results): exact-match accuracy at each run's training
loop count on 1-digit-input problems with **3–5 digit answers, depths 2–6**: the cells where
the grid showed normal order failing.

**Decision rule.** Adopt reversed order as the default if, on the looped model, it improves the
primary metric by **≥ 0.05** (beyond the ±0.03 noise) **and** costs no more than 0.03 on 1–2
digit answers. It does not need to win every cell. If the gain is < 0.05, keep normal order
(more natural) and treat order as irrelevant.

**Secondary (the looping question):** the looped − shallow gap under each order. If reversing
widens the gap, the loop benefits from the cleaner arithmetic; if it narrows it, reversal
mostly helps the no-loop model.

### 2. Place-value embeddings (D-009 / D-011) — running
Walter's design: screen on shallow 2×1 + looped 2×8 only, tags with normal and with reversed
answers (4 runs). Thresholds below are for direction, not proof; the second seed in step 4
confirms. Seed-to-seed noise is still unmeasured, which is why 0.05, not 0.03, is the bar.

**Earlier version of this step:**
Probe set with the order chosen in step 1, `--place_emb` on (the off condition already exists
from step 1). 3 runs, ≈ 1.3 h.

**Decision rule.** Adopt if it improves the primary metric by ≥ 0.05 on the looped model
without hurting 1–2 digit answers by more than 0.03.

### 3. Second seed for whatever was decisive
Re-run the cells that decided steps 1–2 with seed 1 (≈ 1 h). Keep a result only if it holds
on both seeds.

## Medium term (this week)

### 3b. Long probe set (D-012): the checkpoint
Shallow / looped / deep × 2 seeds, 120k steps, recipe from the D-011 screen by D-012's rule.
≈ 6 h. A good stopping point: the arithmetic result is then checked on 2 seeds with fully
trained models. Then update the visual report.

### 4. The recipe on the full grid, trained longer
All D-007 models were still improving at 40k steps (loss falling steadily; train ≈ test
accuracy, i.e. undertrained), so the full-matrix runs use more steps (≈ 3×).

**Earlier version of this step:**
Core {2,4,8,16} × loops, with the settled recipe (order + embeddings), 2 seeds (≈ 6 h, overnight).
**Question:** with the arithmetic handled, does the looped-vs-deep gap and the gain from loops
grow with depth? **Success:** looped 2×8 within 0.03 of deep 16×1 at depths 4–8 with 4× fewer
weights, and loops helping more at deeper depths than shallow ones.

### 5. More loops at test time (extrapolation)
Train on depths ≤ 8 with a **random** loop count, test depths 9–16 at more loops than trained.
Fixed-loop models can't do this (they only work at their training loop count). Random loops
cost accuracy in the first grid, so this step also tries to fix that (e.g. bias the random K
toward the maximum). **Success:** accuracy at depth 9–16 rises with test-time loops.

### 6. Loop placement
Fixed 4 unique blocks, same compute: 0+4+0, 1+2+1, 2+1+1, 1+1+2, 3+1+0. **Question:** which
blocks should recur?

### 7. Chain task (intermediate values)
Run the recipe on `data/chain`. **Question:** does looping help with following references,
not just nested evaluation?

## Later

- **Stage 3: algebra** (does step-wise power transfer to solving `3x+5=20`?). New generator.
- **Adaptive exit** (fewer loops on easy problems), once step 5 works.
- **Number base 2 / 10 / 16** (many easy steps vs few hard ones).
- **Side quest:** hidden-state / embedding space per loop pass.

## Not doing (and why)
- Full grid for every fix: ≈ 4× the cost of the probe set (D-010).
- Back to Qwen: only once a from-scratch recipe clearly works (D-002).
