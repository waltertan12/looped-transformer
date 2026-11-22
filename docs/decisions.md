# Decisions

One card per decision that changes what an experiment means or what we spend compute on.
Newest first. Written with the `decision-brief` skill. Routine execution (bug fixes, launching
an already-approved run) doesn't need a card.

Card format: **Question** · **Options** (with what each outcome would tell us) ·
**Recommendation** · **Needs Walter?** · **Status** (open / decided: … by whom, when).

---

## D-006 · Fix the curriculum stall at depth 3 (open)

**Question.** Every grid run plateaus at ~0.85–0.9 on depth-3 training problems and never
reaches the 0.9 needed to add depth 4, so nothing is trained on depths 4–8. How should the
curriculum advance?

**Options.**
- **A. Lower the threshold to 0.8.** Simple; models move on while still imperfect at depth 3.
  If they then master depth 4+, the stall was the threshold, not a capability limit.
- **B. Advance on plateau** (no improvement for N checks). Adapts per model, but "steps to
  master depth d" stops meaning the same thing across models.
- **C. No curriculum after depth 3: train on 1–8 mixed from then on.** Removes the gate
  entirely; risks the original "no foothold" problem at the deep levels.

**What would change our minds.** If even with A no model gets past depth 4, the limit is the
task/model (likely number size), not the curriculum → check answer-length breakdown first.

**Recommendation.** A, after checking `per_example.npz` from the current grid: if depth-3
failures are mostly big answers, the fix is about number size, not the curriculum.

**Needs Walter?** Yes. **Status.** Open.

## D-005 · Let the remaining 6 grid cells finish? (open)

**Question.** Seed 0's other 6 cells (2×4, 4×2, 8×1, 2×2, 4×1, 2×1) will also stall at depth
3. Run them (~1.5 h) or stop and fix the curriculum first?

**Options.** Run: completes the depth-3 comparison along rows/columns (same weights, more
loops; same loops, more weights). Stop: saves 1.5 h, but the depth-3 grid stays half-filled.

**Recommendation.** Run them; the comparison at depth 3 is valid and cheap to finish.

**Needs Walter?** Yes (cheap to override). **Status.** Open; running by default.

## D-004 · Grid design: fixed loops per cell + two random-loop runs (decided)

Fixed K makes each cell compute-matched in training as well as at test time; random-K runs
(2×1–8, 4×1–4) test "more loops at test time". Decided by Claude after Walter proposed the
grid; Walter said to proceed. 2026-11-22.

## D-003 · Real-valued answers, unbounded (decided)

Mod 7 dropped as unnatural; considered a ±1000 bound; Walter chose unbounded real answers
("we're training it to do math"). Answers written as digits + `#`. 2026-11-22.

## D-002 · Tiny models from scratch on synthetic arithmetic (decided)

Qwen retrofit paused: too slow per run, too many causes per failure, data didn't reward
depth. Staged plan: arithmetic → looping on long chains → algebra. Walter, 2026-11-22.

## D-001 · Exp2 dials start at zero (decided, in hindsight wrong)

Chosen by Claude to avoid exp1's broken start; it let the model opt out of the loop, and it
did. Not flagged as a decision at the time, which is why this file exists.
