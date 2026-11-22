# Decisions

One card per decision that changes what an experiment means or what we spend compute on.
Newest first. Written with the `decision-brief` skill. Routine execution (bug fixes, launching
an already-approved run) doesn't need a card.

Card format: **Question** · **Options** (with what each outcome would tell us) ·
**Recommendation** · **Needs Walter?** · **Status** (open / decided: … by whom, when).

---

## D-009 · Add Abacus-style digit-position embeddings? (open, after D-007)

**Question.** McLeish et al. (NeurIPS 2024) found transformers fail at multi-digit arithmetic
mainly because they lose track of each digit's place within its number; an extra embedding for
"place within this number" fixed it, and they used looped transformers with input injection.

**Options.** (a) No change. (b) Add a learned embedding per digit token for its position within
its number (counted from the number's end, so place value lines up), on top of RoPE.

**What would change our minds.** If reversed output (D-007) already fixes 3–5 digit answers,
Abacus may be unnecessary; if big answers still fail after reversing, it's the next fix.

**Recommendation.** Decide after D-007's results. **Needs Walter?** Yes. **Status.** Open.

## D-008 · Balance answer sizes and add an input-size dial? (decided: c)

**Question.** Big answers dominate errors but are rare in training (4-digit answers ≈ 1% of
the depth ≤ 3 curriculum mix; 7+ digits ≈ 0.1% overall), and depth-1 problems can't have big
answers at all (max 81), so "many steps" and "big numbers" always arrive together.

**Options.**
- **(a) Balanced test sets only:** 1,000 test problems per (depth × answer-digit) cell that
  can exist (±3% each). Better measurement, same training.
- **(b) (a) + balanced training:** each answer size gets a fair share within a depth. Tests
  "was it just too few examples?"
- **(c) (b) + input-size dial:** depth 1–2 problems with 2–4 digit inputs (`347*28`), which
  measures big-number arithmetic separately from step count.

**What would change our minds.** If balanced training doesn't lift 4–5 digit accuracy, it's
not a data-quantity problem: it's the arithmetic (→ digit order D-007, or a wider model).

**Recommendation.** (c), then run D-007 on it. **Needs Walter?** Yes. **Status.** Decided: (c), by Walter, 2026-11-22.

## D-007 · Write answers smallest digit first? (decided: run both)

**Question.** At depth 3, accuracy depends mostly on answer size, not depth (deep 16×1:
1 digit 0.99, 2 digits 0.95, 3 digits 0.75, 4 digits 0.40; all models alike). Writing the
biggest digit first means knowing every carry before emitting anything. Is output order the
bottleneck?

**Options.**
- **Keep normal order** (`=258#`). Natural; but depth results stay entangled with
  mental-multiplication difficulty.
- **Reverse the digits** (`=852#`, sign first). How people do it on paper; reported in prior
  small-transformer arithmetic work to help a lot (to verify). Reads oddly.
- **Run both** on 2×8 looped and 16×1 deep, mixed depths 1–8 (4 runs, ~1.5 h).

**What would change our minds.** If reversed is no better on 3–4 digit answers, the limit is
the multiplication itself → wider model, or accept "depth + mental arithmetic" as the task.

**Recommendation.** Run both. **Needs Walter?** Yes. **Status.** Decided: run both, by Walter, 2026-11-22 (`scripts/arith_bal.sh`, results in `results/toy/bal/`).

## D-006 · Fix the curriculum stall at depth 3 (decided: mix all depths)

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

**Needs Walter?** Yes. **Status.** Decided by Walter, 2026-11-22: **not A** (don't lower the
bar; he'd rather raise it to 95%). Instead **mix all training depths (1–8), no curriculum
gate**, and judge per-depth test accuracy with 95% confidence intervals. Evidence found right
after: depth-3 failures are mostly large answers (see D-007), so the stall was number size,
not the curriculum.

## D-005 · Let the remaining 6 grid cells finish? (decided: ran them)

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
