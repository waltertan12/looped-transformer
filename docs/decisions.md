# Decisions

One card per decision that changes what an experiment means or what we spend compute on.
Newest first. Written with the `decision-brief` skill. Routine execution (bug fixes, launching
an already-approved run) doesn't need a card.

Card format: **Question** · **Options** (with what each outcome would tell us) ·
**Recommendation** · **Needs Walter?** · **Status** (open / decided: … by whom, when).

---

## D-012 · Long probe set: the checkpoint run (decided, recipe pending the D-011 screen)

**Question.** Does the looping result hold with fully trained models and on a second seed?
All D-007 models were still improving at 40k steps, and everything so far is one seed.

**Runs.** Shallow 2×1, looped 2×8, deep 16×1 × seeds 0 and 1 = 6 runs, **120k steps** (3×),
`data/nested_bal`, lr 3e-4, mixed depths 1–8. Two at a time, paired by model (both seeds
together): shallow ≈ 45 min, looped ≈ 2.5 h, deep ≈ 2.5 h → **≈ 6 h** total.

**Recipe rule (written before the looped D-011 results):**
- Place-value tags **on** (they help on multi-digit inputs for shallow; no cost elsewhere).
- Answer order: **reversed** if looped + tags + reversed beats looped + tags + normal by
  ≥ 0.05 on multi-digit-input problems (depths 1–3) **and** is no more than 0.03 worse on
  1-digit-input problems (big and small answers); otherwise **normal** (simpler, and better on
  multi-step problems so far).

**What would change our minds.** If shallow catches up with looped at 120k steps, the loop's
advantage was speed of learning, not capacity. If the two seeds differ by more than the
looped − shallow gap, the gap isn't a result yet.

**Stopping point.** After this, the arithmetic question has a checked answer: write it up and
pause, or go on to test-time loops (plan step 5).

**Needs Walter?** Decided by Walter, 2026-11-22 ("do the probe set long, then update the HTML").
Recipe chosen by the rule above. **Status.** Launches after the D-011 screen (~21:35).

## D-011 · Place-value embeddings: which runs? (decided: screen on shallow + looped)

**Question.** Plan step 2. Looping doesn't fix big answers (+0.03), so number representation
is the next target. The plan says test place-value embeddings with the winning order (normal).
But place-value is expected to pair with reversed order (in reversed order the answer's first
digit is the ones place, so tags line up), so reversed may only have lost because it lacked them.

**Options.**
- **(a) Normal + place-value on the probe set** (shallow, looped, deep): 3 runs, ≈ 1.3 h. The
  plan's rule as written.
- **(b) (a) + looped reversed + place-value:** 4 runs, ≈ 1.6 h. Also tests the pairing
  hypothesis on the model we care most about.
- **(c) Both orders on the full probe set:** 6 runs, ≈ 2.5 h.

**What would change our minds.** If place-value lifts 3–5 digit answers by ≥ 0.05 → adopt
(plan rule). If reversed + place-value beats normal + place-value by ≥ 0.05 → revisit D-007.
If place-value doesn't help either order → number size needs width/training length instead.

**Recommendation.** (b). **Needs Walter?** Yes. **Status.** Decided by Walter, 2026-11-22: a
directional screen on **shallow 2×1 and looped 2×8 only**, each with tags + normal and tags +
reversed (4 runs, 40k steps, seed 0; `scripts/arith_place.sh`, results in `results/toy/place/`).
No-tag baselines reused from D-007. Rules: tags help if big answers (3–5 digits, depths 2–6)
rise ≥ 0.05 and small answers drop ≤ 0.03; reversed + tags worth it if ≥ 0.05 over normal +
tags. Winners then go to longer training on the full matrix, with a second seed to confirm.
**Shallow result:** fails the rule as written (big −0.01, small +0.02), but the rule's metric
used 1-digit inputs, where tags carry ~no information (Claude's design mistake). Exploratory:
tags + reversed lifts depth-1 3-digit-input problems 0.14 → 0.43 (3-digit answers 0.38 → 0.89);
tags + normal best at depth 2 (0.35 → 0.43). **Amended before the looped results:** multi-digit-
input problems are a co-primary metric (≥ 0.05 bar).

## D-010 · How much of the grid does each new fix get tested on? (decided: probe set)

**Question.** Walter: shouldn't digit order and place-value embeddings also be tested on the
shallow/medium models, i.e. isn't every test a grid? Without the no-loop baseline we can't tell
"this fix helps looping" from "this fix helps every model". The D-007 runs lack shallow.

**Options.**
- **Full grid per fix:** 10 cells × each condition. Order × place-value (4 conditions) = 40
  runs ≈ 11 GPU-hours per seed.
- **Probe set per fix, full grid for the final recipe:** shallow 2×1, looped 2×8, deep 16×1
  for every condition (12 runs ≈ 2.5 h per seed), then one full grid with the best recipe.
- **Looped + deep only** (current D-007 runs): cheapest, but no no-loop baseline.

**What would change our minds.** If the probe set shows a fix helps looped *more* than shallow
or deep, it's worth confirming on the full grid; if it helps all three equally, it's a general
arithmetic fix and the grid adds little.

**Recommendation.** Probe set (3 corners) for each fix; full grid for the final recipe. Add
shallow normal + reversed to the current D-007 queue now (~15 min). **Needs Walter?** Yes.
**Status.** Decided by Walter, 2026-11-22: probe set (shallow 2×1, looped 2×8, deep 16×1) per
fix; full grid once the recipe is settled. Shallow normal + reversed queued after the deep pair
(`scripts/arith_bal_shallow.sh`).

## D-009 · Add place-value embeddings ("Abacus")? (open, after D-007)

**Question.** McLeish et al. (NeurIPS 2024) found transformers fail at multi-digit arithmetic
mainly because they lose track of each digit's place within its number; an extra embedding for
"place within this number" fixed it, and they used looped transformers with input injection.

**Options.** (a) No change. (b) Add a learned embedding per digit token for its position within
its number (counted from the number's end, so place value lines up), on top of RoPE.

**What would change our minds.** If reversed output (D-007) already fixes 3–5 digit answers,
Abacus may be unnecessary; if big answers still fail after reversing, it's the next fix.

**Recommendation.** Decide after D-007's results. Implementation ready (off by default, `--place_emb`, tested; answer digits counted from the answer's start so nothing leaks). **Needs Walter?** Yes. **Status.** Open.

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

## D-007 · Write answers smallest digit first? (decided: normal order)

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
**Result (seed 0):** reversed worse for looped (primary −0.04, small −0.11) and shallow (−0.04, −0.08) → **keep normal order** per the plan's rule. Deep pending, but the decision doesn't depend on it.

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
