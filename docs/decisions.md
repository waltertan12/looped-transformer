# Decisions

One card per decision that changes what an experiment means or what we spend compute on.
Newest first. Written with the `decision-brief` skill. Routine execution (bug fixes, launching
an already-approved run) doesn't need a card.

Card format: **Question** · **Options** (with what each outcome would tell us) ·
**Recommendation** · **Needs Walter?** · **Status** (open / decided: … by whom, when).

---

## D-014 · H18: which tag fix for longer numbers, on which runs? (decided: (a) and (b), separately)

**Question.** With place-value tags, 4-digit inputs come out as nonsense (within 5% for 0.01–0.10
of sums; 0.92 without tags). Suspected cause: question digits only ever had tag ids 1–3, and id 4
otherwise means "4th answer digit". Which fix do we train, and on which runs?

**Options.**
- (a) **Random offset** (McLeish): during training, add a random shift β ~ U[0, 8] to every
  question digit's tag (β = 0 at test). Ids 1–11 all get trained on question digits. Tests the
  "id 4 was never trained on a question" explanation.
- (b) **Separate tag tables** for question and answer digits, no offset. Removes the collision but
  question id 4 is still untrained. Tests the "it reads as an answer position" explanation.
- (c) Both together.
Each at 40k steps on shallow 2×1 and looped 2×8, seed 0 (same screen design as D-011). Baselines
already exist at 40k: tags (`place/*-normal-pv-s0`) and no tags (`bal/*-normal-s0`). (a)+(b) = 4 runs,
≈ 1.2 h. Adding (c) = 6 runs, ≈ 1.8 h. The winner then gets the long probe set (shallow / looped /
deep × 2 seeds, 120k, ≈ 6 h) as in D-012.
What outcomes mean: (a) fixes it and (b) doesn't → untrained labels were the cause. (b) fixes it →
the collision was. Neither → tags can't extrapolate here; fall back to no tags or train on longer numbers.

**Test data.** Add-only and mixed single-operation sets with exactly 4- and 5-digit inputs (new
cells in the add/sub builder, plus `*`), depths 1–2.

**What would change our minds (pre-registered).** A fix passes if, on looped, (1) 4-digit `a+b`
(depth 1) exact accuracy ≥ 0.5, and (2) in-range accuracy (multi-digit inputs depths 1–3, and big
answers on 1-digit inputs) is within 0.03 of tags-without-fix at 40k. If (1) fails but within-5%
rises to ≥ 0.8, the fix restores "approximately right" only: report it, don't adopt. Confounds:
seed noise (one seed: confirm the winner on 2 seeds in the long set); training length (all
compared at 40k).

**Recommendation.** (a) and (b), not (c): each tests one explanation, 4 runs, ≈ 1.2 h. Run (c)
only if both pass.

**Needs Walter?** Yes. **Status.** Decided by Walter, 2026-11-23: (a) and (b) as separate runs.
Pass bar stays ≥ 0.5 exact; results also reported against ≥ 0.7 ("as good as in-range": tagged
3-digit sums are 0.74).
**Outcome (2026-11-23 15:20):** offset "approximate only" (4-digit `a+b` within 5%: 0.97 vs 0.03 plain
tags, 0.90 no tags; exact 0.01), separate tables "fail" (0.00). No cost on trained sizes. The 0.5-exact
bar was unreachable at 40k: in-range 3-digit sums are only 0.14–0.21 exact. Adopting offset as the
default is open for Walter. Research log 15:20.

## D-013 · H19: how do we tell whether each loop pass handles one level of brackets? (decided: (b))

**Question.** Does the looped model work through a nested problem one level per pass, innermost
first? Measured on saved models only (no training).

**Options.**
- (a) **Attention maps only.** Plot where each head looks on each pass for a few problems. Cheap
  and intuitive, but it's eyeballing: easy to see patterns that aren't there.
- (b) **Probes plus attention maps.** A probe is a tiny linear classifier trained to read a value
  out of the model's hidden state. For every bracketed sub-expression in a problem, at its last
  token, train a probe after each pass to read out the sub-expression's last digit. If passes
  handle levels, a sub-expression 1 level high becomes readable after an early pass, 2 levels high
  later, and so on: a staircase. Attention maps are the illustration, probes the evidence.
- (c) **Read the answer out after each pass.** Doesn't work yet: the model only gives sensible
  answers after exactly 8 passes (≤ 0.11 otherwise). Needs H9 first.
Cost of (b): minutes of GPU. Same probes on deep 16×1 (after each layer) and shallow 2×1 for
comparison: does depth also form a staircase?

**What would change our minds (pre-registered).** Define, for each height h, the first pass at
which the probe reaches 90% of its best accuracy for that height. H19 is supported if that pass
rises with h for h = 1…5 (each step ≥, overall rank correlation ≥ 0.8) on looped. Not supported if
all heights become readable at the same pass, or readability doesn't depend on pass. Confounds:
higher sub-expressions are longer and end later in the sequence (also probe at the matching
position in shallow, which has no passes); small values are easier to decode (report per value
range); a probe can find information the model doesn't use (state it, it's a known limit).

**Recommendation.** (b): probes give a number a rule can judge; maps make it visible.

**Needs Walter?** Yes (it defines what the result means). **Status.** Decided by Walter,
2026-11-23: (b), probes plus attention maps.
**Outcome (2026-11-23 14:20):** not supported by the rule, but the rule was flawed: levels 4–7 are
never readable, so their "ready pass" is noise. Levels 1–3 become readable in order (1 before the
loop, 2–3 together and gradually), in looped and deep alike. Research log 14:20.

---

## D-012 · Long probe set: the checkpoint run (running: tags on, normal + reversed arms)

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
**Recipe (by the rule):** `--place_emb`, normal order (reversed: +0.033 on multi-digit inputs,
−0.110 on 1-digit inputs; fails both halves). **Status.** Launched 2026-11-22 21:40, ~6 h;
results in `results/toy/long/`.
**Extended by Walter, 2026-11-22 22:20:** add a reversed-order arm (tags + `--reverse_answer`),
same 6 runs at 120k, part of this experiment (`scripts/arith_long_arm.sh ... rev`, names
`long/<model>-rev-s<seed>`, queue log `results/toy/long-rev.queue.log`; starts when the normal
arm finishes, ~02:50; done ~08:15). **Rule (written before results):** on each model, reversed
wins if it beats normal by ≥ 0.05 on multi-digit-input problems (depths 1–3) **and** costs
≤ 0.03 on 1-digit-input problems, on **both seeds**. Report per model; the 40k screen said
reversed loses with more steps, so the question is whether longer training changes that.
**Outcome (2026-11-23):** reversed fails the rule on all three models → **normal order stays.**
Multi-digit-input gain: shallow +0.01/+0.03, looped +0.04/+0.06, deep +0.03/−0.01. But the
cost/benefit flips with capacity: big answers shallow −0.07/−0.08, looped −0.03/−0.03, deep
+0.13/+0.10 (research log 08:15).

## D-011 · Place-value embeddings: which runs? (done: tags on, normal order)

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
**Looped result (outcome):** tags + normal vs no tags: big +0.055, small +0.006, multi-digit inputs
+0.035 → **adopt tags.** Tags + reversed vs tags + normal: multi-digit inputs +0.033, 1-digit
inputs −0.110 → **keep normal order.** Details: research log, 2026-11-22 looped pair.

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
