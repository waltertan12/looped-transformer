# Hypotheses

Every claim the project tests, with its status. Open ones come first, in priority order.
To re-rank, reorder them. When a result comes in, move the entry down and link the log entry.

Numbers are seed 0 / seed 1 at 120k training steps unless noted.

## Open, in priority order

### 1. H19 · Each loop pass works through one level of the problem
**Claim.** The looped model solves nested problems the way a person does: innermost brackets
first, one level per pass. Its attention heads should "spotlight" the sub-expression being worked
on, moving outward as the passes go on.
**Why it matters.** It would show *how* loops buy steps, not just that they do.
**Test.** On the saved models, no training needed: plot what each attention head looks at, on
each pass, for problems of known depth. Reading the answer out after each pass works properly
only once H9's training is in place.
**Cost.** Free.

### 2. H18 · Fixed tags can be made to work on longer numbers
**Claim.** The place-value tags break on numbers longer than training because their labels are
absolute and shared with the answer. A random offset on the labels during training, or separate
labels for question and answer, fixes it.
**Today.** 4-digit `a+b` is within 5% for 0.92 of problems without tags and 0.01–0.10 with tags.
**Test.** Train looped and deep with each fix, plus a no-tag 120k control. Success: 4-digit
`a+b` above 0.5 exact, with no loss on trained sizes.
**Cost.** 3–5 runs.

### 3. H9 · A looped model can choose its own loop count
**Claim.** Trained to give a usable answer after every pass, a looped model can stop early on easy
problems and use more passes on hard ones, without losing accuracy.
**Test.** Train with an answer read out after every pass, then stop by a confidence rule, then by
a learned stop signal. Neither close paper did this.
**Cost.** A few runs plus a training change.

### 4. H10 · The model knows when it's done
**Claim.** A stop signal learned from the model's internal state beats a hand rule ("stop when the
answer stops changing") and a separate checker network.
**Test.** After H9, compare the three on the same model.

### 5. H15 · Letting the question see itself helps
**Claim.** Today each token sees only earlier tokens, so in `3+4*5` the `+` can't see the `*`.
Letting all question tokens see each other should help, and may shrink part of the loop gap.
**Test.** One mask change, then the three-model comparison.

### 6. H11 · Which layers loop matters
**Claim.** Looping only the middle layers beats looping all of them.
**Test.** Five layouts with 4 distinct layers and the same compute.

### 7. H16 · Width fixes number size, and loops still add steps
**Claim.** Wider models (512–1024) close the number-size gap to deep, and looping still adds
steps on top.

### 8. H12 · Loops help follow references
**Claim.** Loops help with chained definitions (`a=3*4; b=a+2; b?`), not just nested brackets.
The dataset already exists.

### 9. H17 · Step skill carries over to algebra
**Claim.** A model that is good at steps learns to solve `3*x+5=20` faster.

## Supported

- **H1 · Loops buy steps.** Looped beats shallow (same weights) by +0.11 at depth 4 and +0.25 at
  depth 8 on 1-digit answers.
- **H2 · Looped matches deep on steps** with a quarter of the weights, within ±0.03 at every depth.
- **H3 · Weights buy number size.** Deep beats looped by +0.05 to +0.15 on big answers and inputs.
- **H4 · The step skill reaches past training depth.** At depth 13 (trained up to 8), looped
  scores 0.62 / 0.60 and shallow 0.42 / 0.43, about the same as deep.
- **H8 · The 40k-step models were undertrained.** 120k steps lifted every result by +0.08 to +0.26.
- **H7 · Place-value tags help, a little** (weak). +0.055 on big answers for looped at 40k, one
  seed. No effect on shallow. They also break longer numbers (see H18).

## Rejected or not supported

- **H5 · A model trained at one loop count works at others.** Rejected: at most 0.11 at other counts.
- **H6 · Reversed answer order helps.** Rejected by the pre-set rule on all three models. The
  effect depends on size. On big answers it is −0.07 for shallow, −0.03 for looped and
  +0.13 / +0.10 for deep.
- **H13 · Loops match deep on pure addition and subtraction.** Rejected. On unseen problems
  (depths 1–3, 2–3-digit inputs), deep still wins by +0.05 to +0.15. So the gap is about number
  size, not times tables.
- **H14 · Operator mix predicts errors.** Not supported. With depth and number size held fixed,
  the most-`*` third scores within ±0.015 of the least-`*` third.
