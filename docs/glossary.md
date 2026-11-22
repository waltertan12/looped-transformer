# Glossary (for Walter)

Plain-language definitions, each tied to this project. Terse agent-facing definitions are in
[`glossary-agent.md`](glossary-agent.md). Add a term whenever one comes up that isn't obvious.

## The model

- **Transformer block (layer).** One unit of the model: an attention step (tokens look at each
  other) then a small feed-forward network (each token "thinks" on its own). Our models stack
  a few of these.
- **Token.** One input symbol. Here every character is a token: `3`, `+`, `(`, `a`, `#`.
  The **vocabulary** is the set of all 46 tokens.
- **Embedding.** The list of 256 numbers a token is turned into before the blocks process it.
- **Hidden state.** Each token's current list of 256 numbers as it moves through the blocks.
  The model's "working memory". Sometimes called the residual stream.
- **Causal attention.** Each token can look only at tokens before it, the way you read left to
  right. The answer is predicted after the `=`, once the whole problem has been read.
- **RoPE.** How the model knows token positions (1st, 2nd, …). Technical detail, no need to tune.
- **Prelude / core / coda.** Our model's three parts: read the input (prelude, 1 block), think
  (core, the part that can loop), write the answer (coda, 1 block).
- **Looped transformer.** A model that runs the same core blocks several times instead of
  having many different blocks. Same weights, more thinking time.
- **Loop count (K).** How many times the core runs. "2×8" = 2 core blocks, run 8 times.
- **Effective depth.** Core blocks × loops: how many block-steps a problem gets. 2×8 and
  16×1 both have effective depth 16 (same compute), but 2×8 has far fewer weights.
- **Input injection.** Feeding the original input back in on every loop pass, like re-reading
  the question while working. Without it, each pass drifts (our Qwen version lacked it).
- **Weights / parameters.** The numbers the model learns. "4.4M" = 4.4 million. **Unique**
  parameters don't count reused ones twice, so looping adds compute but not weights.

## The task and data

- **Depth.** How many operations must happen one after another. `(3+4)*2-1` has depth 3.
  Not the same as length: `(1+2)*(3+4)` and `3+4+5+6` both have 3 operations, but the first
  has depth 2 (the brackets can be done side by side) and the second depth 3 (each `+` waits
  for the one before). Looping should help with depth, not length.
- **Number of operations (n_ops).** Total `+ - *` count. Measures length, not depth.
- **Chain length.** In the intermediate-values task (`a=3+4;b=a*2;b?`), how many variables
  you must follow to reach the answer.
- **Distractor.** A statement the answer doesn't need, there to make the model find the
  right one.
- **Train / test split.** Train = problems the model learns from. Test = problems held back to
  check it learned the skill, not the answers.
- **Overlap / memorization.** If a test problem also appears in training, getting it right may
  be recall, not reasoning. We hold out depth-1/2 problems so the test has zero overlap.
- **Out-of-distribution (OOD) / extrapolation.** Testing on harder problems than any in
  training (depth 9–16 when trained on ≤ 8). The "longer problems" question.
- **Stop token (`#`, EOS).** Marks the end of the answer, so the model knows when it's done.
- **Place-value embedding** (the paper calls it "Abacus embedding"). An extra learned
  embedding that tells each digit its place within its number (ones, tens, hundreds…), so the
  model doesn't have to work out place values by counting. Different from RoPE, which encodes
  position in the whole sequence. Pairs naturally with reversed output: the first digit
  written is always the ones place. Decision D-009.
- **Number base.** Writing numbers in base 2, 10, 16… A difficulty dial: smaller bases mean
  more digits and longer carry chains (many easy steps), larger bases fewer digits but harder
  per-digit work.
- **Digit order (normal vs reversed).** Normal writes `258`; reversed writes `852` (sign
  still first). Both are next-digit prediction, but in normal order the first digit needs
  every carry worked out in advance (mental arithmetic, no paper); in reversed order each
  digit needs only a small local step and the digits already written act as scratch paper
  (like long arithmetic on paper). Decision D-007.
- **Input size (op_digits).** How many digits the numbers in the problem have (`3+4` vs
  `347*28`). A second difficulty dial, separate from depth.
- **Balanced dataset.** Training/test problems sampled so each answer size gets a fair share,
  instead of rare big answers being ~1% of the data.
- **Exact match.** A problem counts as right only if every answer character is right.

## Training

- **Loss.** A number measuring how wrong the model's predictions are. Training lowers it.
- **Learning rate (lr).** How big a step the model takes when updating its weights. Too high:
  unstable or stuck (our 18-layer model at 1e-3). Too low: slow. We use 3e-4.
- **AdamW.** The optimizer: the rule for updating weights. It adapts each weight's step size
  automatically, but still needs a base learning rate.
- **Warmup / cosine schedule.** Start with a tiny learning rate, ramp up, then slowly lower it.
- **Curriculum.** Train on easy problems first, add harder ones as each level is mastered
  (here: next depth after 90% exact match on the current top depth).
- **Step.** One update of the weights, from one **batch** (256 problems).
- **Seed.** The random starting point. Two seeds = the same experiment run twice with
  different luck; differences smaller than the seed-to-seed spread aren't real.
- **fp32 / bf16.** Number precision. We store weights in fp32 (precise) and compute in bf16
  (fast). Storing weights in bf16 silently lost small updates in the Qwen runs.
- **Teacher forcing.** During training and scoring, the model sees the correct earlier answer
  digits when predicting the next one.

## Experiment design

- **Baseline / control.** The comparison that tells you if your change mattered. Exp2's
  no-loop control showed a "looping gain" was really just the training.
- **Compute-matched.** Same amount of computation (same effective depth). The grid's
  anti-diagonal.
- **Parameter-matched.** Same number of weights. A grid row.
- **Ablation.** Removing or changing one piece to see what it was doing.
- **Confound.** A second cause mixed in with the one you're testing. Example: with real
  numbers, deep problems also have bigger numbers, so failures could be either.
- **Null result.** The outcome where nothing changes. Worth stating before the run, so you know
  what "it didn't work" looks like.
- **Pre-registration.** Writing down the question, prediction and "what would change my mind"
  *before* running. Prevents moving the goal after seeing results.
- **Goal drift.** Changing what counts as success after seeing results. Different from
  simplifying, which keeps the question and makes the test cheaper.

## From the Qwen track

- **Distillation.** Training a model to copy a bigger model's predictions.
- **KL (divergence).** How different two models' predictions are. 0 = identical.
- **Retrofit.** Converting an already-trained model into a looped one.
- **Dials (gates).** Our exp2/3 numbers that scaled each extra pass; training set them to ~0.

## Side-quest vocabulary

- **Logit lens.** Reading off what answer the model would give at an intermediate point
  (e.g. after loop pass 3).
- **Fixed point.** When running another loop pass no longer changes the hidden state.
