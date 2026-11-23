# Idea backlog

Not planned until promoted into `plan.md`. S = short term (days, reuses current models/data),
M = medium (a week, new runs), L = long term (after directional results). Logged 2026-11-23.
Related hypotheses in `hypotheses.md`.

## Short term
- **I-S1 Operator labels on test examples** (Walter). Label each test example with counts of
  `*`, `+`, `-`, parentheses, plus input size (already have `op_digits`). Note: the data has **no
  division** (operators are `+ - *` only; parentheses are minimal PEMDAS). Can be computed from the
  frozen test tokens without rebuilding anything, and joined to existing `per_example.npz` dumps
  (same row order) → slice every finished run by operator mix for free. Tests H14.
- **I-S2 Report %Gap** = (looped − shallow)/(deep − shallow) per slice, as in Saunshi et al.
  Ours: steps 0.91/1.08, big answers 0.55/0.63, big inputs 0.23/0.46.
- **I-S3 Add-only and subtract-only test sets** (Walter). 1–4 digit inputs, varied depth,
  test-only (training data unchanged). Partly out of distribution on purpose: training has
  multi-digit inputs only at depths ≤ 3, so deep + multi-digit = generalization test. Answers
  stay small-ish without `*`, which separates number size from multiplication lookup. Tests H13
  (matches McLeish's finding that looping ≈ depth on addition). Needs a generator flag (allowed
  operators) + a new data dir + an eval script on saved checkpoints.

## Medium term
- **I-M1 Progressive loss** (McLeish): loss at full K + loss at a random smaller K. Cheap first
  version of plan step 5's deep supervision.
<a id="idea-offset"></a>

- **I-M2 Random offset on place tags** (McLeish, β ~ U[0, max]) + train inputs ≤ 2 digits, test
  3–4 digits. Tests H18.
<a id="idea-attention"></a>

- **I-M3 Attention changes** (Walter: "PEMDAS attention"). Options, from least to most hand-coded:
  (a) look first (now H19 in `hypotheses.md`): plot attention maps per head per loop pass: do heads
  already focus on the sub-expression being computed? (b) prefix attention: question tokens see the whole question
  (only answer tokens stay causal). Today `3+4*5`: the `+` can't see the `*` after it. Generic,
  fair. (c) more / smaller heads. (d) tree-masked attention (each token sees only its own
  sub-expression): likely helps, but it hand-codes the algorithm, so a win says little about
  loops. Do (a) before anything else.
- **I-M4 Saunshi's cosine regularizer** on deep 16×1 (pull blocks toward each other): keeps deep's
  arithmetic and gains looped's step skill?
- **I-M5 Additive vs concat input injection**; McLeish's multiplication tweaks (no injection
  inside the loop, loop gradients ÷ K).
- **I-M6 Fit accuracy = α·log(effective depth) + β** over the grid; compare α for loops vs layers.

## Long term
<a id="idea-wide"></a>

- **I-L1 Much wider model** (Walter): e.g. d_model 512–1024 (McLeish uses 1024). Question: what
  logic appears with width, and do loops still add steps on top? Tests H16.
<a id="idea-algebra"></a>

- **I-L2 Algebra** (plan "Later"). Tests H17.
- **I-L3 Number base 2 / 10 / 16**; embedding-space side quest; routing / branching; back to
  Qwen once a recipe clearly works.

## Docs (later; Walter, 2026-11-23)
- **Research log per topic**: split `research-log.md` into one file per line of work (e.g.
  `log/digit-order.md`, `log/place-value.md`, `log/h13-h14.md`), each ordered by date/time, with
  an index. Today it is one long chronological file.
- **`history.md` as a changelog**: dated entries of direction changes, newest first, each linking
  the hypothesis or decision that caused it.
