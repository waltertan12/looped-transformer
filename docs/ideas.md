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
  (e) attention supervision (Walter, 23 Nov): keep normal attention but add a small extra loss that
  pulls one head toward the expected pattern (for example each `)` looks at the operands inside its
  brackets). Softer than (d) because the model can still ignore it. Same catch as (d): a win shows the
  pattern helps, not that loops find it. The loop-specific test is whether it helps looped more than deep.
  Step (a) now has a viewer: `docs/visual/attention.html` (data from `scripts/attn_export.py`).
- **I-M4 Saunshi's cosine regularizer** on deep 16×1 (pull blocks toward each other): keeps deep's
  arithmetic and gains looped's step skill?
- **I-M5 Additive vs concat input injection**; McLeish's multiplication tweaks (no injection
  inside the loop, loop gradients ÷ K).
<a id="idea-heads"></a>

- **I-M8 Head count** (Walter, 24 Nov; H21). Walter's view: steering attention by hand is hard to
  justify, so let training find the pattern and vary only the head count. 1, 2, 4 (today) or 8
  heads. Width stays 256, so the weights are identical and only how attention is split changes
  (8 heads of 32 dims vs 1 head of 256). Already a flag: `--n_heads`. First test: 2 vs 4 on
  shallow, looped and deep, 2 seeds, 120k steps, on whatever recipe is the default by then. The
  attention viewer shows whether heads take separate jobs.

<a id="idea-intermediate"></a>

- **I-M7 Teach each pass an intermediate step** (Walter, 23 Nov; H20). Add a loss so the state
  after pass p can be read out as the problem with p levels of brackets done, for example
  `(8-3)*(4-6)+7` → `5*-2+7` → `-10+7` → `-3`. A small readout head per pass, or a probe-like
  loss at each `)` once its level ≤ p. Softer version: only require "by pass p, every piece of
  level ≤ p is readable", with no fixed order. Why: H19 found levels 4+ unreadable, so this tells
  us whether forcing explicit steps helps or hurts. Bonus for H9: if the state is explicit, the
  model can stop when it is a single number. Catches: it hand-writes the algorithm and fixes one
  step per pass, which may be worse than what the model finds. Loop-specific test: does it help
  looped more than deep given the same per-layer-group loss? Related: deep supervision, progressive
  loss (McLeish), latent-reasoning work. Prior art: [related work](related-work.md#rw-steps).
- **I-M6 Fit accuracy = α·log(effective depth) + β** over the grid; compare α for loops vs layers.

## Long term
<a id="idea-protein"></a>

- **I-L4 Protein track** (Walter, 24 Nov). Same loop question on real biology: looped vs deep
  protein models from scratch on Swiss-Prot, scored on contacts by protein length and contact
  range. First step: probe ESM-2 8M. Full notes: [protein.md](protein.md).
- **I-L5 Virtual cell toys** (Walter, 24 Nov). Gene networks where a knockout spreads hop by hop:
  hops play the role of bracket depth. Smallest: a 10–50 gene on/off network, predict the state
  after t steps. Notes: [virtual-cell.md](virtual-cell.md).
<a id="idea-objectives"></a>

- **I-L6 Other training objectives** (Walter, 24 Nov). Today the only loss is cross-entropy on the
  answer characters, each digit weighted the same.
  (a) **Custom losses**, cheapest: weight the digits that need carries, add per-pass targets (H20),
  or add small side tasks (answer length, sign). Can hurt if a side task competes with the answer.
  (b) **Energy-based**: learn a score for "this answer fits this question" and find the answer that
  scores best, instead of writing it left to right. A loop pass becomes one step of improving the
  guess, and "stop when the score stops improving" is a natural stop rule for H9. Harder to train
  (needs wrong answers to push away) and slower to answer. Look up energy-based transformer work first.
  (c) **JEPA / latent prediction**: predict the *internal representation* of the next simpler
  form of the problem, not its characters. This is H20 done in latent space: pass p should land
  near an encoder's embedding of the problem with p bracket levels done. Risk: every state
  collapsing to the same point, which needs the usual fixes (a slowly updated target encoder).
  Order if pursued: (a) via H20 first, then (c) as a variant of H20, (b) last.
<a id="idea-wide"></a>

- **I-L1 Much wider model** (Walter): e.g. d_model 512–1024 (McLeish uses 1024). Question: what
  logic appears with width, and do loops still add steps on top? Tests H16.
<a id="idea-algebra"></a>

- **I-L2 Algebra** (plan "Later"). Tests H17.
- **I-L3 Number base 2 / 10 / 16**; embedding-space side quest; routing / branching; back to
  Qwen once a recipe clearly works.

<a id="idea-equilibrium"></a>

### Other objectives and physics-style framings (Walter, 24 Nov)

**Origin.** A 24 Nov discussion of ideas from outside looped transformers: custom losses,
physics framings (diffusion, equilibrium), energy-based models, latent world models (JEPA), and
RL post-training that rewards exact answers. Hypotheses H22–H25 come from it.
Papers named here are from memory. Check them before relying on them.

- **I-L3 Loop to equilibrium → H22** (deep equilibrium models, Bai et al. 2019). Run the loop until the
  state stops changing and train on that fixed point. Cheap first check, no training: on today's
  checkpoints, measure how much the state changes from pass to pass, including past pass 8. H5
  (fails at other loop counts) hints our models don't settle. A settling state would also give
  H9 a free stop rule.
- **I-L4 Energy view** (energy-based models, e.g. iterative reasoning as energy minimization, Du
  et al. 2022). Learn a score for "how wrong is this partial answer" and let each pass lower it.
  Harder problems then take more passes on their own, which is H9's goal.
- **I-L5 Diffusion-style refinement → H25.** Flow matching is the current favored way to train diffusion-style models. Discrete versions exist for tokens. Write a rough answer, then fix all digits over several
  passes instead of left to right. Discrete diffusion models have been tried on puzzles like
  sudoku. Could help carries, which flow right to left.
<a id="idea-solvable"></a>

- **I-L8 Does the model know it will fail? → H23.** Origin: the question of whether JEPA could tell if a
  problem is solvable by the model. JEPA isn't needed: a probe on the state at `=` that
  predicts right vs wrong answers the same question. If it works, it is also a stop signal for
  H9 / H10 (keep looping while the model thinks it would be wrong).
- **I-L6 Latent targets** (JEPA-style). Like H20, but predict the hidden state of the next step
  instead of its text. Close to Coconut (see related work).
<a id="idea-rl"></a>

- **I-L9 RL post-training → H24.** Pretraining learns from lots of data. Mid-training is more
  training on a targeted mix (more math, longer inputs) before the final stage. Post-training is
  the final shaping: supervised fine-tuning, then RL. For us: take a trained checkpoint, sample its
  answers, reward 1 for exact and 0 otherwise. Honest prior: when exact labels exist, normal
  training is usually more efficient. RL's edge is fixing mistakes the model makes on its own
  outputs. Twist that ties to H9: reward exact answers that use fewer passes, which teaches
  stopping. Start on shallow (cheap).
- **I-L7 Other losses.** Weight answer digits by difficulty, reward exact answers only (RL), or
  penalize state change late in the loop to push toward equilibrium (links I-L3).

## Docs (later; Walter, 2026-11-23)
- **Research log per topic**: split `research-log.md` into one file per line of work (e.g.
  `log/digit-order.md`, `log/place-value.md`, `log/h13-h14.md`), each ordered by date/time, with
  an index. Today it is one long chronological file.
- **`history.md` as a changelog**: dated entries of direction changes, newest first, each linking
  the hypothesis or decision that caused it.
