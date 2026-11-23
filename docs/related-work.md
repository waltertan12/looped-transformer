# Related work: the two closest papers (read 2026-11-23)

Raw facts were extracted by Haiku agents from the arXiv full text; numbers below are from those
reports. Check against the PDF before quoting externally.

## Saunshi et al., "Reasoning with Latent Thoughts: On the Power of Looped Transformers" (Google, ICLR 2025, arXiv 2502.17416)
- Notation (k⊗L) = k-layer block looped L times. **No input injection, no halting, fixed loop
  count** (train = test). "Middle looping" (own first/last layers) = our prelude/core/coda.
- Synthetic: n-ary addition of 3-digit numbers (n up to 32), p-hop induction, i-GSM (mod 7,
  depth ≤ 4). d=256 (addition) / 128, 8 heads, Adafactor, 200k steps. 3 seeds, **report the max**.
- Addition: 2⊗1 38.8% at n=32 vs Loop 2⊗6 99.5%; 12⊗1 100%. i-GSM: 1⊗8 73.2 = 8⊗1 73.2.
  Looped ≈ iso-compute deep model with k× fewer weights.
- 1B LM on Pile: looped fills **34–48%** of the gap to the deep model on closed-book QA
  (memorization) but **56–61%** on math word problems and **131–133%** on reasoning primitives.
  Metric: %Gap = (looped − shallow) / (deep − shallow).
- Inductive bias: at equal perplexity, looped models do better on reasoning tasks.
- Regularizer: keep a normal deep model but add a cosine-similarity loss between blocks
  k apart (λ=10, k=4): best of both (PPL 7.38 vs 7.40; math 36.4 vs 29.3; primitives 57.2 vs 47.5).
- Scaling: accuracy ≈ α·log(effective depth) + β for looped and deep alike.
- Theory (expressivity only): looping a 1-layer block log n times does group composition and
  p-hop; m loops can simulate m steps of chain-of-thought ("latent thoughts").
- Not done: test-time loop counts, extrapolation to more loops, adaptive halting, number size.

## McLeish et al., "Transformers Can Do Arithmetic with the Right Embeddings" (NeurIPS 2024, arXiv 2405.17399)
- Abacus = learned embedding of a digit's position *within its number*, **reversed digits**
  (least significant first), so equal-significance digits share an index. Random offset
  β ~ U[1,100] during training (β=1 at test) for length generalization. Combined with NoPE.
- Addition: train ≤ 20 digits, test to 100–120 (6× length generalization).
  d=1024, 16 heads, batch 8192, 1 GPU-day per run, 3 seeds (mean).
- Input injection: embedded input **added** before each layer. Progressive loss: loss at full
  K plus loss at a random smaller K (Bansal 2022).
- Fixed effective depth 16, OOD addition: 16×1 (122M) 97.9, 8×2 (64M) **99.1**, 4×4 98.8,
  2×8 (19M) 97.9, 1×16 (12M) 79.8. Looping matches depth on addition with up to 6× fewer weights.
- Multiplication: only 4×4 looped, ≤ 15 digits, near-perfect **in distribution only**. Needed
  input injection removed from the loop and loop gradients divided by K.
- Not done: natural language, test-time loop counts, adaptive compute. Single-operation tasks only.

## How our results line up (D-012 normal arm, 120k steps, seeds 0 / 1)
- **Same split as Saunshi's memorization/reasoning.** Our %Gap (looped − shallow)/(deep − shallow):
  1-digit answers at depth 8: 0.91 / 1.08 (step-count work). 4-digit answers at depths 3–6:
  0.55 / 0.63; multi-digit inputs: 0.23 / 0.46 (number-size work). Steps ↔ their "reasoning",
  number size (times-table-like lookup) ↔ their "memorization".
- **Why McLeish's looped model matches deep on numbers and ours doesn't:** their task is one
  addition, where the carry chain is a sequential procedure (loop-friendly). Ours is mostly
  multiplication and nested results, which need lookup capacity (weights). Consistent, not
  contradictory, but worth stating carefully.
- **Reversed order:** McLeish uses it for single operations; we find it helps a single big
  multiplication (+0.12…+0.15) but hurts multi-step problems for shallow models; with loops and
  long training it's about a wash. Not shown in either paper.
- **What we add:** nested multi-step arithmetic with exact control of depth vs number size,
  2 seeds reported as both (Saunshi reports the max of 3), and negative results (fixed-K models
  fail at other K; reversed + multi-step).

## Ideas these papers suggest (not planned yet; for Walter)
1. **Report %Gap** as a headline metric in the write-up (directly comparable to Saunshi).
2. **Progressive loss** (McLeish) = cheap first version of plan step 5's deep supervision.
3. **Saunshi's cosine regularizer** on deep 16×1: does a "nearly looped" deep model keep the
   arithmetic of deep and the step skill of looped?
4. **Length generalization:** train inputs ≤ 2 digits, test 3–4 digits, with the random offset
   on our place tags. Tests whether tags + loops extrapolate number size.
5. **Fit accuracy = α·log(depth) + β** across our grid; compare α for loop vs depth.
6. **Additive vs concat injection**, and McLeish's multiplication tweaks (no injection inside the
   loop, gradients ÷ K): cheap ablations on the big-number weakness.
7. A single-operation task (pure big addition) where looping should match deep, as a control
   for the "weights buy arithmetic" claim.
