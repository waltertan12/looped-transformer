# Engineering & Research Playbook: Looped Transformers (TinyLlama to Protein LMs)

> **Source note.** Captured 2026-11-21 from a playbook pasted in by Walter. The equations and
> symbols didn't survive the paste. Where a value is a published model fact (TinyLlama or
> ESM-2 sizes), it has been filled in and marked *(filled)*. Where the original value is
> unknown (thresholds, gate init values, LoRA ranks and so on), it's marked **[?]** and
> should be restored from the original.
>
> **Model choice for this repo:** Phase 1 uses **Qwen** (Qwen3-0.6B-Base to start), not
> TinyLlama. The playbook text below still says TinyLlama; read it as "the Phase 1 base model".
> Layer ranges need adapting to Qwen's depth (28 layers for 0.6B/1.7B, 36 for 4B).

---

## 1. Phase 1 Architecture: Retrofitting `TinyLlama-1.1B`

### 1.1 Baseline Architecture Reference

`TinyLlama-1.1B` (`TinyLlama/TinyLlama_v1.1` or `TinyLlama-1.1B-Chat-v1.0`) uses a standard
Llama-2 topology:

- **Depth & Width:** L = 22 decoder layers, hidden size d = 2048, SwiGLU FFN intermediate
  dimension 5632 *(filled)*.
- **Attention:** Grouped-Query Attention (GQA) with 32 query heads and 4 key-value heads
  (8 query heads per KV head), Rotary Position Embeddings (RoPE, θ = 10,000) *(filled)*.
- **Normalization & Vocab:** Pre-RMSNorm (ε = 1e-5), vocabulary size 32,000 *(filled)*.

### 1.2 The Golden Rule: Prelude / Looped Core / Coda Split

Never loop the very first or very last layers of a pretrained transformer. Early layers map
discrete token IDs into syntactic/lexical subspaces, while final layers project latent
features into vocabulary logits (`lm_head`). Partition the L layers into three functional
zones:

1. **Prelude** (untied, run once): the first few layers **[?: exact range]**. Maps
   embeddings into a stable latent reasoning space.
2. **Looped Core** (shared weights, run K times): a contiguous or strided block of middle
   layers executed recurrently for K passes.
3. **Coda** (untied, run once): the last few layers **[?: exact range]**. Decodes refined
   latent states into next-token predictions.

### 1.3 Define Your Experimental Track Upfront

Do not conflate parameter-compression experiments with depth-scaling experiments. Run them
as distinct experimental tracks:

| Track | Goal | Layer topology example (L = 22 base) | Compute & memory profile |
|---|---|---|---|
| **A: Compute Scaling (Iso-Parameter)** | Increase effective reasoning depth without adding weights. | Prelude (0–5) → loop layers (6–15) K ∈ {1, 2, 3, 4} times → Coda (16–21). | Same weight VRAM; K × core FLOPs; simulates a 32–52 layer model. |
| **B: Parameter Compression (Iso-Depth)** | Match a 22-layer model using far fewer unique weights. | Prelude (0–5) → loop a 4-layer core K = 3 times (replacing 12 layers) → Coda (18–21). | ≈36% fewer layer parameters; identical effective depth (22 layers). |

### 1.4 Weight Initialization Tricks (Preventing Perplexity Explosion)

Naively tying weights across pretrained layers, or duplicating a block at K > 1 without
boundary conditioning, causes immediate perplexity collapse **[?: PPL figure]**. Use these
initialization patterns:

1. **Identity-Preserving Residual Gates (for Track A).**
   When introducing recurrent passes K > 1, multiply the attention and MLP residual updates
   on passes k ≥ 2 by a learnable per-step scalar gate initialized near zero
   **[?: init value]**, or zero-initialize the `o_proj` and `down_proj` matrices of per-step
   adapters. At training step 0, the looped model reproduces the exact forward pass of the
   unlooped TinyLlama baseline.

2. **Strided Selection + SVD Residual LoRA Bases (for Track B).**
   - *Strided Initialization:* initialize an n-layer shared core from strided layers
     **[?: e.g. every 3rd layer]** rather than a contiguous block, so the block spans
     early-to-late semantic abstractions.
   - *Relaxed Recursion (Per-Step LoRA):* attach tiny, unshared LoRA adapters
     (rank **[?]**, targeting `q_proj`, `v_proj`, `o_proj`, `gate_proj`, `down_proj`) indexed
     by loop iteration k. If compressing dropped layers into a base layer, initialize the
     LoRA matrices via a truncated Singular Value Decomposition (SVD) of the mean weight
     residuals (dropped-layer weights minus base-layer weights).

3. **Untied Per-Step RMSNorm.**
   Keep separate `input_layernorm` and `post_attention_layernorm` scale parameters for each
   loop pass k. This costs about 2 · d · K extra parameters per looped layer, a negligible
   amount *(derived)*, while letting each iteration rescale activation magnitudes
   independently.

---

## 2. Critical Implementation Pitfalls & Silent Bugs

Autonomous agents frequently introduce subtle tensor-state bugs when modifying HuggingFace
`LlamaForCausalLM`. Guard against these four failure modes.

### 2.1 The KV-Cache Indexing Bug (Highest Risk)

In HuggingFace `transformers`, `DynamicCache` stores past keys and values indexed by
`layer_idx`. If a physical layer runs K times and passes `layer_idx = 6` on every pass,
`DynamicCache.update()` will either concatenate K times as many tokens along the sequence
axis or overwrite earlier passes, which corrupts causal attention masks during generation.

- **Solution 1: Explicit Per-Pass Virtual Cache (highest accuracy).**
  Assign every `(physical_layer, loop_pass)` pair a unique virtual `cache_idx`. If n middle
  layers loop K times alongside m untied layers, allocate m + n·K cache slots.
- **Solution 2: Terminal KV Sharing (lowest VRAM).**
  During autoregressive generation, only write keys/values into the KV cache on the final
  loop pass (k = K), or use `use_cache=False` for prefix prefill passes k < K while
  attending to cached terminal states from prior tokens.
- **Mandatory unit test:** assert that cached and uncached generation produce identical
  tokens across multi-token autoregressive rollouts in `bf16`/`fp32`.

> *Status in this repo:* `LoopSlot` implements Solution 1 (each slot gets its own cache
> index). `tests/test_loop.py` checks one decoded token; it should be extended to a full
> multi-token rollout.

### 2.2 Double RoPE Application

In Llama, RoPE is applied dynamically to `query_states` and `key_states` inside each
attention forward call using `position_ids`.

- **Pitfall:** do not modify `position_ids` across loop iterations for the same token
  sequence, and never apply rotary rotations directly to `hidden_states` in the residual
  stream. Pass the identical `(cos, sin)` position embedding tuple to every loop pass.

### 2.3 Hidden-State Norm Blowup & "Overthinking"

Repeatedly applying the core causes the hidden-state norm ‖h‖ to grow steadily with K. This
saturates Pre-RMSNorm and erases fine-grained lexical information from the input token
("overthinking").

### 2.4 BPTT Memory Explosion

Unrolling K passes through an n-layer core creates an n·K-layer computation graph in VRAM.

- **Fix:** enable activation checkpointing (`torch.utils.checkpoint`) around each loop
  iteration, or use truncated backpropagation: run the early passes inside
  `torch.no_grad()` and backpropagate through only the final few passes.

---

## 3. Training Recipe & Benchmark Suite for TinyLlama

### 3.1 Post-Training / Healing Data Mix

Never fine-tune a looped LLM only on narrow reasoning datasets. It will suffer severe
catastrophic forgetting on general syntax and fluency. Use a mixed healing curriculum
**[?: mix ratios]**:

- **General pretraining replay:** high-quality web/textbook data (e.g.
  `HuggingFaceFW/fineweb-edu` or `cerebras/SlimPajama-627B`).
- **Algorithmic & multi-hop reasoning:** math, code, and logical deduction (e.g.
  `nvidia/OpenMathInstruct-2`, `cosmopedia`, or synthetic multi-hop tracking tasks).

### 3.2 Composite Loss Function

Combine standard language modeling with teacher distillation and intermediate loop
supervision **[?: exact formula and weights]**:

- **Teacher:** use the frozen unlooped `TinyLlama-1.1B` (or `Llama-3.2-3B` if
  vocabulary-aligned) to stabilize output distributions (KL distillation).
- **Monotonic Deep Supervision:** attach the frozen Coda + `lm_head` to intermediate loop
  exits k < K with small loss weights. This teaches the network to make better predictions
  at every step, and allows early-exit halting at inference time.

### 3.3 Diagnostic Benchmark Suite

Evaluate every checkpoint across several inference loop counts K to plot **depth-scaling
curves** (metric vs. K):

| Benchmark category | Specific tasks | Purpose / what to watch for |
|---|---|---|
| Fluency & overthinking probe | WikiText-2 PPL, FineWeb-Edu val PPL | Verify PPL drops from K=1 → 2 → 3 and does not spike at K=4 (a spike indicates norm drift). |
| Synthetic fast-signal probe | Multi-hop variable tracking, parity, Needle-in-a-Haystack (NIAH) | **Run this first.** Shows clear separation between K=1 and K=3 within 1–2 GPU hours. |
| Reasoning & logic | GSM8K (8-shot), ARC-Challenge, BoolQ | Primary test of whether extra latent loops turn into multi-step reasoning gains. |
| Factual recall control | TriviaQA, PIQA, HellaSwag | Control metric. Looping adds compute depth, not parametric memory, so expect flat scaling. Drops indicate catastrophic forgetting. |

---

## 4. Phase 2 Roadmap: Transitioning to Protein Language Models (pLMs)

Once the looped pipeline is validated on TinyLlama, porting it to a protein language model is
a high-leverage research direction. Candidates: ESM-2 (`facebook/esm2_t33_650M_UR50D` or
`esm2_t30_150M_UR50D`) or ESMC (`EvolutionaryScale/esmc-300m` / `600m`).

### 4.1 Why Protein LMs Are Uniquely Suited for Looping

1. **Folding is iterative constraint satisfaction.** A 1D amino acid chain folds into a 3D
   structure driven by long-range epistatic contacts between residues far apart in sequence.
   Single-pass transformers struggle to propagate global tertiary constraints across all
   residues at once. Looping mimics iterative energy minimization, analogous to AlphaFold2's
   Evoformer recycling and ESMFold2's looped structure trunk.
2. **Bidirectional attention eliminates the KV-cache bottleneck.** Standard pLMs (ESM-2,
   ESMC) are bidirectional encoders trained with Masked Language Modeling (MLM), not causal
   autoregressive decoders. There is no KV cache, which removes the hardest engineering
   bottleneck from Phase 1.
3. **Tiny vocabulary (about 33 tokens).** In TinyLlama the embeddings and `lm_head` take
   about 131M parameters (about 12% of the model) *(filled)*. A pLM's vocabulary
   (20 canonical amino acids plus special tokens) takes about 42K parameters in ESM-2 650M
   *(filled)*. Nearly 100% of parameters live in the transformer blocks, which maximizes the
   impact of weight sharing.

### 4.2 Architectural & Training Adaptations (LLM → pLM)

When porting the looped codebase from `TinyLlama` to `ESM-2` / `ESMC`, apply these four
changes:

1. **Bidirectional attention & RoPE.**
   Set `is_causal = False` across all Prelude, Looped Core, and Coda layers. Every residue
   attends to every other residue on every loop pass. This lets pairwise attention heads
   sharpen contact maps step by step, from local secondary structure (α-helices, β-sheets)
   in the first pass to global tertiary contacts in later passes.

2. **MLM & iterative denoising objectives.**
   - *Masked Language Modeling:* mask 15% of sequence positions (standard 80% `[MASK]` /
     10% random amino acid / 10% unchanged) *(filled: standard BERT/ESM recipe)*. Compute
     cross-entropy loss at the final pass, plus auxiliary deep supervision at intermediate
     passes.
   - *Optional upgrade: pairwise attention residual bias.* In proteins, attention logits
     encode pairwise residue interactions. Pass a lightweight gated residual of the attention
     map from pass k into pass k+1 **[?: equation]**. This acts as a lightweight pair-track
     Evoformer without requiring O(L³) triangle updates.

3. **Strict sequence-identity data splitting (critical biological pitfall).**
   - Never split protein datasets randomly. Two proteins with high sequence identity are
     evolutionary homologs, and random splits cause massive data leakage and fake benchmark
     gains.
   - Always cluster sequences using MMseqs2 at a sequence-identity threshold
     **[?: e.g. 30% or 50%]** (for example, train on UniRef50 clusters and hold out entire
     sequence clusters for validation).

4. **Variable sequence length & padding hygiene.**
   Protein lengths range from tens to thousands of residues **[?: exact range]**. Make sure
   padding tokens are masked out before any global pooling, anchored Prelude gates, or
   per-step RMSNorm statistics, so padding ratios don't shift activation norms across loop
   passes.

### 4.3 Protein Evaluation Benchmarks

Evaluate the looped pLM across several loop counts K on tasks that directly measure
long-range structural reasoning:

- **Unsupervised long-range contact prediction** (`Precision@L`, `Precision@L/5` on
  CASP14 / CAMEO): train a lightweight logistic-regression probe on the final-pass attention
  maps (or use APC-corrected attention) to predict residue contacts (Cβ–Cβ distance < 8 Å,
  *filled: standard definition*) for long sequence separations (≥ 24 residues,
  *filled: standard definition*). Rising precision with K shows the loops are resolving
  folding constraints.
- **Zero-shot variant effect / epistasis prediction** (ProteinGym DMS substitutions,
  Spearman ρ): score single and multi-mutant protein variants via masked marginal
  log-likelihood ratios. Pay special attention to multi-mutant epistasis subsets, where
  single-pass models often fail to capture non-additive residue interactions.
- **Remote homology & fold classification** (SCOPe / TAPE): classify proteins that share
  structural folds despite low sequence similarity. This is a classic test of deep
  structural abstraction.

---

## 5. Agent Execution Checklist

Follow this sequence of steps when starting the repository:

- [ ] **Step 1: Baseline verification.** Run unlooped `TinyLlama-1.1B` on WikiText-2 PPL and
  a small **[?: N]**-sample GSM8K / synthetic multi-hop subset. Record exact VRAM,
  tokens/sec, and baseline metrics.
- [ ] **Step 2: Unit test suite (must pass before training).**
  - `test_identity_init`: verify that at initialization (gates at 0 or K = 1), the looped
    model matches baseline logits within a tight tolerance **[?]**.
  - `test_kv_cache_parity`: verify `use_cache=True` and `use_cache=False` produce identical
    token generations for K > 1 loops.
  - `test_norm_stability`: track ‖h‖ across K on a long **[?: length]**-token sequence;
    confirm norms stay bounded (within some multiple of the prelude norm **[?]**) without
    NaNs or exponential blowup.
  - `test_gradient_flow`: confirm non-zero gradients reach the Prelude, Looped Core, and
    per-step adapters without vanishing or exploding **[?: thresholds]**.
- [ ] **Step 3: Fast synthetic pilot.** Train for a short run **[?: steps]** on a synthetic
  multi-hop state-tracking task plus FineWeb-Edu replay, to verify that accuracy strictly
  improves from K = 1 to higher K.
- [ ] **Step 4: Full TinyLlama healing & depth sweep.** Run the full healing mix
  **[?: token budget]** with stochastic K, then evaluate the full benchmark table across a
  range of K, including K values beyond the training range to measure out-of-distribution
  depth extrapolation.
- [ ] **Step 5: Phase 2 port (`ESM-2` / `ESMC`).** Strip causal masking and KV-cache routing,
  swap in MLM plus pairwise attention residual bias, verify MMseqs2 identity-cluster splits,
  and benchmark contact prediction `Precision@L` and ProteinGym Spearman ρ across K.
