# Glossary (agent-facing)

Precise meanings as used in this repo, with code locations. Human-facing explanations:
[`glossary.md`](glossary.md). Keep both in sync when adding a term.

## Model (`src/looptf/toy/model.py`)
- **ToyLM**: decoder-only, pre-norm RMSNorm, causal SDPA + RoPE, SwiGLU, no biases, untied head. d_model 256, 4 heads, d_ff 1024, vocab 46.
- **prelude / core / coda**: `n_prelude` (1), `n_core` (grid: 2/4/8/16), `n_coda` (1) blocks. Only core is repeated.
- **K / n_loops**: core repetitions in `forward(x, n_loops)`. Notation `c×k` = n_core c, n_loops k.
- **effective depth**: n_core × K (core block applications). Compute-matched cells share it.
- **inject**: `concat` (default): `u = adapter(cat[h, e])` before each core pass, `e` = prelude output; `add`: `h+e`; `none`: `h`.
- **unique params**: `num_params()`; independent of K.

## Data (`src/looptf/toy/data.py`, `chain.py`, `dataset.py`; built by `scripts/build_data.py`)
- **nested**: PEMDAS expression over digits 1–9, ops `+-*`, minimal parens; level = **depth** (max operator nesting on any root-leaf path, AST BinOp depth); `n_ops` = total operators. Width 80.
- **chain**: shuffled `v=expr;` statements, `chain_len` linked variables + 0–4 distractors, query `v?`; level = chain_len. Width 256.
- **sequence format**: `prompt + str(answer) + "#"`; prompt ends `=` (nested) or `?` (chain). Answer exact integer, unbounded (`BOUND=None`).
- **labels**: int64 columns per meta.json; always `answer`, `length` (prompt len incl. terminal), `n_tgt` (answer tokens incl. EOS).
- **place-value embedding** (= Abacus, McLeish et al. 2024): `ToyConfig(place_emb=True)` / `train.py --place_emb`. `model.place_ids(x)`: 0 for non-digits; question digits = 1 + place from the number's end (true place value); answer digits (after the terminal: '?' if present, else '=') = 1 + index from the answer's start (causal); clamp at max_places (24). Added to token embedding; row 0 zero. Reversed answers make answer index = place value.
- **reverse_answer**: `FrozenData(reverse_answer=True)` / `train.py --reverse_answer`: answer digits served smallest-first, sign first, '#' last; files unchanged.
- **nested_bal**: dataset with `op_digits` (leaf digit counts 1..op_digits; depth 1–2: {1,2,3}, depth 3: {1,2}, ≥4: {1}) and training balanced across feasible `ans_digits` buckets per (depth, op_digits); test cells up to 1,000 per (op_digits, ans_digits).
- **holdout**: nested levels 1–2: 20% of distinct prompts excluded from train; test = held-out only. Overlap 0 at all levels.
- **FrozenData.batch(rng, bs, (lo, hi))**: uniform over levels in range, then uniform rows.

## Training / eval (`src/looptf/toy/train.py`)
- **answer_loss**: CE over positions pos..pos+n_tgt-1 predicting x[t+1]; exact = all those argmax-correct (== greedy decode exact match).
- **curriculum**: start at depth ≤ `cur_start` (2); every `cur_every` (250) steps, eval 256 held-in train rows at the top depth at K = max train loops; ≥ `cur_threshold` (0.9) → add next depth (cap `train_depth` hi, 8).
- **train_loops**: `k` fixed or `lo-hi` (uniform per step).
- **eval**: every `eval_every` (2000) steps on test sets, depths 1–16, loops [1] for K=1 baselines else {1,2,4,8,12,16} ∪ {K_max}. End: `per_example.npz` (K, correct, labels).
- **defaults used by the grid**: lr 3e-4, AdamW (0.9, 0.95), wd 0.1 on ≥2-D params, warmup 500, cosine to 10%, bs 256, 40k steps, fp32 weights + bf16 autocast, grad clip 1.0.

## Experiment conventions
- **run name**: `grid/c{core}k{loops}-s{seed}`; outputs `results/toy/<name>/{config.json,log.jsonl,per_example.npz}`, stdout `results/toy/<name>.out`, checkpoint `checkpoints/toy/<name>.pt`.
- **diagonal**: cells with equal effective depth (compute-matched). **row**: equal params.
- **decision card**: entry in `docs/decisions.md` (question, options + what each outcome means, recommendation, owner, status). Required before any experiment or design change that alters what a result means; see skill `decision-brief`.
- **status entry**: `docs/status.md`, via skill `status-update`.
