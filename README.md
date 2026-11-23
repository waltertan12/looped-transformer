# looped-transformer

Does running the same transformer layers several times in a row (a **looped transformer**) add
reasoning depth? This project trains tiny transformers from scratch on nested PEMDAS arithmetic,
where difficulty is controlled exactly, and compares three models:

| model | core | weights | layers run |
|---|---|---|---|
| shallow | 2 blocks, once | 4.4M | 4 |
| looped | the same 2 blocks, 8 times | 4.4M | 18 |
| deep | 16 distinct blocks, once | 19M | 18 |

Looped vs shallow asks whether repeating helps; looped vs deep asks whether repeating is as good
as more distinct layers at the same compute.

## Results so far

120k training steps, two seeds each. Full write-up with charts and a primer on every term:
**[results page](https://waltertan12.github.io/looped-transformer/visual/loop-depth-results.html)**
(all docs: [waltertan12.github.io/looped-transformer](https://waltertan12.github.io/looped-transformer/)). Tables: [`docs/results-ops.md`](docs/results-ops.md),
[`docs/hypotheses.md`](docs/hypotheses.md).

- **Loops buy steps.** On many-step problems with small answers, looped beats shallow by +0.11
  (4 steps) to +0.25 (8 steps) and matches deep within ±0.03, including past the training depth.
- **Weights buy number size.** On big answers and big inputs, deep beats looped by +0.05 to +0.15,
  with or without multiplication.
- **Operator mix doesn't matter** once depth and number size are fixed.
- **Place-value tags break on longer numbers.** No model handles 4-digit inputs; with tags the
  answers are garbage, without tags they are approximately right.
- **A model trained at 8 loops works only at 8 loops.**

Closest prior work: Saunshi et al., *Reasoning with Latent Thoughts* (ICLR 2025) and McLeish et al.,
*Transformers Can Do Arithmetic with the Right Embeddings* (NeurIPS 2024). See `docs/related-work.md`.

## The task

Problems are written as characters and answered one character at a time, ending with `#`:

```
7*8=56#
(8-3)*(4-6)+7=-3#
352+47=399#
```

Two independent difficulty dials: **depth** (operations that must happen in sequence) and
**number size** (digits in the inputs and answer). Answers are exact integers, unbounded.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e ".[dev]"
```

Hardware used: one RTX 5090 (32 GB). Each 120k-step run takes 0.6–2.5 h.

## Reproduce

```bash
# Build the frozen datasets (deterministic) and check every answer independently
.venv/bin/python scripts/build_data.py --which nested_bal --out data     # ~6 min, 2.6 GB
.venv/bin/python scripts/verify_data.py --root data/nested_bal
.venv/bin/python scripts/build_ops.py --kind add --out data/nested_add  # test-only sets
.venv/bin/python scripts/build_ops.py --kind sub --out data/nested_sub

# Train one model (looped 2x8, place-value tags, 120k steps)
.venv/bin/python -m looptf.toy.train --name long/c2k8-s0 --data data/nested_bal --place_emb \
  --n_core 2 --train_loops 8 --train_depth 1-8 --lr 3e-4 --steps 120000 --seed 0

# The main experiment: shallow / looped / deep x 2 seeds
scripts/arith_long.sh "--place_emb"

# Score saved models on the add-only / sub-only sets, then write the report
scripts/eval_ops.sh && .venv/bin/python scripts/ops_report.py --out docs/results-ops.md

.venv/bin/python -m pytest tests -q -k toy
```

Each run writes `results/toy/<name>/` (config, training log, per-example correctness) and a
checkpoint in `checkpoints/toy/`.

## Layout

- `src/looptf/toy/`: data generators (`data.py`, `chain.py`, `ops_data.py`), frozen dataset loader
  (`dataset.py`), model (`model.py`: prelude / looped core / coda, input injection, place-value tags),
  training (`train.py`)
- `scripts/`: dataset build and verify, experiment runners (`arith_*.sh`), scoring and reports
- `docs/`: start with `HANDOFF.md`. `hypotheses.md` (what's tested, what's open), `plan.md`
  (next steps with decision rules written in advance), `decisions.md`, `research-log.md`
  (chronological), `history.md` (how the project's direction changed), `glossary.md`
- `src/looptf/model.py`, `evaluate.py`, `train.py`, `configs/`: the earlier Qwen-retrofit track
  (paused), see `docs/history.md`
