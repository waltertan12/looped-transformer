# loop-transformer

Take a pretrained Qwen model, cut it down, and re-run a block of its layers in a loop
(shared weights) to get depth without parameters. Measure whether looping recovers or
beats the original on benchmarks, and how accuracy scales with the number of loops.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -e ".[dev]"
source .venv/bin/activate
```

Hardware: one RTX 5090 (32 GB, power-limited to 450 W). Keep runs single-GPU.

## Loop spec

`--loop start:end:n_loops[:coda_start]` → prelude `[0,start)`, loop `[start,end)` × n, coda `[coda_start, L)`.
Layers in `[end, coda_start)` are dropped. Example for Qwen3-0.6B (28 layers):

- `4:16:2:28` — keep the first 16 layers, loop 4–15 twice → depth 28, 407M unique params (vs 596M)
- `0:28:1` — identity (matches the base model exactly; see tests)

## Usage

```bash
# 1. Baseline benchmark (any lm-eval task: arc_easy, hellaswag, mmlu, gsm8k, humaneval, ...)
python scripts/eval.py --model Qwen/Qwen3-0.6B-Base --tasks arc_easy,hellaswag --limit 300

# 2. Same model, looped, no training
python scripts/eval.py --tasks arc_easy,hellaswag --limit 300 --loop 4:16:2:28

# 3. Heal the looped model by distilling from the original
python scripts/train.py configs/qwen3-0.6b-half-loop2.yaml

# 4. Evaluate the trained checkpoint
python scripts/eval.py --checkpoint checkpoints/qwen3-0.6b-half-loop2/step2000.pt --tasks arc_easy,hellaswag

pytest tests
```

Results land in `results/*.json`.

## Initial numbers (Qwen3-0.6B-Base, 0-shot, 300 examples each)

| model | unique params | arc_easy acc | hellaswag acc_norm |
|---|---|---|---|
| base | 596M | 0.617 | 0.523 |
| loop 4:16:2:28, untrained | 407M | 0.357 | 0.297 |

Training is needed. Chance is about 0.25.

## Layout

- `src/looptf/model.py`: `LoopSpec`, `apply_loop` (rewires `model.model.layers` into weight-shared `LoopSlot`s, each with its own KV-cache index), save/load
- `src/looptf/evaluate.py`: lm-evaluation-harness wrapper
- `src/looptf/train.py`: distillation (KL to the frozen teacher + CE)
- `configs/`: training configs
- `docs/research-log.md`: running log of the research thinking and results

## Next ideas

- Vary `n_loops` at eval time on a trained checkpoint (does more loops = better?)
- Train with randomly sampled `n_loops` per step so the model generalizes across depths
- Train only the loop block (`train_only_loop: true`), or add LoRA per loop iteration
- Larger models (Qwen3-1.7B/4B) and harder benchmarks (gsm8k, mbpp/humaneval; SWE-bench needs an agent harness + docker and comes later)
