# Qwen retrofit (stopped, 21–22 Nov 2026)

**What it was.** The first approach to the loop question: take a pretrained Qwen3-0.6B, cut it
down, and run a block of its layers in a loop (shared weights), hoping to match the original's
benchmark scores with fewer unique weights. Experiments exp1–exp3.

**Status.** Stopped. Not a goal any more (Walter, 2026-11-24). Kept as a record.

**Why it stopped.** Looping collapsed accuracy, training switched the loop off by itself, and no
recipe beat the no-loop control. A pretrained model mixes too many effects to see whether looping
adds reasoning depth. Full story: `../README.md` (Phase 1).

## Files here

| file | holds |
|---|---|
| `log-2026-11-21.md` | the research-log entries from this era, unchanged |
| `playbook.md` | the original pasted playbook (TinyLlama → protein LMs), with gaps marked |

Decision D-001 (exp2 dials start at zero) is in `../../decisions.md`.

## Code left in the repo

`src/looptf/model.py`, `src/looptf/train.py`, `src/looptf/evaluate.py`, `configs/`,
`scripts/train.py`, `scripts/eval.py`, `scripts/exp2.sh`, `scripts/exp3.sh`, `scripts/memory.py`,
`tests/test_loop.py`. Old outputs: `results/*.json`, `results/exp*`.

## Technical notes (moved from the root CLAUDE.md)

- The models cache is `/mnt/data/walter/huggingface`. Set `HF_HOME` to it.
- Train memory is dominated by the fp32 full-vocab logits (152k vocab). Raise `grad_accum`, not
  `batch_size`.
- transformers 5.x: Qwen3 attention uses `self_attn.layer_idx` for the KV cache. `LoopSlot`
  overwrites it per slot. `apply_loop` updates `config.num_hidden_layers` and `config.layer_types`.
- After you change `model.py`, run `pytest tests/test_loop.py`. The identity-loop test must match
  the base logits exactly.
