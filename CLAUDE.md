# loop-transformer

Research repo for turning pretrained Qwen models into looped (weight-shared) transformers. See README.md.

- Lives at `/mnt/data/walter/loop-transformer` (spare SSD). HF models/datasets cache: `/mnt/data/walter/huggingface` (set `HF_HOME` to it, or symlink `~/.cache/huggingface`).
- Python venv at `.venv`; run things with `.venv/bin/python` or after `source .venv/bin/activate`.
- GPU: a single RTX 5090, 32 GB, power-limited to 450 W. Everything must fit on that card.
- Train memory is dominated by the fp32 full-vocab logits (152k vocab). Raise `grad_accum` instead of `batch_size`.
- transformers 5.x: Qwen3 attention uses `self_attn.layer_idx` for the KV cache. `LoopSlot` overwrites it per slot, and `apply_loop` updates `config.num_hidden_layers` and `config.layer_types`.
- Run `pytest tests` after touching `model.py`. The identity-loop test must match the base logits exactly.
- `results/` and `checkpoints/` are gitignored.

## Working with Walter
- **New session? Read `docs/HANDOFF.md` first**: current state, what is running, next actions.
- Walter is newer to ML research and wants to see the key calls, not have them made silently.
  Before any experiment or design change that alters what a result means (or spends >30 min of
  GPU), write a decision card with the `decision-brief` skill (`docs/decisions.md`) and ask.
  Routine execution proceeds without asking.
- Status snapshots: `status-update` skill → `docs/status.md`. Reasoning: `docs/research-log.md`.
- Forward plan with pre-registered decision rules: `docs/plan.md`. Check results against its
  rules; revise it (and say why) when a step finishes.
- Keep `docs/glossary.md` (plain language) and `docs/glossary-agent.md` (terse) in sync when a
  new term comes up. Explain in plain language, using glossary terms.
- Current track: tiny from-scratch arithmetic models in `src/looptf/toy/` (the Qwen sections
  above describe the paused retrofit track).
