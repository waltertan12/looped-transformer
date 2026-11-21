# loop-transformer

Research repo for turning pretrained Qwen models into looped (weight-shared) transformers. See README.md.

- Python venv at `.venv`; run things with `.venv/bin/python` or after `source .venv/bin/activate`.
- GPU: a single RTX 5090, 32 GB, power-limited to 450 W. Everything must fit on that card.
- Train memory is dominated by the fp32 full-vocab logits (152k vocab). Raise `grad_accum` instead of `batch_size`.
- transformers 5.x: Qwen3 attention uses `self_attn.layer_idx` for the KV cache. `LoopSlot` overwrites it per slot, and `apply_loop` updates `config.num_hidden_layers` and `config.layer_types`.
- Run `pytest tests` after touching `model.py`. The identity-loop test must match the base logits exactly.
- `results/` and `checkpoints/` are gitignored.
