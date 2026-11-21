"""Heal a looped model by distilling from the original (unlooped) model.

Teacher = original pretrained model (frozen). Student = same weights, rewired with a
LoopSpec. Loss = alpha * KL(teacher || student) + (1 - alpha) * next-token CE on raw text.
"""

from __future__ import annotations

import itertools
import random
import math
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from datasets import load_dataset

from .model import LoopSpec, apply_loop, describe, load_model, save_looped, set_n_loops


@dataclass
class TrainConfig:
    model: str = "Qwen/Qwen3-0.6B-Base"
    loop: str = "4:16:2:16"
    gated: bool = False  # Track A: extra passes start as no-ops (gate = 0)
    k_choices: str = ""  # e.g. "2,3,4": sample the number of loop passes per step
    teacher: str = ""  # defaults to `model` (distill from the unlooped original)
    gate_lr: float = 1e-2
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"
    seq_len: int = 1024
    batch_size: int = 2
    grad_accum: int = 16
    lr: float = 1e-4
    warmup: int = 100
    steps: int = 2000
    alpha: float = 0.9  # weight on distillation KL
    temperature: float = 1.0
    train_only_loop: bool = False  # freeze prelude/coda/embeddings, train only looped layers
    grad_checkpointing: bool = True
    log_every: int = 10
    save_every: int = 500
    out: str = "checkpoints/run"
    extra: dict = field(default_factory=dict)


def token_stream(tok, cfg: TrainConfig):
    ds = load_dataset(cfg.dataset, cfg.dataset_config, split="train", streaming=True)
    buf: list[int] = []
    n = cfg.seq_len + 1
    for ex in ds:
        buf.extend(tok(ex["text"])["input_ids"] + [tok.eos_token_id])
        while len(buf) >= n * cfg.batch_size:
            chunk = torch.tensor(buf[: n * cfg.batch_size]).view(cfg.batch_size, n)
            buf = buf[n * cfg.batch_size :]
            yield chunk


def gate_str(model) -> str:
    gates = getattr(model.model, "loop_gates", None)
    if gates is None:
        return ""
    return "gates " + "/".join(f"{r.abs().mean():.3f}" for r in gates.weight.data) + " "


def train(cfg: TrainConfig):
    torch.manual_seed(0)
    random.seed(0)
    teacher, tok = load_model(cfg.teacher or cfg.model)
    teacher.eval().requires_grad_(False)
    student, _ = load_model(cfg.model)
    spec = LoopSpec.parse(cfg.loop, gated=cfg.gated)
    ks = [int(k) for k in cfg.k_choices.split(",")] if cfg.k_choices else [spec.n_loops]
    spec.n_loops = max(ks)  # build gates for the largest K up front
    apply_loop(student, spec)
    print("teacher:", describe(teacher))
    print("student:", describe(student))

    if cfg.train_only_loop:
        student.requires_grad_(False)
        for i in range(spec.loop_start, spec.loop_end):
            for slot in student.model.layers:
                if slot.source_idx == i:
                    slot.layer.requires_grad_(True)
                    break
    if cfg.grad_checkpointing:
        student.gradient_checkpointing_enable()
        student.config.use_cache = False
    student.train()

    params = [p for p in student.parameters() if p.requires_grad]
    gates = getattr(student.model, "loop_gates", None)
    gate_ids = {id(p) for p in gates.parameters()} if gates is not None else set()
    groups = [{"params": [p for p in params if id(p) not in gate_ids], "lr": cfg.lr}]
    if gate_ids:
        groups.append({"params": [p for p in params if id(p) in gate_ids], "lr": cfg.gate_lr})
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.1f}M  K choices: {ks}")
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / cfg.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)),
    )

    data = token_stream(tok, cfg)
    t0 = time.time()
    for step in range(cfg.steps):
        stats = {"kl": 0.0, "ce": 0.0}
        k = random.choice(ks)
        if len(ks) > 1:
            set_n_loops(student, k)
        for batch in itertools.islice(data, cfg.grad_accum):
            batch = batch.cuda()
            x, y = batch[:, :-1], batch[:, 1:]
            with torch.no_grad():
                t_logits = teacher(x).logits.float()
            s_logits = student(x).logits.float()
            T = cfg.temperature
            kl = F.kl_div(
                F.log_softmax(s_logits / T, -1).flatten(0, 1),
                F.log_softmax(t_logits / T, -1).flatten(0, 1),
                log_target=True,
                reduction="batchmean",
            ) * T * T
            ce = F.cross_entropy(s_logits.flatten(0, 1), y.flatten())
            loss = (cfg.alpha * kl + (1 - cfg.alpha) * ce) / cfg.grad_accum
            loss.backward()
            stats["kl"] += kl.item() / cfg.grad_accum
            stats["ce"] += ce.item() / cfg.grad_accum
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            tps = (step + 1) * cfg.grad_accum * cfg.batch_size * cfg.seq_len / (time.time() - t0)
            print(
                f"step {step} K {k} kl {stats['kl']:.4f} ce {stats['ce']:.4f} gnorm {gnorm:.2f} "
                f"lr {sched.get_last_lr()[0]:.2e} {gate_str(student)}tok/s {tps:.0f} mem {torch.cuda.max_memory_allocated()/2**30:.1f}G",
                flush=True,
            )
        if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.steps:
            set_n_loops(student, max(ks))
            save_looped(student, cfg.model, f"{cfg.out}/step{step+1}.pt")
            print(f"saved {cfg.out}/step{step+1}.pt")
