#!/usr/bin/env python
"""Distill a looped student from the original model.

  python scripts/train.py configs/qwen3-0.6b-half-loop2.yaml
  python scripts/train.py configs/qwen3-0.6b-half-loop2.yaml --steps 20   # smoke test
"""
import argparse
import dataclasses

import yaml

from looptf.train import TrainConfig, train

p = argparse.ArgumentParser()
p.add_argument("config", nargs="?")
for f in dataclasses.fields(TrainConfig):
    if f.name != "extra":
        p.add_argument(f"--{f.name.replace('_', '-')}", dest=f.name, default=None)
a = vars(p.parse_args())

cfg = TrainConfig(**(yaml.safe_load(open(a.pop("config"))) if a.get("config") else {}))
for k, v in a.items():
    if v is not None:
        typ = type(getattr(cfg, k))
        setattr(cfg, k, v.lower() in ("1", "true", "yes") if typ is bool else typ(v))
train(cfg)
