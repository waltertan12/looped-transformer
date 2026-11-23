#!/usr/bin/env bash
# H13: score every D-012 checkpoint on the add-only / sub-only test sets, two jobs at a time.
#   scripts/eval_ops.sh   -> results/toy/ops/<run>-<add|sub>/per_example.npz
set -euo pipefail
cd "$(dirname "$0")/.."
runs=(c2k1-s0 c2k1-s1 c2k8-s0 c2k8-s1 c16k1-s0 c16k1-s1 c2k1-rev-s0 c2k1-rev-s1 c2k8-rev-s0 c2k8-rev-s1 c16k1-rev-s0 c16k1-rev-s1)
job() {  # run kind
  .venv/bin/python scripts/eval_ckpt.py --ckpt "checkpoints/toy/long/$1.pt" --data "data/nested_$2" \
    --out "results/toy/ops/$1-$2/per_example.npz" --mem_fraction 0.4 && echo "done $1-$2" || echo "FAILED $1-$2"
}
for r in "${runs[@]}"; do
  job "$r" add &
  job "$r" sub &
  wait
done
