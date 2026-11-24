#!/usr/bin/env bash
# One-off (24 Nov): the two width-512 runs crashed when pausing at 40k (missing checkpoint folder,
# fixed in train.py). Waits for the running pause queue, then reruns them from scratch, 2 at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "arith_wide.sh pause" >/dev/null; do sleep 60; done
for m in c16k1 c2k8; do
  case $m in c2k8) core=2 loops=8 ;; c16k1) core=16 loops=1 ;; esac
  name=wide/$m-w512-s0
  ( nice -n 5 .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested_bal \
      --d_model 512 --d_ff 2048 --n_heads 4 --n_core "$core" --train_loops "$loops" \
      --train_depth 1-8 --lr 3e-4 --steps 120000 --seed 0 --mem_fraction 0.45 --eval_every 6000 \
      --pause_at 40000 > "results/toy/$name.out" 2>&1 && echo "done $name (pause)" || echo "FAILED $name (pause)" ) &
done
wait
