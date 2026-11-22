#!/usr/bin/env bash
# D-007 + D-010 probe set on nested_bal: normal vs reversed answers for
# looped 2x8, shallow 2x1 (no-loop baseline), deep 16x1. Mixed depths 1-8, no curriculum.
# Two runs at a time (Walter's preference), each capped at 40% of GPU memory.
#   scripts/arith_bal.sh SEED [FIRST_WAVE]     waves: 1 looped, 2 shallow (fast), 3 deep
# FIRST_WAVE lets a queue resume after an earlier wave already ran.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?seed}
FIRST=${2:-1}
mkdir -p results/toy/bal
run() {  # name core loops [extra args]
  local name=bal/$1-s$SEED; shift; local c=$1 k=$2; shift 2
  .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested_bal \
    --n_core "$c" --train_loops "$k" --train_depth 1-8 --lr 3e-4 --steps 40000 --seed "$SEED" \
    --mem_fraction 0.4 "$@" \
    > "results/toy/$name.out" 2>&1 && echo "done $name" || echo "FAILED $name"
}
if [ "$FIRST" -le 1 ]; then
  run c2k8-normal 2 8 &
  run c2k8-rev 2 8 --reverse_answer &
  wait
fi
if [ "$FIRST" -le 2 ]; then
  run c2k1-normal 2 1 &
  run c2k1-rev 2 1 --reverse_answer &
  wait
fi
if [ "$FIRST" -le 3 ]; then
  run c16k1-normal 16 1 &
  run c16k1-rev 16 1 --reverse_answer &
  wait
fi
