#!/usr/bin/env bash
# D-007 on nested_bal: looped 2x8 vs deep 16x1, normal vs reversed answers,
# mixed depths 1-8 (no curriculum, D-006). Two runs at a time (Walter's preference),
# each capped at 40% of GPU memory.
#   scripts/arith_bal.sh SEED
# Wave 1 (looped, normal vs reversed) answers D-007 first; wave 2 repeats it for deep.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?seed}
mkdir -p results/toy/bal
run() {  # name core loops [extra args]
  local name=bal/$1-s$SEED; shift; local c=$1 k=$2; shift 2
  .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested_bal \
    --n_core "$c" --train_loops "$k" --train_depth 1-8 --lr 3e-4 --steps 40000 --seed "$SEED" \
    --mem_fraction 0.4 "$@" \
    > "results/toy/$name.out" 2>&1 && echo "done $name" || echo "FAILED $name"
}
run c2k8-normal 2 8 &
run c2k8-rev 2 8 --reverse_answer &
wait
run c16k1-normal 16 1 &
run c16k1-rev 16 1 --reverse_answer &
wait
