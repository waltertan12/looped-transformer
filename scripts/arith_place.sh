#!/usr/bin/env bash
# D-011 screen: place-value tags (--place_emb), normal vs reversed answers, on shallow 2x1 and
# looped 2x8. No-tag baselines come from scripts/arith_bal.sh (same data, steps, lr, seed).
# Two runs at a time; shallow first (fast signal).   scripts/arith_place.sh SEED
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?seed}
mkdir -p results/toy/place
run() {  # name core loops [extra args]
  local name=place/$1-s$SEED; shift; local c=$1 k=$2; shift 2
  .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested_bal --place_emb \
    --n_core "$c" --train_loops "$k" --train_depth 1-8 --lr 3e-4 --steps 40000 --seed "$SEED" \
    --mem_fraction 0.4 "$@" \
    > "results/toy/$name.out" 2>&1 && echo "done $name" || echo "FAILED $name"
}
run c2k1-normal-pv 2 1 &
run c2k1-rev-pv 2 1 --reverse_answer &
wait
run c2k8-normal-pv 2 8 &
run c2k8-rev-pv 2 8 --reverse_answer &
wait
