#!/usr/bin/env bash
# D-014 screen (H18): tag fixes for longer numbers, on shallow 2x1 and looped 2x8, 40k steps.
# Same settings as the D-011 screen (scripts/arith_place.sh), so its runs are the baselines:
# tags without fix = place/*-normal-pv-s0, no tags = bal/*-normal-s0.
# Two runs at a time; shallow first.   scripts/arith_tags.sh SEED
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?seed}
mkdir -p results/toy/tags
run() {  # name core loops [extra args]
  local name=tags/$1-s$SEED; shift; local c=$1 k=$2; shift 2
  .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested_bal --place_emb \
    --n_core "$c" --train_loops "$k" --train_depth 1-8 --lr 3e-4 --steps 40000 --seed "$SEED" \
    --mem_fraction 0.4 "$@" \
    > "results/toy/$name.out" 2>&1 && echo "done $name" || echo "FAILED $name"
}
run c2k1-offset 2 1 --tag_offset 8 &
run c2k1-sep 2 1 --sep_tags &
wait
run c2k8-offset 2 8 --tag_offset 8 &
run c2k8-sep 2 8 --sep_tags &
wait
