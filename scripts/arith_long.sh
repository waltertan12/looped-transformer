#!/usr/bin/env bash
# D-012: long probe set. shallow 2x1, looped 2x8, deep 16x1 x seeds 0,1; 120k steps.
#   scripts/arith_long.sh "<recipe flags>"   e.g. "--place_emb" or "--place_emb --reverse_answer"
# Two runs at a time, paired by model (both seeds together).
set -euo pipefail
cd "$(dirname "$0")/.."
FLAGS=${1:?recipe flags}
mkdir -p results/toy/long
run() {  # name core loops seed
  local name=long/$1-s$4
  .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested_bal $FLAGS \
    --n_core "$2" --train_loops "$3" --train_depth 1-8 --lr 3e-4 --steps 120000 --seed "$4" \
    --mem_fraction 0.4 --eval_every 6000 \
    > "results/toy/$name.out" 2>&1 && echo "done $name" || echo "FAILED $name"
}
for model in "c2k1 2 1" "c2k8 2 8" "c16k1 16 1"; do
  set -- $model
  run "$1" "$2" "$3" 0 &
  run "$1" "$2" "$3" 1 &
  wait
done
