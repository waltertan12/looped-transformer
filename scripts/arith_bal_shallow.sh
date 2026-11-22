#!/usr/bin/env bash
# D-010 baseline for D-007: shallow 2x1 (no loop), normal vs reversed, on nested_bal.
# Waits for scripts/arith_bal.sh's queue to finish so only two runs share the GPU.
#   scripts/arith_bal_shallow.sh SEED
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?seed}
until [ "$(grep -cE '^(done|FAILED)' results/toy/bal-s$SEED.queue.log 2>/dev/null || true)" -ge 4 ]; do sleep 60; done
run() {
  local name=bal/$1-s$SEED; shift; local c=$1 k=$2; shift 2
  .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested_bal \
    --n_core "$c" --train_loops "$k" --train_depth 1-8 --lr 3e-4 --steps 40000 --seed "$SEED" \
    --mem_fraction 0.4 "$@" \
    > "results/toy/$name.out" 2>&1 && echo "done $name" || echo "FAILED $name"
}
run c2k1-normal 2 1 &
run c2k1-rev 2 1 --reverse_answer &
wait
