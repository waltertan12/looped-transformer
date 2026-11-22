#!/usr/bin/env bash
# Arithmetic grid: unique core blocks x loops, on the frozen nested dataset.
#   scripts/arith_grid.sh SEED [MAX_PARALLEL]
# Fixed-K cells answer "same compute, different weights" (the anti-diagonals); the two
# random-K runs answer "does adding loops at test time help on deeper problems?".
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=${1:?seed}
PAR=${2:-6}
mkdir -p results/toy/grid
# core loops — longest first so the queue packs well
CELLS="2:8 4:4 8:2 16:1 2:1-8 4:1-4 2:4 4:2 8:1 2:2 4:1 2:1"
for cell in $CELLS; do
  c=${cell%%:*}; k=${cell#*:}
  echo "$c $k"
done | xargs -P "$PAR" -L 1 bash -c '
  c=$0; k=$1; name=grid/c${c}k${k}-s'"$SEED"'
  .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested --curriculum \
    --n_core "$c" --train_loops "$k" --lr 3e-4 --steps 40000 --seed '"$SEED"' \
    > "results/toy/$name.out" 2>&1 && echo "done $name" || echo "FAILED $name"'
