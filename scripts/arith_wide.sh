#!/usr/bin/env bash
# D-015 / H16: widths 256 / 384 / 512 x shallow / looped / deep, no tags, seed 0, 120k schedule.
#   scripts/arith_wide.sh pause     # train each run to 40k, then pause (saves state + @40k snapshot)
#   scripts/arith_wide.sh resume80  # continue each paused run to 80k, pause again (@80k snapshot)
#   scripts/arith_wide.sh resume    # continue each paused run to 120k (final checkpoint)
# Full state is also saved every 10k steps, so a crash loses at most 10k steps: rerun the same mode.
# 2 runs at a time, longest first.
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?pause, resume80 or resume}
SEED=${SEED:-0}
mkdir -p results/toy/wide
job() {  # model width
  local m=$1 w=$2 core loops
  case $m in c2k1) core=2 loops=1 ;; c2k8) core=2 loops=8 ;; c16k1) core=16 loops=1 ;; esac
  local name=wide/$m-w$w-s$SEED extra
  case $MODE in
    pause) extra="--pause_at 40000" ;;
    resume80) extra="--resume --pause_at 80000" ;;
    resume) extra="--resume" ;;
    *) echo "unknown mode $MODE"; return 1 ;;
  esac
  nice -n 5 .venv/bin/python -m looptf.toy.train --name "$name" --data data/nested_bal \
    --d_model "$w" --d_ff $((4 * w)) --n_heads 4 --n_core "$core" --train_loops "$loops" \
    --train_depth 1-8 --lr 3e-4 --steps 120000 --seed "$SEED" --mem_fraction 0.45 --eval_every 6000 \
    --save_every 10000 $extra >> "results/toy/$name.out" 2>&1 && echo "done $name ($MODE)" || echo "FAILED $name ($MODE)"
}
export -f job; export MODE SEED
printf '%s\n' "c16k1 512" "c2k8 512" "c16k1 384" "c2k8 384" "c16k1 256" "c2k8 256" \
  "c2k1 512" "c2k1 384" "c2k1 256" | xargs -P 2 -L 1 bash -c 'job $0 $1'
