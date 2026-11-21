#!/usr/bin/env bash
# Experiment 3: does the loop help once it's actually switched on?
# exp2's dials (gates) started at 0 and stayed ~0, so the extra passes went unused.
# Same setup as exp2's looped run (layers 8-19, teacher 1.7B, K in {2,3,4}), 500 steps,
# but three ways of getting the extra passes used:
#   forced   - no dials: every extra pass is applied in full
#   half     - dials start at 0.5 and are learned
#   fastdial - dials start at 0 but learn 10x faster (gate_lr 0.1)
# Each run is scored at K=1, 2, 4. Compare against exp2's no-loop control.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TASKS=arc_easy,arc_challenge,hellaswag,gsm8k
LIMIT=300
OUT=results/exp3
STEPS=${STEPS:-500}
mkdir -p $OUT
log() { echo "[$(date +%H:%M:%S)] $*"; }
ev() { $PY scripts/eval.py --tasks $TASKS --limit $LIMIT --out $OUT "$@" 2>&1 | grep -E "^\||memory|saved|Error"; }
mem() { $PY scripts/memory.py --out $OUT/memory.jsonl "$@" 2>&1 | grep -E "^\{|Error"; }
COMMON="--model Qwen/Qwen3-0.6B-Base --teacher Qwen/Qwen3-1.7B-Base --loop 8:20:4 --k-choices 2,3,4 --steps $STEPS --save-every $STEPS --log-every 25"

declare -A ARGS=(
  [forced]="--gated false"
  [half]="--gated true --gate-init 0.5"
  [fastdial]="--gated true --gate-init 0 --gate-lr 0.1"
)
for name in forced half fastdial; do
  log "train $name (${ARGS[$name]})"
  $PY -u scripts/train.py $COMMON ${ARGS[$name]} --out checkpoints/exp3-$name 2>&1 \
    | grep --line-buffered -E "^step|saved|trainable|Error" > $OUT/train_$name.log
  for k in 1 2 4; do
    log "eval $name K=$k"
    ev --checkpoint checkpoints/exp3-$name/step$STEPS.pt --n-loops $k
    mem --checkpoint checkpoints/exp3-$name/step$STEPS.pt --n-loops $k
  done
done
log "done"
