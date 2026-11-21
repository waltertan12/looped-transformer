#!/usr/bin/env bash
# Experiment 3: which layers should loop? Same recipe as exp2's looped run
# (gated, teacher 1.7B, K in {2,3,4}), shorter (500 steps), different core blocks.
# Each run is scored at K=1 (no extra passes) and K=4.
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

# name:loop_spec   (Qwen3-0.6B has 28 layers)
for run in early:4:16:4 late:12:24:4 small:10:16:4; do
  name=${run%%:*}; spec=${run#*:}
  log "train $name ($spec)"
  $PY scripts/train.py --model Qwen/Qwen3-0.6B-Base --teacher Qwen/Qwen3-1.7B-Base \
    --loop $spec --gated true --k-choices 2,3,4 --steps $STEPS --save-every $STEPS --log-every 25 \
    --out checkpoints/exp3-$name 2>&1 | grep -E "^step|saved|trainable|Error" > $OUT/train_$name.log
  for k in 1 4; do
    log "eval $name K=$k"
    ev --checkpoint checkpoints/exp3-$name/step$STEPS.pt --n-loops $k
    mem --checkpoint checkpoints/exp3-$name/step$STEPS.pt --n-loops $k
  done
done
log "done"
