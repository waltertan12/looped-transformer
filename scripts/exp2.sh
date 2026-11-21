#!/usr/bin/env bash
# Experiment 2 (Track A): can looping a 0.6B model close the gap to the 1.7B?
#   - baselines: 0.6B and 1.7B, unchanged
#   - control:   0.6B distilled from 1.7B, no loops
#   - looped:    0.6B distilled from 1.7B, gated loop of layers 8-19, K in {2,3,4}
# Each model is benchmarked and memory-profiled; the looped one at K = 1,2,3,4,6.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TASKS=arc_easy,arc_challenge,hellaswag,gsm8k
LIMIT=300
OUT=results/exp2
STEPS=${STEPS:-1000}
mkdir -p $OUT
log() { echo "[$(date +%H:%M:%S)] $*"; }

ev() { $PY scripts/eval.py --tasks $TASKS --limit $LIMIT --out $OUT "$@" 2>&1 | grep -E "^\||memory|saved|Error"; }
mem() { $PY scripts/memory.py --out $OUT/memory.jsonl "$@" 2>&1 | grep -E "^\{|Error"; }

log "baselines"
for m in Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base; do
  ev --model $m; mem --model $m
done

log "train control (no loops)"
$PY scripts/train.py --model Qwen/Qwen3-0.6B-Base --teacher Qwen/Qwen3-1.7B-Base \
  --loop 0:28:1 --steps $STEPS --save-every 500 --log-every 25 --out checkpoints/exp2-control \
  2>&1 | grep -E "^step|saved|trainable|Error" > $OUT/train_control.log
ev --checkpoint checkpoints/exp2-control/step$STEPS.pt
mem --checkpoint checkpoints/exp2-control/step$STEPS.pt

log "train looped (gated 8-19, K in 2,3,4)"
$PY scripts/train.py --model Qwen/Qwen3-0.6B-Base --teacher Qwen/Qwen3-1.7B-Base \
  --loop 8:20:4 --gated true --k-choices 2,3,4 --steps $STEPS --save-every 500 --log-every 25 \
  --out checkpoints/exp2-looped 2>&1 | grep -E "^step|saved|trainable|Error" > $OUT/train_looped.log
for k in 1 2 3 4 6; do
  log "eval looped K=$k"
  ev --checkpoint checkpoints/exp2-looped/step$STEPS.pt --n-loops $k
  mem --checkpoint checkpoints/exp2-looped/step$STEPS.pt --n-loops $k
done
log "done"
