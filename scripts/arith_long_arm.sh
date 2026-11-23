#!/usr/bin/env bash
# D-012 extra arm: same as arith_long.sh but with a name suffix, so arms don't collide.
#   scripts/arith_long_arm.sh "<recipe flags>" <suffix>   e.g. "--place_emb --reverse_answer" rev
# Optional: WAIT_PID=<pid> waits for that process (e.g. the running queue) before starting.
set -euo pipefail
cd "$(dirname "$0")/.."
FLAGS=${1:?recipe flags}; SUF=${2:?name suffix}
if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
mkdir -p results/toy/long
run() {  # name core loops seed
  local name=long/$1-$SUF-s$4
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
