#!/usr/bin/env bash
# H16: score wide checkpoints on the add-only / subtract-only test sets (for the breakdown page).
#   scripts/wide_ops.sh 40k|80k|120k   -> results/toy/wide-ops/<run>@<at>-<add|sub>/per_example.npz
# One job at a time, small memory share, niced: it shares the GPU with training. Skips finished ones.
set -euo pipefail
cd "$(dirname "$0")/.."
AT=${1:?40k, 80k or 120k}
for ck in checkpoints/toy/wide/*-s0$([ "$AT" = 120k ] && echo "" || echo "@$AT").pt; do
  [ -e "$ck" ] || continue
  run=$(basename "$ck" .pt); run=${run%@*}
  for kind in add sub; do
    out=results/toy/wide-ops/$run@$AT-$kind/per_example.npz
    [ -e "$out" ] && continue
    nice -n 10 .venv/bin/python scripts/eval_ckpt.py --ckpt "$ck" --data "data/nested_$kind" --out "$out" \
      --mem_fraction 0.25 > /dev/null 2>&1 && echo "done $run@$AT-$kind" || echo "FAILED $run@$AT-$kind"
  done
done
