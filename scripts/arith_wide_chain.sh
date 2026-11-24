#!/usr/bin/env bash
# H16 (D-015), Walter 24 Nov: after every run is paused at 40k, continue all 9 to 80k (pause),
# then all 9 to 120k. Waits for the width-512 reruns to reach their 40k pause first.
set -euo pipefail
cd "$(dirname "$0")/.."
while pgrep -f arith_wide_rerun512.sh >/dev/null; do sleep 60; done
n40=$(ls checkpoints/toy/wide/*@40k.pt | wc -l)
[ "$n40" -eq 9 ] || { echo "only $n40 of 9 runs paused at 40k; stopping"; exit 1; }
echo "$(date +%H:%M) all 9 at 40k; resuming to 80k"
scripts/arith_wide.sh resume80
n80=$(ls checkpoints/toy/wide/*@80k.pt | wc -l)
[ "$n80" -eq 9 ] || { echo "only $n80 of 9 runs paused at 80k; stopping"; exit 1; }
echo "$(date +%H:%M) all 9 at 80k; resuming to 120k"
scripts/arith_wide.sh resume
echo "$(date +%H:%M) chain finished"
