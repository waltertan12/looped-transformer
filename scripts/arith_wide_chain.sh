#!/usr/bin/env bash
# H16 (D-015), Walter 24 Nov: after every run is paused at 40k, continue all 9 to 80k (pause),
# then all 9 to 120k. Waits for the width-512 reruns to reach their 40k pause first.
set -euo pipefail
cd "$(dirname "$0")/.."
# Wait until every run has its 40k snapshot and no 40k-stage training is still running.
until [ "$(ls checkpoints/toy/wide/*@40k.pt 2>/dev/null | wc -l)" -eq 9 ] && ! pgrep -f "pause_at 40000" >/dev/null; do
  pgrep -f "pause_at 40000" >/dev/null || { echo "no 40k run left but not all 9 snapshots; stopping"; exit 1; }
  sleep 60
done
# Score add-only / sub-only sets only while the GPU is free of training (they share memory).
echo "$(date +%H:%M) all 9 at 40k; scoring add/sub sets"; scripts/wide_ops.sh 40k
echo "$(date +%H:%M) resuming to 80k"
scripts/arith_wide.sh resume80
n80=$(ls checkpoints/toy/wide/*@80k.pt | wc -l)
[ "$n80" -eq 9 ] || { echo "only $n80 of 9 runs paused at 80k; stopping"; exit 1; }
echo "$(date +%H:%M) all 9 at 80k; scoring add/sub sets"; scripts/wide_ops.sh 80k
echo "$(date +%H:%M) resuming to 120k"
scripts/arith_wide.sh resume
scripts/wide_ops.sh 120k
echo "$(date +%H:%M) chain finished"
