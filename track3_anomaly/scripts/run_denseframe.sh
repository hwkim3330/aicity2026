#!/usr/bin/env bash
# Paired 16 vs 32 frames on all 321 labelled train MCQ items.
#
# Same seed, same shuffled order, same prompts, same checkpoint: only
# TAR_MAX_FRAMES differs, so every item is its own control. Full 321 rather
# than a sample because the last time this pipeline was judged on a 24-item
# paired win, the win did not survive contact with the leaderboard.
set -euo pipefail
cd "$(dirname "$0")"
OUT=/tmp/claude-1000/-home-kim/c14f0b79-a240-4e70-bb21-4cfc5bc33161/scratchpad
for f in 16 32; do
  echo "===== TAR_MAX_FRAMES=$f"
  TAR_MAX_FRAMES=$f python3 psi_mcq_cv.py --n 321 --seed 11 \
    --out "$OUT/psi_mcq_f$f.jsonl" 2>&1 | grep -E '^FINAL|^\[3[0-9][0-9]/'
done
