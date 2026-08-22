#!/usr/bin/env bash
# Same items, same 16 frames, same seed: only the generation budget changes.
#
# The shipped cap is 320 tokens and 38 of 321 items (11.8%) hit it mid-sentence,
# never reach "Final answer:", and fall through to the literal "A" -- correct
# 26.3% of the time where chance is 25%. This is a cause, not a knob: the
# question is whether letting the model finish recovers those items, not which
# budget scores best.
set -euo pipefail
cd "$(dirname "$0")"
OUT=/tmp/claude-1000/-home-kim/c14f0b79-a240-4e70-bb21-4cfc5bc33161/scratchpad
for t in 640; do
  echo "===== TAR_MIN_NEW_TOKENS=$t"
  TAR_MAX_FRAMES=16 TAR_MIN_NEW_TOKENS=$t python3 psi_mcq_cv.py --n 321 --seed 11 \
    --out "$OUT/psi_mcq_t$t.jsonl" 2>&1 | grep -E '^FINAL'
done
