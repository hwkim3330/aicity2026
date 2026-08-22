#!/usr/bin/env bash
# Wait for the dense-frame pair to finish, then run the token-budget arm.
# Detached from any harness timeout: the 32-frame arm alone needs ~2 hours.
# Waits on the output file rather than a process name, because a pgrep pattern
# broad enough to match the runner also matches this script.
set -u
cd "$(dirname "$0")"
OUT=/tmp/claude-1000/-home-kim/c14f0b79-a240-4e70-bb21-4cfc5bc33161/scratchpad
while [[ $(wc -l < "$OUT/psi_mcq_f32.jsonl" 2>/dev/null || echo 0) -lt 321 ]]; do
  sleep 120
done
echo "[$(date +%H:%M)] dense-frame done"
grep FINAL denseframe.log
./run_tokenbudget.sh
echo "[$(date +%H:%M)] token budget done"
