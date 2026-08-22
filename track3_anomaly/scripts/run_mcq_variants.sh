#!/usr/bin/env bash
# Two prompt variants against the shipped psi_mcq, same 321 items and seed.
#
# This is the first task today with valid local validation: the harness measures
# the shipped prompt honestly (0.2804 over 321 labelled items), the 0.6044 on the
# board came from a configuration that predates this repo, and AI City reopened
# evaluation. So a prompt that beats 0.2804 here is a real candidate rather than
# a number that cannot transfer.
#
# Both variants put the letter FIRST. The measured failure is truncation, not
# confusion -- 11.8% of items run past the cap mid-elimination and never emit a
# letter, and doubling the budget to 640 tokens leaves that at 11.2%. Emitting
# the answer before the reasoning makes running long harmless.
set -u
cd "$(dirname "$0")"
OUT=/tmp/claude-1000/-home-kim/c14f0b79-a240-4e70-bb21-4cfc5bc33161/scratchpad
while pgrep -f 'psi_mcq_cv.py --n 321 --seed 11 --out .*t640' >/dev/null 2>&1; do sleep 120; done
for v in answer_first answer_first_grounded; do
  echo "===== variant $v"
  TAR_MAX_FRAMES=16 python3 psi_mcq_cv.py --n 321 --seed 11 --variant "$v" \
    --out "$OUT/psi_mcq_$v.jsonl" 2>&1 | grep -E '^FINAL'
done
echo "[$(date +%H:%M)] variants done"
