#!/usr/bin/env bash
# Runs make_vqa_submission.py in chunks, restarting the process between each
# chunk so the OS fully reclaims memory -- works around an unresolved memory
# leak (RSS jumped from 2.4GB to 15GB after ~13 calls in testing; in the
# original un-chunked run it grew to 16-17GB within ~60 items and crashed
# throughput to ~0.03 items/s via swap thrashing).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT="../submissions/test_vqa.json"
TOTAL=19624
CHUNK=100

while true; do
    DONE=$(python3 -c "import json,os; print(len(json.load(open('$OUT'))) if os.path.exists('$OUT') else 0)")
    echo "=== $(date '+%T') progress: $DONE/$TOTAL ==="
    if [ "$DONE" -ge "$TOTAL" ]; then
        echo "=== all done ==="
        break
    fi
    python3 make_vqa_submission.py \
        --vqa_json /home/kim/aicity2026/shared/data/wts/vqa_test/WTS_VQA_PUBLIC_TEST.json \
        --video_roots /home/kim/aicity2026/shared/data/wts/WTS_DATASET_PUBLIC_TEST/videos/test/public /home/kim/aicity2026/shared/data/wts/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public \
        --out "$OUT" \
        --resume --chunk_size "$CHUNK" --write_every 20
done
