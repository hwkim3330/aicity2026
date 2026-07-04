#!/usr/bin/env bash
# Run the full Track 1 pipeline over every test-split scene with the
# fine-tuned detector, then concatenate per-scene outputs into one
# submission file (track1_test.txt).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SCENES=$(ls data/MTMC_Tracking_2026/test/)
OUT_FINAL="track1_test.txt"
mkdir -p test_outputs

for SCENE in $SCENES; do
    echo "=========================================="
    echo "=== scene $SCENE ($(date '+%T')) ==="
    echo "=========================================="
    ./run_pipeline.sh --scene "$SCENE" --split test --out "test_outputs/${SCENE}.txt" \
        || { echo "!!! $SCENE FAILED, continuing with next scene"; continue; }
done

cat test_outputs/Warehouse_*.txt > "$OUT_FINAL"
echo "=== combined $(wc -l < "$OUT_FINAL") lines -> $OUT_FINAL ==="
ls -la "$OUT_FINAL"
echo "ALL SCENES DONE"
