#!/usr/bin/env bash
# One-shot pipeline to run once a new fine-tuned checkpoint is ready:
# re-embed the gallery with it, retrieve, validate, and leave the result
# ready to submit -- so finishing training and getting a submittable
# answer.txt is a single command instead of remembering the right flags.
#
# Usage:
#   ./run_after_finetune.sh clip_finetuned_big answer_big.txt
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:?usage: run_after_finetune.sh <model_dir> <out_answer_file>}"
OUT="${2:?usage: run_after_finetune.sh <model_dir> <out_answer_file>}"
SAFE_NAME="$(echo "$MODEL" | tr '/' '_')"
EMBEDS="data/gallery_embeds__${SAFE_NAME}.pt"

echo "[1/3] embedding gallery with $MODEL -> $EMBEDS"
python3 embed_gallery.py --model "$MODEL" --out "$EMBEDS"

echo "[2/3] retrieving -> $OUT"
python3 retrieve.py --model "$MODEL" --gallery-embeds "$EMBEDS" --out "$OUT"

echo "[3/3] validating"
python3 validate_submission.py --answer "$OUT"

echo "done -- $OUT is ready to submit"
