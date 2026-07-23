#!/usr/bin/env bash
set -euo pipefail

export TAR_MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
export TAR_MAX_FRAMES="16"
export TAR_MAX_PIXELS="151200"

cd "$(dirname "$0")/.."
python track3_anomaly/scripts/fetv_submission.py \
  --clips "${FETV_CLIPS:-/path/to/FETV_public_clips}" \
  --out track3_anomaly/submissions/reproduced_fetv_v11.json \
  --quant bf16
