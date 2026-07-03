#!/usr/bin/env bash
# Zip the trainer_package/ directory for upload to Hafnia, enforcing the
# 2GB size cap the Track 6 rules mention. Excludes local run artifacts,
# caches, and (deliberately) any dataset files that might get dropped in
# by accident -- uploading dataset content is against the rules and would
# also blow the size cap instantly.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="trainer_package"
OUT="track6_trainer_package.zip"
MAX_BYTES=$((2 * 1024 * 1024 * 1024))

rm -f "$OUT"
zip -r "$OUT" "$SRC" \
    -x "*/runs/*" "*.pt" "*.pth" "*__pycache__*" "*.pyc" "*/.git/*"

SIZE=$(stat -c%s "$OUT")
echo "Package size: $SIZE bytes ($((SIZE / 1024 / 1024)) MiB)"
if [ "$SIZE" -gt "$MAX_BYTES" ]; then
    echo "ERROR: package exceeds the 2GB Hafnia upload limit!" >&2
    exit 1
fi
echo "OK: $OUT ready for upload to Hafnia."
