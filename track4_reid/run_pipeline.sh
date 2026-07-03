#!/usr/bin/env bash
# Track 4 end-to-end baseline: embed gallery once, then retrieve for all queries.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f data/gallery_embeds.pt ]; then
    echo "embedding gallery (one-time, ~36.8k images)..."
    python3 embed_gallery.py
fi

python3 retrieve.py --out answer.txt
echo "done -> answer.txt"
