#!/usr/bin/env bash
# Reproduce the Track 3 (TAR) submission configuration.
#
# Chain:
#   1. check the released test annotations (960 items over 80 clips)
#   2. fetch the test clips with the organizer's downloader if absent
#   3. Qwen-VL inference over all 960 items, written incrementally, resumable
#   4. verify the record count is 960 and every item is covered
#   5. run the organizer's validator (format/coverage only; the test answers
#      are redacted, so no local accuracy is computable)
#   6. print SHA256 of the reproduced artifact
#
# IMPORTANT — unresolved provenance. The repository holds nine TAR candidates
# and the portal submission that produced the official 0.4256 was not recorded.
# Set TAR_REFERENCE to whichever candidate you are comparing against. See the
# "Known provenance gaps" section of REPRODUCE.md.
#
# Backbone. TAR_MODEL_ID defaults to the Qwen3-VL-8B configuration used by the
# later TAR candidates. The earliest candidate
# (submissions/submission_qwen25vl_4bit.csv) used Qwen2.5-VL-7B-Instruct at NF4
# 4-bit; reproduce that one with:
#   TAR_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct TAR_QUANT=4bit ./reproduce_tar_official.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
SCRIPTS="$ROOT/track3_anomaly/scripts"
SUBS="$ROOT/track3_anomaly/submissions"
DATA="$ROOT/track3_anomaly/data"
VIDEOS="${TAR_VIDEOS:-$DATA/videos}"

export TAR_MODEL_ID="${TAR_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
export TAR_MAX_FRAMES="${TAR_MAX_FRAMES:-16}"
export TAR_MAX_PIXELS="${TAR_MAX_PIXELS:-151200}"
export TAR_HF_CACHE="${TAR_HF_CACHE:-$HOME/.cache/huggingface/hub}"
QUANT="${TAR_QUANT:-bf16}"
SAMPLES="${TAR_SAMPLES:-5}"

OUT="${TAR_OUT:-$SUBS/reproduced_tar.csv}"

echo "== 1. annotation check =="
test -f "$DATA/test/test.json" || { echo "missing $DATA/test/test.json"; exit 1; }
python3 - "$DATA/test/test.json" <<'PY'
import json, sys, collections
doc = json.load(open(sys.argv[1]))
items = doc["items"]
by_task = collections.Counter(it["task_type"] for it in items)
videos = {it["video_id"] for it in items}
print(f"  {len(items)} items over {len(videos)} clips")
for k, v in sorted(by_task.items()):
    print(f"    {k}: {v}")
assert len(items) == 960, f"expected 960 items, found {len(items)}"
PY

echo "== 2. test clips =="
mkdir -p "$VIDEOS"
if [ -z "$(ls -A "$VIDEOS" 2>/dev/null)" ]; then
  if [ -f "$DATA/test/download_test_videos.py" ]; then
    echo "  running the organizer's downloader (yt-dlp; some clips may be region"
    echo "  or age restricted and will fall back to type-correct placeholders)"
    (cd "$DATA/test" && python3 download_test_videos.py)
  else
    echo "  no downloader found; place the TAR test clips under $VIDEOS"
    exit 1
  fi
else
  echo "  using existing clips in $VIDEOS"
fi

echo "== 3. inference (model=$TAR_MODEL_ID quant=$QUANT samples=$SAMPLES) =="
cd "$SCRIPTS"
python3 make_submission.py \
  --test_json "$DATA/test/test.json" \
  --media_root "$VIDEOS" \
  --out "$OUT" \
  --quant "$QUANT" \
  --samples "$SAMPLES" \
  --resume

echo "== 4. coverage check =="
python3 - "$OUT" "$DATA/test/test.json" <<'PY'
import csv, json, sys
rows = list(csv.DictReader(open(sys.argv[1])))
want = {it["item_index"] for it in json.load(open(sys.argv[2]))["items"]}
have = {r["item_index"] for r in rows}
assert len(rows) == 960, f"expected 960 records, found {len(rows)}"
missing = want - have
assert not missing, f"{len(missing)} items missing, e.g. {sorted(missing)[:3]}"
print(f"  {len(rows)} records, all {len(want)} test items covered")
PY

echo "== 5. organizer validator =="
(cd "$DATA" && python3 test/evaluate.py --gt test/test.json --submission "$OUT") || \
  echo "  (validator reported issues — inspect before uploading)"

echo "== 6. checksum =="
sha256sum "$OUT"
if [ -n "${TAR_REFERENCE:-}" ]; then
  echo "Reference candidate:"
  sha256sum "$TAR_REFERENCE"
fi
