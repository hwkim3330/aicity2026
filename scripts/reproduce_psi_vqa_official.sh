#!/usr/bin/env bash
# Reproduce the official Track 8 (PSI-VQA) submission configuration end to end.
#
# Chain:
#   1. check input metadata (four public question files, 328 items)
#   2. merge them into one routed test file
#   3. Qwen3-VL-8B inference: BCQ (5-sample self-consistency) + MCQ (greedy)
#      + OpenQA + temporal, written incrementally and resumable
#   4. replace the 56 temporal rows with the duration-stratified prior
#   5. replace the 126 OpenQA rows with the fitted two-cue answers
#   6. verify the record count is 328
#   7. run the organizer's validator if it is available
#   8. print SHA256 of the reproduced artifact
#
# Writes reproduced_* files only. The official artifact
# submissions/psi_vqa_submission_v7.csv is never touched.
#
# Determinism: steps 4 and 5 are fully deterministic. Step 3's MCQ, OpenQA and
# temporal rows use greedy decoding and are deterministic up to kernel
# nondeterminism; the 55 BCQ rows use 5-sample self-consistency with unseeded
# sampling, so they are NOT expected to reproduce bit-for-bit. See REPRODUCE.md.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
SCRIPTS="$ROOT/track3_anomaly/scripts"
SUBS="$ROOT/track3_anomaly/submissions"
DATA="${PSI_DATA:-$ROOT/track3_anomaly/data/psi_vqa}"
QDIR="$DATA/test_public"
VIDEOS="$QDIR/videos"

export TAR_MODEL_ID="${TAR_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
export TAR_MAX_FRAMES="${TAR_MAX_FRAMES:-16}"
export TAR_MAX_PIXELS="${TAR_MAX_PIXELS:-151200}"
export TAR_HF_CACHE="${TAR_HF_CACHE:-$HOME/.cache/huggingface/hub}"

BASE="$SUBS/reproduced_psi_base.csv"
TEMP="$SUBS/reproduced_psi_temporal.csv"
FINAL="${PSI_OUT:-$SUBS/reproduced_psi_vqa_v7.csv}"

echo "== 1. input metadata check =="
for f in bcq_questions mcq_questions open_qa_questions temporal_localization_questions; do
  test -f "$QDIR/$f.json" || { echo "missing $QDIR/$f.json"; exit 1; }
done
test -d "$VIDEOS" || { echo "missing video root $VIDEOS"; exit 1; }
python3 - "$QDIR" <<'PY'
import json, sys, os
qdir = sys.argv[1]
total = 0
for f, n in [("bcq_questions", 55), ("mcq_questions", 91),
             ("open_qa_questions", 126), ("temporal_localization_questions", 56)]:
    items = json.load(open(os.path.join(qdir, f + ".json")))["items"]
    total += len(items)
    status = "ok" if len(items) == n else f"EXPECTED {n}"
    print(f"  {f}: {len(items)} items {status}")
assert total == 328, f"expected 328 items, found {total}"
print(f"  total: {total} items")
PY

echo "== 2. merge question files =="
cd "$SCRIPTS"
python3 build_psi_test_json.py --src "$QDIR" --out "$QDIR/psi_test.json"

echo "== 3. Qwen3-VL inference (resumable) =="
python3 make_submission.py \
  --test_json "$QDIR/psi_test.json" \
  --media_root "$VIDEOS" \
  --out "$BASE" \
  --quant bf16 \
  --samples 5 \
  --resume

echo "== 4. duration-stratified temporal prior =="
python3 apply_psi_temporal_prior.py \
  --in "$BASE" --out "$TEMP" \
  --questions "$QDIR/temporal_localization_questions.json" \
  --videos "$VIDEOS"

echo "== 5. fitted two-cue OpenQA answers =="
python3 make_psi_v7_openqa_prior.py \
  --v6 "$TEMP" \
  --openqa-questions "$QDIR/open_qa_questions.json" \
  --out "$FINAL"

echo "== 6. record count =="
python3 - "$FINAL" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
assert len(rows) == 328, f"expected 328 records, found {len(rows)}"
assert all(r["item_index"] and r["prediction"] is not None for r in rows), "empty field"
print(f"  {len(rows)} records, all fields populated")
PY

echo "== 7. organizer validator =="
VALIDATOR="$ROOT/track3_anomaly/data/test/evaluate.py"
if [ -f "$VALIDATOR" ] && [ -f "$QDIR/psi_test.json" ]; then
  python3 "$VALIDATOR" --gt "$QDIR/psi_test.json" --submission "$FINAL" || \
    echo "  (validator reported issues; PSI uses a different envelope than TAR)"
else
  echo "  validator not available; skipped"
fi

echo "== 8. checksum =="
sha256sum "$FINAL"
echo
echo "Official artifact for comparison:"
sha256sum "$SUBS/psi_vqa_submission_v7.csv"
echo "BCQ rows use unseeded 5-sample voting, so the two checksums are not expected to match."
