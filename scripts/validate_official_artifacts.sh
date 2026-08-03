#!/usr/bin/env bash
# Verify every official artifact in this repository: record counts, structural
# shape, and SHA256 against the values recorded in ARTIFACTS.md / REPRODUCE.md.
#
# Runs offline. No model, dataset, or GPU required.
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
SUBS="$ROOT/track3_anomaly/submissions"
fail=0

check_sha () {
  local path="$1" want="$2" label="$3"
  if [ ! -f "$path" ]; then
    echo "  MISSING  $label ($path)"
    fail=1
    return
  fi
  local got
  got="$(sha256sum "$path" | cut -d' ' -f1)"
  if [ "$got" = "$want" ]; then
    echo "  ok       $label"
  else
    echo "  MISMATCH $label"
    echo "           expected $want"
    echo "           actual   $got"
    fail=1
  fi
}

echo "== SHA256 =="
check_sha "$SUBS/fetv_submission_v11.json" \
  "39abdb0a8cca7a7fa18dbd31374ee353e032977df9928d54734a53e9ec43e835" \
  "FETV official (fetv_submission_v11.json)"
check_sha "$SUBS/psi_vqa_submission_v7.csv" \
  "a3829a36f591907bb8838098b1cc61feb907fec1cc6215f6098094aafaafb110" \
  "PSI-VQA official (psi_vqa_submission_v7.csv)"
check_sha "$SUBS/submission_qwen3vl8b_v9.csv" \
  "243a5e8b67310428096cfc760ddeedaf5bc9d280729ad73f4c940eb3da759f6f" \
  "TAR official (submission_qwen3vl8b_v9.csv)"

echo
echo "== structure =="
python3 - "$SUBS" <<'PY'
import csv, json, os, sys
subs = sys.argv[1]
bad = 0

fetv = os.path.join(subs, "fetv_submission_v11.json")
if os.path.exists(fetv):
    doc = json.load(open(fetv))
    recs = doc if isinstance(doc, list) else doc.get("items", doc)
    n = len(recs)
    print(f"  FETV: {n} records", "ok" if n == 200 else "EXPECTED 200")
    bad += n != 200
    sample = recs[0] if isinstance(recs, list) else next(iter(recs.values()))
    print(f"        fields on first record: {len(sample)}")
else:
    print("  FETV: MISSING")
    bad += 1

for label, fname, want in (("PSI", "psi_vqa_submission_v7.csv", 328),
                           ("TAR", "submission_qwen3vl8b_v9.csv", 960)):
    path = os.path.join(subs, fname)
    if os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        n = len(rows)
        empty = [r["item_index"] for r in rows if not r["prediction"].strip()]
        print(f"  {label}:  {n} records", "ok" if n == want else f"EXPECTED {want}")
        print(f"        empty predictions: {len(empty)}", "ok" if not empty else "PROBLEM")
        bad += (n != want) or bool(empty)
    else:
        print(f"  {label}:  MISSING")
        bad += 1

sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && fail=1

echo
echo "== derived-step verification =="
OPENQA_Q="$ROOT/track3_anomaly/data/psi_vqa/test_public/open_qa_questions.json"
if [ ! -f "$OPENQA_Q" ]; then
  echo "  skipped  v6 -> v7 OpenQA reconstruction (dataset not present)"
  echo "           fetch it with:"
  echo "             hf download ise-ice-lab/PSI_VQA --repo-type dataset \\"
  echo "               --local-dir track3_anomaly/data/psi_vqa --include 'test_public/*'"
elif (cd "$ROOT/track3_anomaly/scripts" && python3 make_psi_v7_openqa_prior.py --verify >/dev/null 2>&1); then
  echo "  ok       v6 -> v7 OpenQA reconstruction matches the shipped file"
else
  echo "  FAILED   v6 -> v7 OpenQA reconstruction"
  fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "All official artifact checks passed."
else
  echo "Some checks failed — see above."
fi
exit "$fail"
