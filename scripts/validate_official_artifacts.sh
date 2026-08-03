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
  "PSI-VQA repository-side final candidate (psi_vqa_submission_v7.csv)"

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

psi = os.path.join(subs, "psi_vqa_submission_v7.csv")
if os.path.exists(psi):
    rows = list(csv.DictReader(open(psi)))
    n = len(rows)
    empty = [r["item_index"] for r in rows if not r["prediction"].strip()]
    print(f"  PSI:  {n} records", "ok" if n == 328 else "EXPECTED 328")
    print(f"        empty predictions: {len(empty)}", "ok" if not empty else "PROBLEM")
    bad += (n != 328) or bool(empty)
else:
    print("  PSI:  MISSING")
    bad += 1

sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && fail=1

echo
echo "== derived-step verification =="
if (cd "$ROOT/track3_anomaly/scripts" && python3 make_psi_v7_openqa_prior.py --verify >/dev/null 2>&1); then
  echo "  ok       v6 -> v7 OpenQA reconstruction matches the shipped file"
else
  echo "  FAILED   v6 -> v7 OpenQA reconstruction"
  echo "           (needs data/psi_vqa/test_public/open_qa_questions.json)"
  fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "All official artifact checks passed."
else
  echo "Some checks failed — see above."
fi
exit "$fail"
