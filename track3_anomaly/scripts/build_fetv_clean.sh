#!/usr/bin/env bash
# End-to-end FETV submission from the public clips, with nothing outside this repo.
#
# This is the pipeline the July submission should have been and was not: three
# committed steps, no environment overrides, no uncommitted script, no lookup
# table, no manual edit. The configuration is the one the camera-ready describes
# -- Qwen3-VL-8B-Instruct in bf16, 16 frames max / 4 min, 360x420 pixels per
# frame, greedy decoding -- and all of it comes from code defaults, so the
# command below *is* the configuration.
#
# It does not reconstruct the 2026-07-11 artifact and is not offered as one; the
# first pass that produced that artifact ran from an uncommitted working tree
# that no longer exists (see REPRODUCE.md). What this does is produce a
# deterministic output a third party can regenerate byte-for-byte.
set -euo pipefail
cd "$(dirname "$0")/.."
CLIPS="${FETV_CLIPS:-data/fetv/FETV_public_clips}"
STAMP="${STAMP:-$(date +%Y-%m-%d)}"

python3 scripts/fetv_submission.py --clips "$CLIPS" --quant bf16 \
        --out "submissions/clean_pass1_${STAMP}.json"

python3 scripts/fetv_second_pass.py --base "submissions/clean_pass1_${STAMP}.json" \
        --clips "$CLIPS" --quant bf16 \
        --out "submissions/clean_pass2_${STAMP}.json" \
        --sidecar "submissions/clean_pass2_raw_${STAMP}.json"

python3 - "$STAMP" <<'PY'
import json, sys
sys.path.insert(0, 'scripts')
from make_fetv_v11_descriptions import build
stamp = sys.argv[1]
rows = json.load(open(f'submissions/clean_pass2_{stamp}.json'))
for r in rows:                       # the template covers violation rows only,
    if str(r['answer_violation_type']).lower() != 'no_violation':
        r['answer_description'] = build(r)
json.dump(rows, open(f'submissions/clean_fetv_{stamp}.json', 'w'),
          ensure_ascii=False, indent=1)
print(f'wrote submissions/clean_fetv_{stamp}.json')
PY
sha256sum "submissions/clean_fetv_${STAMP}.json"
