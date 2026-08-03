# Qualitative case studies

The submitted paper argues that violator mis-selection drives the FETV errors
and that identity loss drives the PSI errors, but shows no frame, no predicted
record, and no worked failure anywhere. These two cases supply that evidence —
including where it contradicts the paper.

| Case | Claim under test | Outcome |
|---|---|---|
| [`fetv_violator_misselection.json`](fetv_violator_misselection.json) | A wrong actor selection simultaneously corrupts violation type, actor type, color, position, and lane | **Supported.** In 51 of 200 clips the whole actor-centric block flips as one unit between two versions of the same system |
| [`psi_mcq_output_contract_failure.json`](psi_mcq_output_contract_failure.json) | PSI failures come from losing the marked pedestrian's identity | **Not supported by this case.** The model tracked the pedestrian and described the correct option, then emitted no parseable answer at all |

## Why the second case matters

It is a counterexample to our own diagnosis. On a held-out training item the
model's opening description matches ground-truth option D almost word for word,
then the routed prompt's elimination step mis-negates that same option and the
generation runs out of token budget before the required `Final answer:` line.
Identity was never lost; geometry was never needed. Between 15% and 25% of
generations on that sample produce no parseable letter under every prompt
program tried, and each is silently replaced with a fallback answer — so these
contract violations are scored as ordinary wrong answers and never surface.

The paper lists metric-compatible output contracts and strict parsing as a
system contribution. This case shows the contract failing quietly at a
non-trivial rate, which is a more actionable finding than the identity story it
displaces.

## No frames are committed

The FETV clips are distributed through the challenge portal and PSI-VQA
inherits the TASI Benchmark Data Sharing Agreement from PSI 2.0. Redistribution
terms are not established for either, so this directory contains only JSON
descriptions and the renderer.

```bash
python3 scripts/render_case_studies.py --list

python3 scripts/render_case_studies.py \
  --case psi_mcq_output_contract_failure \
  --data-root track3_anomaly/data/psi_vqa

python3 scripts/render_case_studies.py \
  --case fetv_violator_misselection \
  --data-root /path/to/FETV_public_clips
```

Frames land in `docs/case_studies/rendered/`, which is gitignored. Requires
`ffmpeg`.

## Reproducing the population statistics

```bash
# FETV cascade coupling across the 200-clip test set
cd track3_anomaly/submissions && python3 - <<'PY'
import json
v11 = {r['clip_name']: r for r in json.load(open('fetv_submission_v11.json'))}
v8  = {r['clip_name']: r for r in json.load(open('fetv_submission_v8.json'))}
dep = ['answer_violator_type', 'answer_color', 'answer_initial_position',
       'answer_final_position', 'answer_initial_lane', 'answer_final_lane']
both = [k for k in v11 if k in v8]
vt  = sum(v11[k]['answer_violation_type'] != v8[k]['answer_violation_type'] for k in both)
any_ = sum(any(v11[k][f] != v8[k][f] for f in dep) for k in both)
all_ = sum(all(v11[k][f] != v8[k][f] for f in dep)
           and v11[k]['answer_violation_type'] != v8[k]['answer_violation_type']
           for k in both)
print(f'{len(both)} clips | violation_type differs {vt} | any dependent {any_} | full block {all_}')
PY

# PSI unparseable-answer rates per prompt program
cd track3_anomaly/analysis && python3 prompt_ablation_psi_mcq.py
```
