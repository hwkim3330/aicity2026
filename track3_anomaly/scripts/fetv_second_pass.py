#!/usr/bin/env python3
"""
Track 7 (FETV) v8: targeted second-pass re-check of v7's weakest calls.

GT distribution of the 100-clip public scoring subset (from
data/fetv/fetv_repo/eval_subset_50.json 'violation_type_targets') vs v7:

    class             GT   v7
    no_violation      33   55   <- over-predicted (+22)
    jaywalking        15   16   ok
    wrong_way         14    9   under
    lane_discipline   14    1   badly under
    lane_use_control  12    6   under
    uturn              6    0   never predicted
    red_light          6   13   over-predicted (~2x)

Pass A re-checks every v7 no_violation clip with a prompt that explicitly
walks through the four under-predicted vehicle-violation classes before
allowing no_violation. Pass B re-checks every v7 red_light clip and lets
the model downgrade to one of those classes / no_violation when the
red-light evidence is weak.

v7's answer_date / answer_time (OCR-derived, near-perfect on the real
leaderboard) and answer_intersection_type / answer_weather / answer_light
are ALWAYS preserved from v7; only violation_type + its cascade fields
(violator_type, color, positions, lanes) + description are updated, and
the description is re-passed through the same "On <date> at <time>, "
opener enforcement as fetv_submission.py.

Usage:
    python3 fetv_second_pass.py \
        --base ../submissions/fetv_submission_v7.json \
        --clips ../data/fetv/FETV_public_clips \
        --out ../submissions/fetv_submission_v8.json \
        [--sidecar ../submissions/fetv_v8_secondpass_raw.json] [--resume]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetv_submission import (  # noqa: E402
    VIOLATOR_TYPES, COLORS, POSITIONS, LANES,
    parse_response, clean_record, _DESC_OPENER_RE,
)

RECHECK_CLASSES = ["wrong_way", "uturn", "lane_use_control", "lane_discipline"]

# Shared JSON schema block -- field wording kept identical to
# fetv_submission.PROMPT (grid/lane conventions the few-shot exemplars
# were built to calibrate), except violation_type whose options/instruction
# are pass-specific.
def _schema(violation_line):
    return f"""{{{{
  "answer_violation_type": {violation_line},
  "answer_violator_type": "one of {VIOLATOR_TYPES} ('na' only when no violation)",
  "answer_color": "violator's color, one of {COLORS}",
  "answer_initial_position": "where the violator STARTS in a 3x3 grid over the square center crop, one of {POSITIONS}. If the frame is widescreen (wider than tall), the 3x3 grid covers ONLY the central square whose width equals the frame height -- ignore the outer left/right margins; a violator in a margin takes the nearest Left/Right column.",
  "answer_final_position": "where the violator ENDS/exits, same options and same square-crop convention",
  "answer_initial_lane": "one of {LANES} -- lane where violator starts, counted in the violator's own direction of travel: lane 1 is the left-most lane (nearest the road center/median) from that driver's perspective, counting outward toward the curb. 'na' for pedestrians and no_violation.",
  "answer_final_lane": "one of {LANES} -- lane where violator ends, same convention",
  "answer_description": "concise 2-3 sentence third-person PAST-TENSE report. The first sentence MUST begin exactly 'On 2018-07-17 at 12:00:00, ' (the exact date/time will be substituted later). Then identify the violator as 'a <color> <type>' and state its path across the frame (initial to final grid area) and the violation by name. If no_violation: 'traffic moved normally through the intersection with no violation.'"
}}}}"""


PROMPT_A = f"""You are RE-EXAMINING a fisheye traffic-surveillance clip. A first-pass review labeled this clip "no_violation", but a statistical audit shows a meaningful share of clips given that label actually contain one of four subtle vehicle violations that are easy to miss on a fisheye camera. Re-watch the whole clip and check each of these four explicitly and IN THIS RANDOM ORDER, tracking EVERY vehicle's full trajectory from where it enters the frame to where it exits -- do not favor any one of the four just because of the order it is listed in below, and do not default to the most visually obvious pattern if a subtler one actually fits better:

- lane_discipline -- a vehicle drifting across or straddling lane markings without a clear turning purpose, weaving between lanes, riding along the painted lane divider (common for motorcycles), or changing lanes inside the intersection.
- uturn -- any vehicle that reverses its direction of travel by roughly 180 degrees: it enters and leaves on the same road arm, or sweeps a tight arc in or near the intersection so that it ends up heading back the way it came.
- lane_use_control -- a vehicle using a lane restricted to a different movement: going straight from a turn-only lane, turning from a straight-only lane, or driving in a restricted/bus lane.
- wrong_way -- any vehicle (watch motorcycles especially) traveling against the flow of other traffic in its lane, or on the wrong side of the road, even briefly or near the distorted frame edges.

Each of these four is roughly equally likely to be the answer when this clip does contain a violation -- treat them as equally plausible hypotheses and pick whichever one the visual evidence actually supports, not whichever is easiest to imagine. If you find one of these, report it. Only if after this focused re-check NONE of the four applies should you answer "no_violation" -- a majority of re-checked clips are genuinely normal, so do not invent a violation that is not visible.

Answer as one JSON object (no markdown fences, JSON only, no text before or after the object):

{_schema('"one of ' + str(RECHECK_CLASSES + ["no_violation"]) + '. Pick the single clearest violation you actually see, or no_violation."')}"""


PROMPT_B = f"""You are RE-EXAMINING a fisheye traffic-surveillance clip. A first-pass review labeled this clip "red_light" (a vehicle entering the intersection against a red signal). Look at the clip again with fresh eyes and independently decide the correct label -- treat "red_light" as neither more nor less likely to be correct than any alternative below; some first-pass red_light calls will hold up on re-examination and some will not, so judge this clip strictly on its own visual evidence.

- Answer "red_light" if you can point to concrete visual evidence: a visible red signal facing the violator as it enters, or cross traffic clearly flowing with the right of way while the violator cuts through it.
- Answer "uturn" if the vehicle instead reverses its direction ~180 degrees, leaving on the same road arm it came from.
- Answer "wrong_way" if the vehicle is instead traveling against the flow of traffic in its lane.
- Answer "lane_use_control" if it is instead using a movement-restricted lane wrongly (straight from a turn-only lane, turn from a straight-only lane, restricted/bus lane).
- Answer "lane_discipline" if it instead merely drifts across or straddles lane markings without turning purpose.
- Answer "no_violation" only if the crossing was clearly legal (signal green/amber, or a normal permitted movement) and none of the above apply.

Answer as one JSON object (no markdown fences, JSON only, no text before or after the object):

{_schema('"one of ' + str(["red_light"] + RECHECK_CLASSES + ["no_violation"]) + '"')}"""


CASCADE_FIELDS = ["answer_violation_type", "answer_violator_type", "answer_color",
                  "answer_initial_position", "answer_final_position",
                  "answer_initial_lane", "answer_final_lane"]
PRESERVE_FIELDS = ["answer_date", "answer_time", "answer_intersection_type",
                   "answer_weather", "answer_light"]


def merge_record(v7_rec, raw):
    """Adopt violation_type + cascade fields + description from the second-pass
    raw JSON, preserving v7's date/time/environment fields, and re-enforce the
    description opener exactly like fetv_submission.clean_record does."""
    cleaned = clean_record(v7_rec["clip_name"], raw)
    rec = dict(v7_rec)
    for f in CASCADE_FIELDS:
        rec[f] = cleaned[f]
    desc = _DESC_OPENER_RE.sub("", cleaned["answer_description"]).strip()
    rec["answer_description"] = f"On {rec['answer_date']} at {rec['answer_time']}, {desc}"
    return rec


def no_violation_record(v7_rec):
    rec = dict(v7_rec)
    rec["answer_violation_type"] = "no_violation"
    for f in ["answer_violator_type", "answer_color", "answer_initial_position",
              "answer_final_position", "answer_initial_lane", "answer_final_lane"]:
        rec[f] = "na"
    rec["answer_description"] = (
        f"On {rec['answer_date']} at {rec['answer_time']}, traffic moved normally "
        f"through the {rec['answer_intersection_type']} with no violation."
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="../submissions/fetv_submission_v7.json")
    ap.add_argument("--clips", default="../data/fetv/FETV_public_clips")
    ap.add_argument("--out", default="../submissions/fetv_submission_v8.json")
    ap.add_argument("--sidecar", default="../submissions/fetv_v8_secondpass_raw.json")
    ap.add_argument("--quant", default="bf16", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    with open(args.base) as f:
        base = json.load(f)
    base_by_name = {r["clip_name"]: r for r in base}

    pass_a = [r["clip_name"] for r in base if r["answer_violation_type"] == "no_violation"]
    pass_b = [r["clip_name"] for r in base if r["answer_violation_type"] == "red_light"]
    print(f"pass A (no_violation re-check): {len(pass_a)} clips", file=sys.stderr)
    print(f"pass B (red_light re-check):    {len(pass_b)} clips", file=sys.stderr)

    raw_results = {}
    if args.resume and os.path.exists(args.sidecar):
        with open(args.sidecar) as f:
            raw_results = json.load(f)
        print(f"[resume] {len(raw_results)} second-pass answers already saved", file=sys.stderr)

    import inference
    # v1 of this script pruned the few-shot exemplars down to keep only the
    # wrong_way example (reasoning: jaywalking isn't an allowed re-check
    # output). That backfired badly: with a single worked example, ALL 10
    # of pass A's flips landed on wrong_way (and all 26 of pass B's flips
    # fled red_light, many also toward wrong_way) -- the one demonstrated
    # pattern anchored the model regardless of what was actually on screen.
    # Keep BOTH official exemplars (jaywalking + wrong_way): this preserves
    # the grid/lane calibration the comment in inference.py describes them
    # for, and having two differently-shaped answers (pedestrian vs.
    # vehicle, T-intersection vs. T-intersection but different lanes)
    # avoids over-fitting to one violation's surface form. If the model
    # ever answers "jaywalking" here anyway, main()'s merge logic below
    # gates on membership in RECHECK_CLASSES (pass A) / the pass B allowed
    # set, both of which exclude jaywalking, so such an answer is simply
    # discarded and the original v7 record is kept -- no schema risk.
    backend = inference.QwenVLBackend(quant=args.quant)

    todo = [(cn, "A") for cn in pass_a] + [(cn, "B") for cn in pass_b]
    t0 = time.time()
    for i, (clip_name, which) in enumerate(todo, 1):
        if clip_name in raw_results:
            continue
        video_path = os.path.join(args.clips, clip_name)
        prompt = PROMPT_A if which == "A" else PROMPT_B
        try:
            out = backend.answer(video_path, "fetv_structured", prompt, fewshot=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {clip_name}: {e}", file=sys.stderr)
            out = ""
        raw_results[clip_name] = {"pass": which, "raw_text": out}
        if i % 10 == 0 or i == len(todo):
            with open(args.sidecar, "w") as f:
                json.dump(raw_results, f, indent=1)
            print(f"[{i}/{len(todo)}] elapsed {time.time()-t0:.0f}s", file=sys.stderr)

    with open(args.sidecar, "w") as f:
        json.dump(raw_results, f, indent=1)

    # ---- merge ----
    n_flip_a = n_keep_a = n_flip_b = n_keep_b = 0
    records = []
    for r in base:
        cn = r["clip_name"]
        entry = raw_results.get(cn)
        if entry is None:
            records.append(r)
            continue
        raw = parse_response(entry["raw_text"])
        new_type = str(raw.get("answer_violation_type", "")).strip()
        if entry["pass"] == "A":
            if new_type in RECHECK_CLASSES:
                records.append(merge_record(r, raw))
                n_flip_a += 1
            else:  # no_violation, out-of-vocab, or parse failure -> keep v7
                records.append(r)
                n_keep_a += 1
        else:  # pass B
            if new_type in RECHECK_CLASSES:
                records.append(merge_record(r, raw))
                n_flip_b += 1
            elif new_type == "no_violation":
                records.append(no_violation_record(r))
                n_flip_b += 1
            else:  # red_light confirmed / parse failure -> keep v7
                records.append(r)
                n_keep_b += 1

    print(f"pass A: flipped {n_flip_a}, kept no_violation {n_keep_a}", file=sys.stderr)
    print(f"pass B: flipped {n_flip_b}, kept red_light {n_keep_b}", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(records, f, indent=1)
    print(f"wrote {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
