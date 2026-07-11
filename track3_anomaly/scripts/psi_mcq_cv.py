#!/usr/bin/env python3
"""Fresh PSI MCQ CV harness: runs the current psi_mcq pipeline on a held-out
train sample, records GT letter, predicted letter, AND the raw generation
text (for error analysis) to a JSONL. Supports prompt-suffix variants via
--variant and sampling knobs via the TAR_* env vars inference.py reads.

Usage:
  python3 psi_mcq_cv.py --n 60 --seed 11 --out baseline.jsonl [--variant name]
"""
import argparse, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO_ROOT = "../data/psi_vqa/train/videos"

# Variant suffixes (None = use prompts.py's current psi_mcq suffix as-is)
VARIANTS = {}

VARIANTS["describe_first"] = (
    "\nThis question is about a PEDESTRIAN'S CROSSING INTENT, not a "
    "collision. Focus only on the pedestrian marked by the red bounding "
    "box at the start and track them through the whole clip.\n"
    "Step 1 -- BEFORE looking at the options, describe in 3-4 sentences "
    "what this pedestrian actually does: their position (sidewalk / curb "
    "/ road edge / already in the roadway or a lane / crosswalk), their "
    "motion (standing still / walking / running) and direction relative "
    "to the road (parallel along it, toward it, across it), which way "
    "their body faces, and any context (crosswalk, parked cars, other "
    "pedestrians with them, passing traffic).\n"
    "Step 2 -- now compare each option A-D against your description. "
    "Exactly ONE option describes THIS pedestrian; the others describe "
    "pedestrians from different videos. Eliminate every option "
    "containing a claim that is false for this video (wrong position, "
    "wrong motion, wrong direction, wrong activity), and say why. Note "
    "the question's direction: if it asks why the pedestrian might NOT "
    "intend to cross, the correct option lists true observations that "
    "argue AGAINST crossing; if it asks why they might INTEND TO CROSS, "
    "the correct option lists true observations that argue FOR crossing "
    "-- but in both cases every bullet of the correct option must be "
    "factually true in this video.\n"
    "Step 3 -- on a new line, write 'Final answer: X' where X is the "
    "single capital letter of the option whose bullets all match what "
    "you observed."
)


VARIANTS["box_aware"] = (
    "\nThis question is about a PEDESTRIAN'S CROSSING INTENT, not a "
    "collision. IMPORTANT: the red bounding box that marks the pedestrian "
    "is drawn only for about one second of the clip and may appear at ANY "
    "point (beginning, middle, or end), not necessarily at the start; it "
    "can also be small if the pedestrian is far away. First find the "
    "frame(s) where the red box appears, identify exactly which pedestrian "
    "it marks, then track THAT pedestrian through the whole clip. Each "
    "option lists observations; exactly ONE option describes what THIS "
    "pedestrian actually does -- the other options describe pedestrians "
    "from different videos. Establish the facts: (1) POSITION: on the "
    "sidewalk, at the curb/road edge, one step into the road, or already "
    "in the roadway / middle of the road / a traffic lane / a crosswalk; "
    "(2) MOTION: standing still, walking slowly or normally, or running "
    "-- and the movement direction relative to the road (parallel along "
    "it, toward it, or across it, and to the left or right of the frame); "
    "(3) BODY AND GAZE: which way the body faces relative to the road, "
    "and whether they look at the camera car or at oncoming traffic; "
    "(4) CONTEXT: crosswalk, parked cars, other pedestrians moving with "
    "them, passing traffic they might be waiting on. Then check EVERY "
    "bullet of each option A-D against these facts and eliminate any "
    "option with a bullet that is false for this video (wrong position, "
    "wrong motion, wrong direction, wrong activity). Pick the option "
    "whose bullets all match what you observed, even if it describes "
    "weak or indirect evidence; if more than one option survives, pick "
    "the one that most specifically and completely matches this "
    "pedestrian's actual position and movement. Then, on a new line, "
    "write 'Final answer: X' where X is the single capital letter of "
    "the correct option."
)


def box_hint(video_id, index):
    """One-sentence factual hint about when/where the red box is visible,
    computed offline by psi_box_detect.py (pure CV, no model)."""
    rel = video_id.split("PSI/", 1)[-1]
    info = index.get(rel)
    if not info or not info.get("found"):
        return ""
    hx = "left" if info["cx"] < 0.36 else ("right" if info["cx"] > 0.64 else "center")
    hy = "upper" if info["cy"] < 0.40 else ("lower" if info["cy"] > 0.62 else "middle")
    size = ""
    if info["h"] < 45:
        size = " The box is small -- the marked pedestrian is far away."
    return (
        f"\n(Annotation note: the red bounding box is actually visible only "
        f"from t={info['t0']}s to t={info['t1']}s in this clip, in the "
        f"{hy}-{hx} area of the frame.{size})"
    )


def resolve(video_id):
    return os.path.join(VIDEO_ROOT, video_id.split("PSI/", 1)[-1])


def gt_letter(answer):
    m = re.match(r"^([A-D])\)", answer.strip())
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--hint", action="store_true",
                    help="inject red-box timing/location hint from red_box_index.json")
    ap.add_argument("--task", default="psi_mcq",
                    help="task_type to run (psi_mcq, or mcq for the generic Track3 prompt)")
    ap.add_argument("--hint-index", default="../data/psi_vqa/red_box_index.json")
    args = ap.parse_args()

    import prompts
    if args.variant:
        prompts.TASK_CONFIG["psi_mcq"]["suffix"] = VARIANTS[args.variant]

    box_index = {}
    if args.hint:
        box_index = json.load(open(args.hint_index))

    from inference import QwenVLBackend

    d = json.load(open("../data/psi_vqa/train/mcq.json"))
    items = [it for it in d["items"]
             if os.path.exists(resolve(it["video_id"])) and gt_letter(it["answer"])]
    random.Random(args.seed).shuffle(items)
    items = items[: args.n]

    backend = QwenVLBackend(quant="bf16")

    # wrap _generate_once so we can log the raw reasoning text
    raw_holder = {}
    orig_gen = backend._generate_once
    def wrapped(inputs, mnt, do_sample):
        out = orig_gen(inputs, mnt, do_sample)
        raw_holder["text"] = out
        return out
    backend._generate_once = wrapped

    correct = 0
    n_done = 0
    with open(args.out, "w") as f:
        for i, it in enumerate(items, 1):
            vpath = resolve(it["video_id"])
            raw_holder["text"] = ""
            question = it["question"]
            if args.hint:
                question = question + box_hint(it["video_id"], box_index)
            try:
                pred = backend.answer(vpath, args.task, question)
            except Exception as e:  # noqa: BLE001
                print(f"ERROR {it['video_id']}: {e}", file=sys.stderr)
                continue
            gt = gt_letter(it["answer"])
            ok = pred.strip().upper() == gt
            correct += ok
            n_done += 1
            polarity = "NOT" if "might NOT" in it["question"] else "CROSS"
            f.write(json.dumps({
                "item_index": it["item_index"], "video_id": it["video_id"],
                "gt": gt, "pred": pred, "ok": ok, "polarity": polarity,
                "question": it["question"], "gt_answer": it["answer"],
                "raw": raw_holder["text"],
            }) + "\n")
            f.flush()
            print(f"[{i}/{len(items)}] acc so far {correct}/{n_done} = {correct/max(1,n_done):.3f}",
                  file=sys.stderr)
    print(f"FINAL acc: {correct}/{n_done} = {correct/max(1,n_done):.4f}")


if __name__ == "__main__":
    main()
