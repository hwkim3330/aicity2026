"""
Prompt templates / generation configs per task_type for the AI City 2026
Track 3 (TAR) baseline.

v2 notes (informed by reading data/test/evaluate.py and our first
leaderboard submission at mean=0.34):

- bcq/mcq are scored by exact match after regex extraction; our first run
  answered directly with max_new_tokens<=8 and showed a strong "No" bias
  (83% No, 0.4625 acc). v2 has the model reason about the clip first and
  finish with a "Final answer:" line. inference.py extracts just the final
  Yes/No/letter and writes ONLY that to the CSV, so the reasoning text can
  never confuse the organizers' extraction regex.
- All open-ended tasks are scored with BERTScore-F1 (rescale_with_baseline),
  which rewards matching the reference answers' content coverage and
  narrative style, and punishes overly terse text hard (our summarization
  scored 0.0993 with "2-4 concise sentences"). v2 prompts ask for fuller,
  structured narrative coverage and raise max_new_tokens.
"""

from dataclasses import dataclass


SYSTEM_PROMPT = (
    "You are an expert traffic-surveillance video analyst. You watch short "
    "CCTV/dashcam-style clips and answer questions about traffic events, "
    "including anomalies such as collisions, near-misses, and unusual "
    "vehicle/pedestrian behavior. Ground every statement in what is visually "
    "observable in the clip. Do not add disclaimers about being an AI. "
    "Write in fluent third-person present tense. Refer to vehicles as "
    "'<color> <type>' (e.g. 'a white sedan', 'a black SUV'). Describe "
    "movement relative to the video frame (e.g. 'toward the bottom of the "
    "frame', 'toward the camera'), not relative to compass directions."
)

# Per task_type:
#   suffix          -- appended after the raw question
#   max_new_tokens  -- generation budget
#   final_answer    -- "yesno" | "letter" | None; when set, inference.py
#                      extracts the token after "Final answer:" and submits
#                      only that token (plus optional restated text).
#   self_consistency-- eligible for multi-sample majority vote in
#                      make_submission.py --samples N
TASK_CONFIG = {
    "bcq": {
        # v3: our v7 submission showed a 73% No / 27% Yes prediction skew
        # against a real train GT split of ~50/50 -- the old symmetric
        # caution ("do not assume X just because... do not assume not-X
        # just because...") was net-pushing toward No given these clips
        # almost always contain a real collision/anomaly and the probed
        # detail is often a fast, easy-to-miss post-impact behavior.
        "suffix": (
            "\nThese clips almost always contain a collision or anomaly; the "
            "question probes whether a SPECIFIC detail of it is true (which "
            "vehicles are involved, what maneuver occurred, what happens "
            "after impact). Track every vehicle/pedestrian mentioned in the "
            "question through the WHOLE clip, including after any impact. "
            "A brief contact lasting only one or two frames still counts as "
            "an event -- do not default to No just because it is fast. "
            "First, describe in 2-3 sentences exactly what you see happening "
            "that is relevant to this question, paying close attention to "
            "the specific objects, colors, and actions mentioned; if the "
            "answer is No, state what actually happened instead (e.g. "
            "'the collision involved a white SUV, not a second white van'). "
            "Then, on a new line, write 'Final answer: Yes' or "
            "'Final answer: No'."
        ),
        "max_new_tokens": 192,
        "final_answer": "yesno",
        "self_consistency": True,
    },
    "bcq_openended": {
        # v2's "2-3 sentences of evidence" dropped F1 0.470->0.359: the GT
        # style is short. One sentence maximum.
        "suffix": (
            "\nStart your response with exactly 'Yes.' or 'No.' followed by "
            "one short sentence of explanation."
        ),
        "max_new_tokens": 64,
    },
    "mcq": {
        # v3: test-set MCQ options are almost all forensic attribute-flips
        # of the same small fact set (entry side/turn direction, signal
        # color, impact type, post-impact outcome), not near-miss temporal
        # permutations -- so a forensic checklist beats generic
        # "eliminate what contradicts" reasoning. Also self_consistency
        # dropped to False: 5-sample majority vote was found to actively
        # hurt MCQ specifically (same failure mode confirmed independently
        # on Track8/PSI-VQA) -- correlated derailments from shared visual
        # input outvote a correct greedy answer more often than they
        # correct one.
        "suffix": (
            "\nFirst establish the facts: (1) each involved vehicle's color "
            "and type, (2) which side of the frame each enters from and "
            "whether it goes straight or turns, (3) the traffic-signal "
            "state if visible, (4) the type and moment of impact (T-bone/ "
            "head-on/rear-end/side-swipe) and which vehicle strikes which, "
            "(5) what happens after impact (stalls, flips, pushed which "
            "way, drives off). Then check each option A-D against these "
            "facts and eliminate any that get a fact wrong. Then, on a new "
            "line, write 'Final answer: X' where X is the single capital "
            "letter of the correct option."
        ),
        "max_new_tokens": 320,
        "final_answer": "letter",
        "self_consistency": False,
    },
    "psi_bcq": {
        # Track 8 (PSI-VQA OOD): pedestrian crossing-INTENT binary question
        # ("Does this pedestrian intend to cross in front of our car?").
        # These 'clear'-split clips contain NO collision -- the generic
        # Track3 "bcq" suffix ("these clips almost always contain a
        # collision or anomaly... track through impact") was a cross-track
        # prompt mismatch. Train GT split is 141 Yes / 104 No (58/42).
        # self_consistency stays True: 5-sample voting was confirmed to
        # HELP BCQ on the real PSI leaderboard (it hurt only MCQ).
        "suffix": (
            "\nThis question is about a PEDESTRIAN'S CROSSING INTENT -- "
            "whether the pedestrian marked by the red bounding box at the "
            "start intends to cross the road in front of the camera car. "
            "There is no collision in these clips; judge intent from body "
            "language and position. Track that specific pedestrian through "
            "the whole clip. Evidence FOR crossing intent (Yes): already in "
            "the roadway or stepping off the curb; walking toward or into "
            "the road or across it; body or face oriented toward/across the "
            "road; standing at the curb or crosswalk edge watching traffic "
            "as if waiting for a gap; crossing together with a group. "
            "Evidence AGAINST (No): walking parallel to the road along the "
            "sidewalk; moving away from the road; engaged in another "
            "activity (talking, loading a car, waiting at a bus stop) with "
            "body facing away from the roadway; far from the camera car's "
            "path with no movement toward it. A pedestrian waiting at the "
            "curb for cars to pass can still intend to cross -- do not "
            "answer No merely because they have not yet stepped into the "
            "road. First describe in 2-3 sentences the pedestrian's "
            "position (sidewalk / curb / in the roadway), their movement "
            "direction and speed, and their body/gaze orientation relative "
            "to the road. Then, on a new line, write 'Final answer: Yes' "
            "or 'Final answer: No'."
        ),
        "max_new_tokens": 192,
        "final_answer": "yesno",
        "self_consistency": True,
    },
    "psi_mcq": {
        # Track 8 (PSI-VQA OOD): 'ambiguous'-split clips. Question is "why
        # the pedestrian might INTEND TO CROSS" (or "might NOT intend to
        # cross"); each option A-D is a bullet list of observations that
        # annotators cited. Verified on train/mcq.json (321 items): exactly
        # one option's bullets factually describe THIS video's pedestrian --
        # distractors are cue-sets lifted from other videos whose bullets
        # are false here (wrong position/motion/direction). So the task is
        # visual fact-matching per bullet, NOT picking the strongest-
        # sounding intent cue (the correct 'INTEND TO CROSS' answer can be
        # "standing still on the sidewalk... waiting to cross" if that is
        # what the video shows). Track3's forensic vehicle-collision
        # checklist was a cross-track prompt mismatch here.
        # self_consistency=False: 5-sample voting confirmed to hurt MCQ on
        # the real leaderboard.
        "suffix": (
            "\nThis question is about a PEDESTRIAN'S CROSSING INTENT, not a "
            "collision. Focus only on the pedestrian marked by the red "
            "bounding box at the start and track them through the whole "
            "clip. Each option lists observations; exactly ONE option "
            "describes what THIS pedestrian actually does -- the other "
            "options describe pedestrians from different videos. First "
            "establish the facts: (1) POSITION: on the sidewalk, at the "
            "curb/road edge, one step into the road, or already in the "
            "roadway / middle of the road / a traffic lane / a crosswalk; "
            "(2) MOTION: standing still, walking slowly or normally, or "
            "running -- and the movement direction relative to the road "
            "(parallel along it, toward it, or across it, and to the left "
            "or right of the frame); (3) BODY AND GAZE: which way the body "
            "faces relative to the road, and whether they look at the "
            "camera car or at oncoming traffic; (4) CONTEXT: crosswalk, "
            "parked cars, other pedestrians moving with them, passing "
            "traffic they might be waiting on. Then check EVERY bullet of "
            "each option A-D against these facts and eliminate any option "
            "with a bullet that is false for this video (wrong position, "
            "wrong motion, wrong direction, wrong activity). Pick the "
            "option whose bullets all match what you observed, even if it "
            "describes weak or indirect evidence; if more than one option "
            "survives, pick the one that most specifically and completely "
            "matches this pedestrian's actual position and movement. Then, "
            "on a new line, write 'Final answer: X' where X is the single "
            "capital letter of the correct option."
        ),
        "max_new_tokens": 320,
        "final_answer": "letter",
        "self_consistency": False,
    },
    "mcq_openended": {
        # leaders score 0.958 F1 here, which is only possible if the GT is
        # essentially "X) <option text>" verbatim -- restate exactly, no
        # added explanation (v2's added evidence sentence dropped us
        # 0.631->0.548).
        "suffix": (
            "\nRespond with exactly the correct option letter, a closing "
            "parenthesis, and the chosen option's text verbatim -- "
            "'X) <option text>' -- and nothing else."
        ),
        "max_new_tokens": 64,
    },
    "open_qa": {
        # v3: rewritten to match real GT style (found via statistical audit
        # of data/train/open_qa.json, 3670 items) -- GT overwhelmingly
        # echoes the question's own subject as the answer's opening subject
        # ("The root cause...", "The traffic flow...") since that's the
        # single strongest lexical-overlap lever for BERTScore-F1 here.
        "suffix": (
            "\nAnswer in 2-3 sentences (about 50-80 words). Begin by echoing "
            "the subject of the question as your sentence subject (for "
            "'What is the root cause...' start 'The root cause of the "
            "incident is...'; for 'Describe the overall traffic flow...' "
            "start 'The traffic flow is...'; for 'Describe the behavior of "
            "the red truck...' start 'The red truck...'). State concrete "
            "observable facts, then a brief characterization (steady, "
            "free-flowing, congested, orderly, or anomalous). Answer only "
            "what is asked; do not add unrelated details."
        ),
        "max_new_tokens": 160,
    },
    "scene_description": {
        # v3: GT (audited, 3670 items) opens "This is a/an ..." or "This "
        # video is captured from ..." naming the CAMERA PERSPECTIVE first
        # (91% of answers) -- something v2 never asked for -- then covers
        # layout/markings/controls/surroundings in that order, 5-6
        # sentences (mode), not 3-5.
        "suffix": (
            "\nDescribe the scene in 5-6 sentences (about 100-125 words), in "
            "this order. (1) Open with 'This is a ...' or 'This video is "
            "captured from ...' stating the camera perspective (elevated / "
            "CCTV / traffic camera / dashcam / vehicle-mounted), time of "
            "day, the road type (multi-lane divided highway, four-way "
            "intersection, T-junction, narrow street), and the weather. "
            "(2-3) The road layout: number of lanes, medians or dividers, "
            "and the direction of traffic flow on each side relative to the "
            "frame; lane markings (dashed white lines, double yellow lines, "
            "painted arrows) and whether the road surface is dry or wet. "
            "(4) Traffic controls and infrastructure: signals, signs, "
            "crosswalks, sidewalks, bus stops -- explicitly note absence if "
            "none are visible (e.g. 'No traffic signals or signboards are "
            "visible'). (5) Surroundings: buildings, trees, streetlights, "
            "plus any on-screen text or timestamp overlays. (6) Optionally "
            "the road users present, naming vehicle colors and types."
        ),
        "max_new_tokens": 256,
    },
    "video_summarization": {
        # v3: GT (audited, 3670 items) opens "The video shows..." (98%),
        # cites timestamps mid-narrative (77%), and closes with "The root "
        # cause of the incident is..." for anomaly clips -- but ~half the
        # dataset is normal traffic (so-tad-heavy) with an explicit
        # normality closer instead ("All vehicles maintain their lanes...").
        # v2 presupposed an anomaly always exists, inviting hallucination
        # on the normal half.
        "suffix": (
            "\nWrite a 4-6 sentence summary (about 90-140 words). The first "
            "sentence must begin 'The video shows' and characterize the "
            "overall scene and traffic in one line. Then narrate the key "
            "events in chronological order, naming vehicles by color and "
            "type and citing timestamps like 'At 00:15' for key moments. If "
            "an anomaly occurs (collision, pedestrian strike, stalled "
            "vehicle, sudden stop), describe how it unfolds and its "
            "aftermath, and end with a sentence beginning 'The root cause "
            "of the incident is'. If traffic is normal, do NOT invent an "
            "anomaly; instead end with an explicit statement such as 'All "
            "vehicles maintain their lanes and steady speeds without any "
            "collisions or anomalies.'"
        ),
        "max_new_tokens": 320,
    },
    "causal_linkage": {
        # v3: GT (audited, 3670 items) is a rigid 3-sentence template: (1)
        # "At <T1>, ..., while by <T2>, ..." verbatim-echoing both question
        # timestamps, (2) "The relationship is <classification>: <why>",
        # (3) elaboration. v2 presupposed causality, forcing the model to
        # invent causal chains on the ~majority of normal-traffic clips
        # where GT explicitly says "one of continuity and stability".
        "suffix": (
            "\nAnswer in exactly 3 sentences (about 100-115 words) following "
            "this template. Sentence 1: 'At <first timestamp>, <what is "
            "happening>, while by <second timestamp>, <what is happening>.' "
            "Copy both timestamps verbatim from the question. Sentence 2: "
            "begin with 'The relationship is' and classify it -- for steady "
            "unchanged traffic: 'one of continuity and stability'; for "
            "ordinary progression of flow or a maneuver completing: "
            "'sequential and operational'; for an incident and its result: "
            "'one of direct cause and effect'. Then explain the mechanism "
            "linking the two moments. Sentence 3: elaborate the consequence "
            "-- what the first moment's state or action led to at the "
            "second moment, or confirm the state simply persisted. Many "
            "clips show normal traffic: do NOT invent an accident or "
            "causal chain if the two moments just show continuing steady "
            "flow."
        ),
        "max_new_tokens": 224,
    },
    "temporal_description": {
        # v3: GT (audited, 3670 items) is exactly 2 sentences in 69% of
        # cases, opens with the primary actor as subject ("A white "
        # sedan...", 90% of answers) rather than restating the interval,
        # and its second sentence is an interpretive/contextual line
        # ("This movement represents...") in ~half the answers -- v2 asked
        # for neither the subject-first opener nor the interpretive coda.
        "suffix": (
            "\nAnswer in exactly 2 sentences (about 55-75 words). Sentence "
            "1: begin directly with the main road user as the subject, "
            "giving its color and vehicle type (e.g. 'A white sedan...', "
            "'A black SUV...'), and describe its concrete action during "
            "that interval. Do NOT restate the timestamps and do NOT begin "
            "with 'Between'. Sentence 2: interpret the movement in traffic "
            "terms (e.g. 'This movement represents a standard flow of "
            "traffic in a light-volume corridor'). If an accident or "
            "anomaly occurs in the interval, sentence 2 instead states its "
            "immediate cause or consequence."
        ),
        "max_new_tokens": 128,
    },
    "fetv_structured": {
        # Track 7 (FETV OOD leaderboard): one combined structured-JSON call
        # per clip, prompt built in fetv_submission.py.
        "suffix": "",
        "max_new_tokens": 512,
    },
    "temporal_localization": {
        "suffix": (
            "\nRespond with ONLY a JSON object in the form "
            '{"start": "MM:SS", "end": "MM:SS"} and nothing else '
            "(no markdown fences, no extra text)."
        ),
        "max_new_tokens": 40,
    },
}

DEFAULT_CONFIG = {"suffix": "", "max_new_tokens": 128}


@dataclass
class PromptSpec:
    system: str
    user: str
    max_new_tokens: int
    final_answer: str | None = None
    self_consistency: bool = False


def build_prompt(task_type: str, question: str) -> PromptSpec:
    cfg = TASK_CONFIG.get(task_type, DEFAULT_CONFIG)
    user = question.rstrip() + cfg["suffix"]
    return PromptSpec(
        system=SYSTEM_PROMPT,
        user=user,
        max_new_tokens=cfg["max_new_tokens"],
        final_answer=cfg.get("final_answer"),
        self_consistency=cfg.get("self_consistency", False),
    )
