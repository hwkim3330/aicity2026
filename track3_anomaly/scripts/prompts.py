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
    "observable in the clip. Do not add disclaimers about being an AI."
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
        "suffix": (
            "\nFirst, describe in 2-3 sentences exactly what you see happening "
            "that is relevant to this question, paying close attention to the "
            "specific objects, colors, and actions mentioned. Then, on a new "
            "line, write 'Final answer: Yes' or 'Final answer: No'. Base your "
            "answer only on what is visible; do not assume an event happened "
            "just because the clip shows an anomaly, and do not assume it "
            "didn't happen just because it is fast or partially occluded."
        ),
        "max_new_tokens": 192,
        "final_answer": "yesno",
        "self_consistency": True,
    },
    "bcq_openended": {
        "suffix": (
            "\nStart your response with exactly 'Yes.' or 'No.' and then give "
            "2-3 sentences of visual evidence for that verdict: which road "
            "user(s) are involved, what they do, and what outcome is visible."
        ),
        "max_new_tokens": 160,
    },
    "mcq": {
        "suffix": (
            "\nFirst, briefly reason about which option best matches the clip: "
            "eliminate options that contradict what is visible. Then, on a new "
            "line, write 'Final answer: X' where X is the single capital letter "
            "of the correct option."
        ),
        "max_new_tokens": 224,
        "final_answer": "letter",
        "self_consistency": True,
    },
    "mcq_openended": {
        "suffix": (
            "\nRespond in the format 'X) <text of the chosen option>' -- the "
            "correct option letter, a closing parenthesis, and then restate the "
            "chosen option's text, followed by one sentence of visual evidence."
        ),
        "max_new_tokens": 128,
    },
    "open_qa": {
        "suffix": (
            "\nAnswer in 3-5 sentences describing the concrete visual events: "
            "name the road users involved (vehicle types/colors, pedestrians), "
            "what each does over the course of the clip, and the outcome."
        ),
        "max_new_tokens": 224,
    },
    "scene_description": {
        "suffix": (
            "\nDescribe the scene in 3-5 sentences covering: the type of road "
            "or intersection and its layout, lane markings and traffic "
            "controls, the surrounding environment (buildings, vegetation, "
            "weather, lighting, time of day), and the general traffic "
            "conditions visible in the clip."
        ),
        "max_new_tokens": 224,
    },
    "video_summarization": {
        "suffix": (
            "\nWrite a thorough narrative summary (4-7 sentences) of the whole "
            "clip in chronological order: the setting, the road users present, "
            "the normal traffic flow at the start, the anomalous event (what "
            "happens, who is involved, how it unfolds), and the aftermath "
            "visible by the end of the clip."
        ),
        "max_new_tokens": 320,
    },
    "causal_linkage": {
        "suffix": (
            "\nExplain the causal chain in 3-4 sentences: the initiating "
            "condition or action, the intermediate behavior that follows from "
            "it, and how that directly leads to the resulting event. Refer to "
            "the specific road users involved."
        ),
        "max_new_tokens": 192,
    },
    "temporal_description": {
        "suffix": (
            "\nDescribe what happens in that time interval in 2-4 sentences, "
            "in chronological order, naming the road users involved and their "
            "specific actions."
        ),
        "max_new_tokens": 176,
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
