"""
Prompt templates / generation configs per task_type for the AI City 2026
Track 3 (TAR) baseline.

The `question` field in the dataset already inlines task-specific
instructions (e.g. "Answer with only Yes or No."), so for most tasks we
simply forward the question verbatim to the VLM with a short system
prompt. We add a light task-specific wrapper on top to nudge format
compliance (matching the regex extractors in test/evaluate.py) and to set
sane generation kwargs (max_new_tokens, temperature) per task.
"""

from dataclasses import dataclass


SYSTEM_PROMPT = (
    "You are an expert traffic-surveillance video analyst. You watch short "
    "CCTV/dashcam-style clips and answer questions about traffic events, "
    "including anomalies such as collisions, near-misses, and unusual "
    "vehicle/pedestrian behavior. Answer concisely and follow the exact "
    "output format requested in the question. Do not add disclaimers about "
    "being an AI."
)

# Per task_type: (instruction_suffix appended after the raw question,
#                 max_new_tokens, do_sample)
TASK_CONFIG = {
    "bcq": {
        "suffix": "\nRespond with exactly one word: Yes or No. Do not add punctuation, explanation, or extra text.",
        "max_new_tokens": 8,
    },
    "bcq_openended": {
        "suffix": "\nStart your response with exactly 'Yes.' or 'No.' followed by one short sentence of explanation.",
        "max_new_tokens": 96,
    },
    "mcq": {
        "suffix": "\nRespond with exactly one capital letter (A, B, C, or D) and nothing else.",
        "max_new_tokens": 4,
    },
    "mcq_openended": {
        "suffix": "\nStart your response with the correct option letter followed by ') ' and then one short sentence of explanation.",
        "max_new_tokens": 96,
    },
    "open_qa": {
        "suffix": "\nAnswer in 2-4 concise sentences describing concrete visual events.",
        "max_new_tokens": 160,
    },
    "scene_description": {
        "suffix": "\nDescribe the road layout, environment, and setting in 2-4 concise sentences.",
        "max_new_tokens": 160,
    },
    "video_summarization": {
        "suffix": "\nSummarize the key events and anomalies in 2-4 concise sentences.",
        "max_new_tokens": 180,
    },
    "causal_linkage": {
        "suffix": "\nExplain the causal relationship in 2-3 concise sentences.",
        "max_new_tokens": 140,
    },
    "temporal_description": {
        "suffix": "\nDescribe what happens in that interval in 1-3 concise sentences.",
        "max_new_tokens": 120,
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


def build_prompt(task_type: str, question: str) -> PromptSpec:
    cfg = TASK_CONFIG.get(task_type, DEFAULT_CONFIG)
    user = question.rstrip() + cfg["suffix"]
    return PromptSpec(system=SYSTEM_PROMPT, user=user, max_new_tokens=cfg["max_new_tokens"])
