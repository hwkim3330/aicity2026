#!/usr/bin/env python3
"""A/B test the v3 open-ended prompt rewrites (Fable-audited GT style match)
against the old v2 prompts, using real BERTScore-F1 against train GT --
mirrors the Track2 caption-prompt validation methodology."""
import json, os, random, sys
sys.path.insert(0, ".")
from inference import QwenVLBackend

VIDEO_ROOT = "../data/videos"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEED = 5
CATEGORIES = ["causal_linkage", "scene_description", "temporal_description",
              "video_summarization", "open_qa"]

OLD_SUFFIXES = {
    "open_qa": "\nAnswer in 2-4 concise sentences describing concrete visual events.",
    "scene_description": (
        "\nDescribe the scene in 3-5 sentences covering: the type of road "
        "or intersection and its layout, lane markings and traffic "
        "controls, the surrounding environment (buildings, vegetation, "
        "weather, lighting, time of day), and the general traffic "
        "conditions visible in the clip."
    ),
    "video_summarization": (
        "\nWrite a thorough narrative summary (4-7 sentences) of the whole "
        "clip in chronological order: the setting, the road users present, "
        "the normal traffic flow at the start, the anomalous event (what "
        "happens, who is involved, how it unfolds), and the aftermath "
        "visible by the end of the clip."
    ),
    "causal_linkage": (
        "\nExplain the causal chain in 3-4 sentences: the initiating "
        "condition or action, the intermediate behavior that follows from "
        "it, and how that directly leads to the resulting event. Refer to "
        "the specific road users involved."
    ),
    "temporal_description": (
        "\nDescribe what happens in that time interval in 2-4 sentences, "
        "in chronological order, naming the road users involved and their "
        "specific actions."
    ),
}
OLD_MAX_TOKENS = {"open_qa": 160, "scene_description": 224, "video_summarization": 320,
                   "causal_linkage": 192, "temporal_description": 176}


def load_sample(cat, n):
    d = json.load(open(f"../data/train/{cat}.json"))
    items = [it for it in d["items"] if it.get("answer")
              and os.path.exists(f"{VIDEO_ROOT}/{it['video_id']}")]
    rng = random.Random(SEED)
    rng.shuffle(items)
    return items[:n]


def main():
    from bert_score import BERTScorer
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    backend = QwenVLBackend(quant="bf16")
    import prompts as prompts_module

    for cat in CATEGORIES:
        items = load_sample(cat, N)
        old_preds, new_preds, refs = [], [], []
        for it in items:
            vpath = f"{VIDEO_ROOT}/{it['video_id']}"
            refs.append(it["answer"])
            # new (v3, already active in prompts.py)
            new_preds.append(backend.answer(vpath, cat, it["question"]))
            # old (v2) -- monkey-patch suffix/max_tokens temporarily
            orig_cfg = dict(prompts_module.TASK_CONFIG[cat])
            prompts_module.TASK_CONFIG[cat]["suffix"] = OLD_SUFFIXES[cat]
            prompts_module.TASK_CONFIG[cat]["max_new_tokens"] = OLD_MAX_TOKENS[cat]
            old_preds.append(backend.answer(vpath, cat, it["question"]))
            prompts_module.TASK_CONFIG[cat].update(orig_cfg)

        _, _, f1_old = scorer.score(old_preds, refs)
        _, _, f1_new = scorer.score(new_preds, refs)
        print(f"[{cat}] OLD (v2) BERTScore-F1: {f1_old.mean().item():.4f}")
        print(f"[{cat}] NEW (v3) BERTScore-F1: {f1_new.mean().item():.4f}")


if __name__ == "__main__":
    main()
