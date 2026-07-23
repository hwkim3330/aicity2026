#!/usr/bin/env python3
"""
Qwen-VL inference backend for AI City Challenge 2026. The official FETV v11
run used Qwen/Qwen3-VL-8B-Instruct in bf16; TAR experiments may select the
Qwen2.5-VL 4-bit path through TAR_MODEL_ID/--quant.

Loads the model once (4-bit NF4 quantized via bitsandbytes, ~6-8GB VRAM)
and exposes `answer_one(video_path, task_type, question)` plus a CLI for
smoke-testing a single clip.

Usage:
    python inference.py --video /path/to/clip.mp4 \
        --task_type bcq \
        --question "Does a collision occur in the video?\nAnswer with only Yes or No."
"""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import build_prompt  # noqa: E402

MODEL_ID = os.environ.get("TAR_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
HF_CACHE = os.environ.get(
    "TAR_HF_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_cache"),
)

# Video sampling knobs. Kept conservative (short clips, <=20s typical) so a
# single 3090 handles it comfortably at 4-bit.
MAX_FRAMES = int(os.environ.get("TAR_MAX_FRAMES", 16))
MIN_FRAMES = 4
MAX_PIXELS_PER_FRAME = int(os.environ.get("TAR_MAX_PIXELS", 360 * 420))  # official FETV budget

# Few-shot examples pulled from data/train/{bcq,mcq}.json (real ground truth,
# not test data -- allowed per the rules, which only forbid training/
# tuning on the TEST distribution). Answer text is written in the same
# reason-then-"Final answer:" style the real prompts ask for, so the model
# sees a worked example of the exact output format alongside the content.
_VIDEO_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "videos")
FEWSHOT_EXAMPLES = {
    "bcq": [
        {
            "video_path": os.path.join(_VIDEO_ROOT, "Accident-Bench/land_space/long/videos/000001.mp4"),
            "question": "Does a white van collide with a white SUV in the video?\nAnswer with Yes or No.",
            "answer_text": (
                "A white van moves straight from the top of the frame toward the "
                "bottom while a white SUV approaches the intersection from the "
                "left. The two vehicles make contact in a T-bone collision in "
                "the middle of the intersection.\nFinal answer: Yes"
            ),
        },
        {
            "video_path": os.path.join(_VIDEO_ROOT, "Accident-Bench/land_space/long/videos/000001.mp4"),
            "question": "Is there a collision between a red SUV and a silver sedan?\nAnswer with Yes or No",
            "answer_text": (
                "A red SUV and another red SUV both move straight and exit the "
                "frame without incident. A silver sedan is also visible and "
                "does not collide with the red SUVs anywhere in the clip.\n"
                "Final answer: No"
            ),
        },
    ],
    # Real, publicly-documented GT from github.com/MoyoG/FETV's own README
    # (not private test data) -- used to calibrate the model's position/
    # lane grid convention, which it otherwise has no reference for.
    "fetv_structured": [
        {
            "video_path": os.path.join(os.path.dirname(_VIDEO_ROOT),
                                        "fetv", "FETV_public_clips", "001_001.mp4"),
            "question": None,  # filled in by fetv_submission.py with its own PROMPT
            "answer_text": json.dumps({
                "answer_date": "2018-07-17",
                "answer_time": "06:04:50",
                "answer_violation_type": "jaywalking",
                "answer_violator_type": "pedestrian",
                "answer_color": "mixed",
                "answer_initial_position": "Bottom-Right",
                "answer_final_position": "Middle-Right",
                "answer_initial_lane": "na",
                "answer_final_lane": "na",
                "answer_intersection_type": "T-intersection",
                "answer_weather": "clear",
                "answer_light": "daylight",
                "answer_description": (
                    "On 2018-07-17 at 06:04:50, a pedestrian wearing "
                    "mixed-colored clothing crossed the roadway outside the "
                    "marked crosswalk at a T-intersection, jaywalking from "
                    "the bottom-right of the frame toward the middle-right "
                    "while vehicle traffic was present."
                ),
            }),
        },
        {
            "video_path": os.path.join(os.path.dirname(_VIDEO_ROOT),
                                        "fetv", "FETV_public_clips", "002_014.mp4"),
            "question": None,
            "answer_text": json.dumps({
                "answer_date": "2018-07-17",
                "answer_time": "17:06:47",
                "answer_violation_type": "wrong_way",
                "answer_violator_type": "motorcycle",
                "answer_color": "red",
                "answer_initial_position": "Top-Right",
                "answer_final_position": "Middle-Left",
                "answer_initial_lane": "1",
                "answer_final_lane": "2",
                "answer_intersection_type": "T-intersection",
                "answer_weather": "clear",
                "answer_light": "daylight",
                "answer_description": (
                    "On 2018-07-17 at 17:06:47, a traffic incident occurred "
                    "at a T-intersection. A red motorcycle traveled against "
                    "the flow of traffic, moving from lane 1 into lane 2 as "
                    "it crossed from the top-right of the frame toward the "
                    "middle-left in a wrong-way driving violation."
                ),
            }),
        },
    ],
    "mcq": [
        {
            "video_path": os.path.join(_VIDEO_ROOT, "Accident-Bench/land_space/long/videos/000001.mp4"),
            "question": (
                "Regarding the collision involving the black and blue truck "
                "between 00:53 and 00:57, what is the correct sequence of "
                "events?\nA. The truck loses control and collides with both the "
                "road boundary and a pedestrian simultaneously.\nB. The truck "
                "crashes into the road boundary first, then collides with a "
                "pedestrian.\nC. The truck collides with another vehicle, which "
                "then pushes the truck into the road boundary and a pedestrian."
                "\nD. The truck collides with a pedestrian first, then crashes "
                "into the road boundary.\nAnswer with a single letter."
            ),
            "answer_text": (
                "The truck moves from the top of the frame toward the bottom. "
                "It first loses control and collides with the road boundary on "
                "the left side of the frame. Only after this initial impact "
                "does it go on to collide with a pedestrian.\nFinal answer: B"
            ),
        },
    ],
}

# Diagnostic knob: after the fps/video_metadata crash fix, real portal scoring
# showed MCQ regress (0.575 -> 0.425) even though BCQ and Temporal mIoU both
# improved. Frame *selection* (which frames get sampled, via MAX_FRAMES/
# MIN_FRAMES) is unaffected by the fix either way -- only the fps VALUE told
# to the model's M-RoPE temporal position encoding changed, from an implicit
# ~24 (whatever the old crashing-but-sometimes-limping-through default was)
# to the real extracted fps (~29.97 for these clips). Setting this env var
# forces that one value while leaving everything else (frame indices, crash
# safety) untouched, to isolate whether the fps *value* itself is what's
# driving the MCQ regression, vs. sample noise on a 50%-subset scored run.
TAR_FPS_OVERRIDE = os.environ.get("TAR_FPS_OVERRIDE")
TAR_FPS_OVERRIDE = float(TAR_FPS_OVERRIDE) if TAR_FPS_OVERRIDE else None


class QwenVLBackend:
    def __init__(self, quant="bf16", device="cuda", dtype=torch.bfloat16, verbose=True):
        from transformers import AutoProcessor, BitsAndBytesConfig

        if "qwen3" in MODEL_ID.lower():
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

        self.verbose = verbose
        self._log(f"Loading {MODEL_ID} (quant={quant}) from cache_dir={HF_CACHE} ...")

        quant_config = None
        kwargs = {}
        if quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["quantization_config"] = quant_config
        elif quant == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["quantization_config"] = quant_config
        elif quant == "bf16":
            kwargs["torch_dtype"] = dtype
        else:
            raise ValueError(f"unknown quant mode {quant!r}")

        self.model = ModelClass.from_pretrained(
            MODEL_ID,
            cache_dir=HF_CACHE,
            device_map=device,
            attn_implementation="sdpa",
            **kwargs,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, cache_dir=HF_CACHE, min_pixels=64 * 28 * 28,
            max_pixels=MAX_PIXELS_PER_FRAME,
        )
        self._log("Model loaded.")

    def _log(self, msg):
        if self.verbose:
            print(f"[QwenVLBackend] {msg}", file=sys.stderr)

    @torch.inference_mode()
    def _generate_once(self, inputs, max_new_tokens, do_sample):
        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out_text.strip()

    @staticmethod
    def _extract_time_window(question, pad_s=3.0):
        """Many bcq/mcq questions reference a specific MM:SS-MM:SS window
        (e.g. "between 00:53 and 00:57") within a longer clip, but our
        uniform MAX_FRAMES=16 sampling spreads frames across the WHOLE
        clip regardless -- for a clip much longer than the referenced
        window, most sampled frames land outside it entirely. qwen_vl_utils
        natively supports video_start/video_end (seconds) to concentrate
        sampling on a sub-range; use it whenever we can find two
        timestamps in the question, padded a few seconds on each side so
        we don't lose context immediately around the window."""
        # Checked PSI-VQA's own "t=0s to t=8.50s" style separately -- those
        # videos are already pre-trimmed to exactly that window (verified
        # via ffprobe), so this windowing would be a no-op there. This
        # regex is Track3-specific (MM:SS clips that run much longer than
        # the sub-window referenced in the question).
        times = re.findall(r"\b(\d{1,2}):(\d{2})(?:\.\d+)?\b", question)
        if len(times) < 2:
            return None, None
        secs = sorted(int(m) * 60 + int(s) for m, s in times)
        start, end = secs[0], secs[-1]
        if end - start < 1:
            return None, None
        return max(0.0, start - pad_s), end + pad_s

    @staticmethod
    def _extract_final(text, kind):
        """Pull the token after the last 'Final answer:' marker; fall back to
        scanning the whole text. Returns None if nothing extractable."""
        m = re.findall(r"final answer\s*:\s*\(?([A-Za-z]+)\)?", text, re.IGNORECASE)
        cand = m[-1] if m else None
        if kind == "yesno":
            if cand and cand.lower() in ("yes", "no"):
                return cand.capitalize()
            m2 = re.findall(r"\b(yes|no)\b", text, re.IGNORECASE)
            return m2[-1].capitalize() if m2 else None
        if kind == "letter":
            if cand and re.fullmatch(r"[A-Da-d]", cand):
                return cand.upper()
            # Fallback (Final answer: marker missing, usually truncation):
            # a bare \b([A-D])\b scan matches the English article "A" in
            # ordinary sentences ("A white sedan...") and was picking that
            # up as the answer. Restrict to answer-shaped contexts only.
            m2 = re.findall(r"(?:answer\s+is|option)\s*:?\s*\(?([A-D])\)?", text, re.IGNORECASE)
            if m2:
                return m2[-1].upper()
            m3 = re.findall(r"^\s*\(?([A-D])\)[.:]", text, re.MULTILINE)
            return m3[-1].upper() if m3 else None
        return None

    @torch.inference_mode()
    def answer(self, video_path: str, task_type: str, question: str,
               samples: int = 1, fewshot: bool = False) -> str:
        spec = build_prompt(task_type, question)
        messages = [{"role": "system", "content": spec.system}]
        if fewshot:
            for ex in FEWSHOT_EXAMPLES.get(task_type, []):
                ex_question = ex["question"] if ex["question"] is not None else question
                ex_video = {
                    "type": "video",
                    "video": ex["video_path"],
                    "max_frames": MAX_FRAMES,
                    "min_frames": MIN_FRAMES,
                    "max_pixels": MAX_PIXELS_PER_FRAME,
                }
                # The exemplar's own worked answer often narrates a specific
                # MM:SS window (e.g. the mcq exemplar's 00:53-00:57 collision
                # inside a 104s source video) -- without windowing, its 16
                # frames spread uniformly over the WHOLE source video, so
                # the exemplar shows the model confidently describing an
                # event barely any of its own frames actually cover.
                if ex["question"] is not None:
                    ex_start, ex_end = self._extract_time_window(ex_question)
                    if ex_start is not None:
                        ex_video["video_start"] = ex_start
                        ex_video["video_end"] = ex_end
                messages.append({
                    "role": "user",
                    "content": [
                        ex_video,
                        {"type": "text", "text": build_prompt(task_type, ex_question).user},
                    ],
                })
                messages.append({"role": "assistant", "content": ex["answer_text"]})
        video_start, video_end = self._extract_time_window(question)
        video_content = {
            "type": "video",
            "video": video_path,
            "max_frames": MAX_FRAMES,
            "min_frames": MIN_FRAMES,
            "max_pixels": MAX_PIXELS_PER_FRAME,
        }
        if video_start is not None:
            video_content["video_start"] = video_start
            video_content["video_end"] = video_end
        messages.append({
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": spec.user},
            ],
        })

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info

        # return_video_metadata=True is required here, not just for M-RoPE
        # accuracy: without it, process_vision_info returns video_kwargs["fps"]
        # as a per-video list (not a scalar), and the current transformers/
        # huggingface_hub version enforces strict TypedDict validation on
        # processor kwargs -- passing a list where int/float/None is expected
        # raises StrictDataclassFieldValidationError on every single call.
        # Confirmed: reverting this crashes 80/80 sampled clips, including
        # actual tar_test production videos.
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True
        )
        if video_inputs is not None:
            video_metadata = [v[1] for v in video_inputs]
            video_inputs = [v[0] for v in video_inputs]
            if TAR_FPS_OVERRIDE is not None:
                for vm in video_metadata:
                    vm["fps"] = TAR_FPS_OVERRIDE
            video_kwargs["video_metadata"] = video_metadata
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        # Reason-then-extract path (bcq/mcq): the model reasons freely, we
        # submit only the clean final token so the organizers' extraction
        # regex can never latch onto a stray yes/no inside the reasoning.
        if spec.final_answer:
            n = max(1, samples) if spec.self_consistency else 1
            votes = []
            for i in range(n):
                out = self._generate_once(inputs, spec.max_new_tokens, do_sample=(i > 0))
                tok = self._extract_final(out, spec.final_answer)
                if tok:
                    votes.append(tok)
            if not votes:
                return ""
            # majority vote; ties broken by first (greedy) sample
            counts = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == votes[0]))
            return best[0]

        return self._generate_once(inputs, spec.max_new_tokens, do_sample=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--task_type", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--quant", default="bf16", choices=["4bit", "8bit", "bf16"])
    args = ap.parse_args()

    backend = QwenVLBackend(quant=args.quant)
    ans = backend.answer(args.video, args.task_type, args.question)
    print("---ANSWER---")
    print(ans)


if __name__ == "__main__":
    main()
