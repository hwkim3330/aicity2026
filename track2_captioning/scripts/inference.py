#!/usr/bin/env python3
"""
Qwen2.5-VL-7B-Instruct baseline for AI City Challenge 2026 Track 2
(Transportation Safety Captioning + VQA), trained-free / prompting-only
baseline. Same backend pattern as track3_anomaly/scripts/inference.py.

Exposes:
    caption_segment(video_path, start_time, end_time) -> (caption_pedestrian, caption_vehicle)
    answer_mcq(video_path, question, options) -> one of "a"/"b"/"c"/"d"
"""

import os
import re
import sys

import torch

MODEL_ID = os.environ.get("T2_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
HF_CACHE = os.environ.get(
    "T2_HF_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_cache"),
)

MAX_FRAMES = 12
MIN_FRAMES = 2
MAX_PIXELS_PER_FRAME = 360 * 420

CAPTION_SYSTEM = (
    "You are an expert traffic-safety video analyst modeling the WTS "
    "(Woven Traffic Safety) dataset annotation style. You describe a "
    "specific time segment of a pedestrian-vehicle interaction in dense, "
    "factual, third-person prose covering: pedestrian appearance (age, "
    "height, clothing), pedestrian position/orientation/gaze/action "
    "relative to the vehicle, and separately the vehicle's position, "
    "field of view of the pedestrian, action and speed. Do not add "
    "disclaimers about being an AI."
)

VQA_SYSTEM = (
    "You are an expert traffic-safety video analyst. Answer the "
    "multiple-choice question about the clip. Respond with exactly one "
    "letter: a, b, c, or d. No punctuation, no explanation."
)


class QwenVLBackend:
    def __init__(self, quant="bf16", device="cuda", dtype=torch.bfloat16, verbose=True):
        from transformers import AutoProcessor, BitsAndBytesConfig

        if "qwen3" in MODEL_ID.lower():
            from transformers import Qwen3VLForConditionalGeneration as ModelClass
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

        self.verbose = verbose
        self._log(f"Loading {MODEL_ID} (quant={quant}) from cache_dir={HF_CACHE} ...")

        kwargs = {}
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
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

    def _generate(self, video_path, start_time, end_time, system, user, max_new_tokens):
        from qwen_vl_utils import process_vision_info

        video_content = {
            "type": "video",
            "video": video_path,
            "max_frames": MAX_FRAMES,
            "min_frames": MIN_FRAMES,
        }
        if start_time is not None:
            video_content["video_start"] = float(start_time)
        if end_time is not None:
            video_content["video_end"] = float(end_time)

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [video_content, {"type": "text", "text": user}],
            },
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def caption_segment(self, video_path, start_time, end_time):
        pedestrian = self._generate(
            video_path, start_time, end_time, CAPTION_SYSTEM,
            "Describe the PEDESTRIAN's appearance, position, orientation, "
            "line of sight, and action during this segment, plus the "
            "weather/road context.",
            220,
        )
        vehicle = self._generate(
            video_path, start_time, end_time, CAPTION_SYSTEM,
            "Describe the VEHICLE's position relative to the pedestrian, "
            "its field of view of the pedestrian, its action and speed "
            "during this segment, plus the weather/road context.",
            220,
        )
        return pedestrian, vehicle

    def answer_mcq(self, video_path, question, options, start_time=None, end_time=None):
        letters = [k for k in "abcd" if k in options]
        opts_text = "\n".join(f"{k}) {options[k]}" for k in letters)
        prompt = f"{question}\n{opts_text}"
        raw = self._generate(video_path, start_time, end_time, VQA_SYSTEM, prompt, 4)
        m = re.search(f"[{''.join(letters)}]", raw.lower())
        return m.group(0) if m else letters[0]


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--mode", choices=["caption", "vqa"], default="caption")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--question", default=None)
    ap.add_argument("--quant", default="bf16", choices=["4bit", "8bit", "bf16"])
    args = ap.parse_args()

    backend = QwenVLBackend(quant=args.quant)
    if args.mode == "caption":
        ped, veh = backend.caption_segment(args.video, args.start, args.end)
        print("---PEDESTRIAN---")
        print(ped)
        print("---VEHICLE---")
        print(veh)
    else:
        ans = backend.answer_mcq(
            args.video, args.question,
            {"a": "opt_a", "b": "opt_b", "c": "opt_c", "d": "opt_d"},
        )
        print("---ANSWER---", ans)


if __name__ == "__main__":
    main()
