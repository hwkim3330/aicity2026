#!/usr/bin/env python3
"""
Qwen2.5-VL-7B-Instruct baseline inference for AI City Challenge 2026
Track 3 (Anomalous Events in Transportation / TAR).

Loads the model once (4-bit NF4 quantized via bitsandbytes, ~6-8GB VRAM)
and exposes `answer_one(video_path, task_type, question)` plus a CLI for
smoke-testing a single clip.

Usage:
    python inference.py --video /path/to/clip.mp4 \
        --task_type bcq \
        --question "Does a collision occur in the video?\nAnswer with only Yes or No."
"""

import argparse
import os
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
MAX_FRAMES = 16
MIN_FRAMES = 4
MAX_PIXELS_PER_FRAME = 360 * 420  # keeps token count per frame low


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
    def answer(self, video_path: str, task_type: str, question: str) -> str:
        spec = build_prompt(task_type, question)
        messages = [
            {"role": "system", "content": spec.system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_frames": MAX_FRAMES,
                        "min_frames": MIN_FRAMES,
                    },
                    {"type": "text", "text": spec.user},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info

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
            max_new_tokens=spec.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return out_text.strip()


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
