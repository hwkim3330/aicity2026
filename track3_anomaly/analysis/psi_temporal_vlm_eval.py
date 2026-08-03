#!/usr/bin/env python3
"""Run the official PSI temporal-localization inference configuration over the
PSI-VQA *training* split, so the VLM can be scored against real ground truth
and compared head-to-head with the metadata-only priors.

This uses exactly the backend, task type, and decoding settings that produced
the shipped temporal rows of psi_vqa_submission_v3/v5 (Qwen3-VL-8B, bf16,
decord with real-fps metadata). Predictions are appended incrementally, so a
crash costs at most one item, and `--resume` continues an interrupted run.

Also records per-item latency and peak VRAM for BENCHMARKS.md.

    python3 psi_temporal_vlm_eval.py --limit 0 \
        --out ../results/psi_temporal_vlm_train_preds.jsonl \
        --stats ../results/psi_temporal_vlm_train_stats.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

GT = "../data/psi_vqa/train/temporal_localization.json"
VIDEO_ROOT = "../data/psi_vqa/train/videos/temporal"


def resolve(video_id, root):
    rel = video_id.split("PSI/", 1)[-1]
    rel = rel.split("temporal/", 1)[-1]
    return os.path.join(root, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--videos", default=VIDEO_ROOT)
    ap.add_argument("--out", default="../results/psi_temporal_vlm_train_preds.jsonl")
    ap.add_argument("--stats", default="../results/psi_temporal_vlm_train_stats.json")
    ap.add_argument("--limit", type=int, default=0, help="0 = all items")
    ap.add_argument("--quant", default="bf16", choices=["4bit", "8bit", "bf16"])
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    items = json.load(open(args.gt))["items"]
    items = [it for it in items if os.path.isfile(resolve(it["video_id"], args.videos))]
    if args.limit:
        items = items[:args.limit]

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["item_index"])
        print(f"[resume] {len(done)} already done", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    import torch
    from inference import QwenVLBackend, MODEL_ID, MAX_FRAMES, MAX_PIXELS_PER_FRAME

    torch.cuda.reset_peak_memory_stats()
    load_t0 = time.time()
    backend = QwenVLBackend(quant=args.quant)
    load_s = time.time() - load_t0
    peak_after_load = torch.cuda.max_memory_allocated()

    lat, n_ok, n_err = [], 0, 0
    t0 = time.time()
    with open(args.out, "a") as out:
        for i, it in enumerate(items, 1):
            if it["item_index"] in done:
                continue
            vpath = resolve(it["video_id"], args.videos)
            it0 = time.time()
            try:
                pred = backend.answer(vpath, "temporal_localization", it["question"])
            except Exception as e:                                  # noqa: BLE001
                print(f"  ERROR {it['item_index']}: {e}", file=sys.stderr)
                n_err += 1
                continue
            dt = time.time() - it0
            lat.append(dt)
            n_ok += 1
            out.write(json.dumps({"item_index": it["item_index"],
                                  "video_id": it["video_id"],
                                  "prediction": pred,
                                  "gt": it["answer"],
                                  "seconds": round(dt, 3)}) + "\n")
            out.flush()
            if i % 20 == 0 or i == len(items):
                print(f"[{i}/{len(items)}] ok={n_ok} err={n_err} "
                      f"{sum(lat)/max(1,len(lat)):.2f}s/item", file=sys.stderr)
    wall = time.time() - t0

    stats = {
        "model_id": MODEL_ID,
        "precision": args.quant,
        "max_frames": MAX_FRAMES,
        "max_pixels_per_frame": MAX_PIXELS_PER_FRAME,
        "gpu": torch.cuda.get_device_name(0),
        "items_attempted": len(items) - len(done),
        "items_ok": n_ok,
        "items_error": n_err,
        "model_load_seconds": round(load_s, 2),
        "inference_wall_seconds": round(wall, 2),
        "seconds_per_item": round(sum(lat) / len(lat), 3) if lat else None,
        "peak_vram_bytes_after_load": int(peak_after_load),
        "peak_vram_bytes_total": int(torch.cuda.max_memory_allocated()),
        "peak_vram_gib_total": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "torch": torch.__version__,
    }
    json.dump(stats, open(args.stats, "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
