#!/usr/bin/env python3
"""Render frames for the qualitative case studies from a local dataset copy.

No video frames are committed to this repository. The FETV clips are
distributed through the challenge portal and PSI-VQA inherits the TASI
Benchmark Data Sharing Agreement, so redistribution terms are not established
for either. This script regenerates the figures from data you already hold.

    python3 scripts/render_case_studies.py --list
    python3 scripts/render_case_studies.py \
        --case psi_mcq_output_contract_failure \
        --data-root track3_anomaly/data/psi_vqa
    python3 scripts/render_case_studies.py \
        --case fetv_violator_misselection \
        --data-root /path/to/FETV_public_clips

Requires ffmpeg. Output goes to docs/case_studies/rendered/ which is
gitignored.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASES = os.path.join(ROOT, "docs", "case_studies")
OUT = os.path.join(CASES, "rendered")

# case_id -> (relative clip path within --data-root, frame timestamps in seconds)
CLIPS = {
    "psi_mcq_output_contract_failure": ("train/videos/ambiguous/video_0026_track_25.mp4",
                                        [0.0, 1.0, 2.0]),
    "fetv_violator_misselection": ("019_004.mp4", [0.0, 1.5, 3.0]),
}


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def render(case_id, data_root):
    meta_path = os.path.join(CASES, case_id + ".json")
    if not os.path.exists(meta_path):
        sys.exit(f"unknown case '{case_id}' (no {meta_path})")
    meta = json.load(open(meta_path))

    rel, stamps = CLIPS[case_id]
    clip = os.path.join(data_root, rel)
    if not os.path.isfile(clip):
        sys.exit(f"clip not found: {clip}\n"
                 f"point --data-root at the directory containing '{rel}'")

    dur = probe_duration(clip)
    if dur:
        print(f"clip duration {dur:.2f}s")
        stamps = [t for t in stamps if t < dur] or [0.0]

    dest = os.path.join(OUT, case_id)
    os.makedirs(dest, exist_ok=True)
    for t in stamps:
        png = os.path.join(dest, f"t{t:05.2f}.png")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-ss", str(t), "-i", clip, "-frames:v", "1", png],
            check=True)
        print(f"  wrote {png}")

    summary = os.path.join(dest, "CASE.txt")
    with open(summary, "w") as f:
        f.write(f"{meta['case_id']}\n\n")
        f.write(f"Claim under test: {meta['claim_under_test']}\n\n")
        if "verdict" in meta:
            f.write(f"Verdict: {meta['verdict']}\n\n")
        for key in ("diagnosis", "what_this_shows"):
            if key in meta:
                f.write(f"{key}:\n")
                val = meta[key]
                for line in (val if isinstance(val, list) else [val]):
                    f.write(f"  - {line}\n")
                f.write("\n")
    print(f"  wrote {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="case id (see --list)")
    ap.add_argument("--data-root", help="local dataset directory")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.case:
        print("available cases:")
        for cid, (rel, _) in CLIPS.items():
            print(f"  {cid}\n      clip: <data-root>/{rel}")
        return
    if not args.data_root:
        sys.exit("--data-root is required")
    render(args.case, args.data_root)


if __name__ == "__main__":
    main()
