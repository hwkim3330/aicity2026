#!/usr/bin/env bash
# run_pipeline.sh -- run detect -> track2d -> project3d -> fuse_mtmc -> export_submission
# for a given scene.
#
# Usage:
#   ./run_pipeline.sh --scene Warehouse_000 --split train --cameras "Camera_0000 Camera_0003 Camera_0015" \
#       --max-frames 300 --device cuda --out track1.txt
#
# All args optional except --scene; defaults: split=train, cameras=all available,
# max-frames=unlimited, device=cuda, out=track1.txt

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SCENE=""
SPLIT="train"
CAMERAS=""
MAX_FRAMES=""
DEVICE="cuda"
OUT="track1.txt"
MODEL="yolo11n.pt"
CONF="0.25"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --cameras) CAMERAS="$2"; shift 2;;
    --max-frames) MAX_FRAMES="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$SCENE" ]]; then
  echo "usage: $0 --scene Warehouse_000 [--split train] [--cameras \"Camera_0000 Camera_0003\"] [--max-frames 300] [--device cuda] [--out track1.txt]"
  exit 1
fi

CAM_ARGS=()
if [[ -n "$CAMERAS" ]]; then
  CAM_ARGS=(--cameras $CAMERAS)
fi
MF_ARGS=()
if [[ -n "$MAX_FRAMES" ]]; then
  MF_ARGS=(--max-frames "$MAX_FRAMES")
fi

echo "=== [1/5] detect.py ==="
python3 detect.py --scene "$SCENE" --split "$SPLIT" "${CAM_ARGS[@]}" "${MF_ARGS[@]}" \
    --device "$DEVICE" --model "$MODEL" --conf "$CONF"

echo "=== [2/5] track2d.py ==="
python3 track2d.py --scene "$SCENE" "${CAM_ARGS[@]}"

echo "=== [3/5] project3d.py ==="
python3 project3d.py --scene "$SCENE" --split "$SPLIT" "${CAM_ARGS[@]}"

echo "=== [4/5] fuse_mtmc.py ==="
python3 fuse_mtmc.py --scene "$SCENE"

echo "=== [5/5] export_submission.py ==="
python3 export_submission.py --scene "$SCENE" --out "$OUT"

echo "=== done: $OUT ==="
