#!/bin/bash
# Runs at boot (via crontab @reboot) to auto-resume any AI City Challenge
# jobs that were killed mid-run by a system crash.
set -u
cd /home/kim/aicity2026

T1=track1_3dperception
T2=track2_captioning
T3=track3_anomaly

log() { echo "[$(date '+%F %T')] $*" >> shared/resume.log; }

log "resume_all.sh triggered"

# Track 1: resume YOLO finetune if it hasn't reached 30 epochs yet
if [ ! -f "$T1/runs_finetune/warehouse7/weights/last.pt" ] || \
   ! grep -q "^ *30/30" "$T1/train_yolo.log" 2>/dev/null; then
  log "Track1: resuming train_yolo.py"
  cd "$T1"
  nohup python3 train_yolo.py >> train_yolo.log 2>&1 &
  cd /home/kim/aicity2026
else
  log "Track1: training already complete, skipping"
fi

# Track 2: resume val40 caption+vqa run if val_caption.json / val_vqa.json incomplete
if [ -d "$T2/submissions" ]; then
  log "Track2: resuming make_submission.py --resume"
  cd "$T2/scripts"
  nohup python3 make_submission.py --root ../data/data/data --split val \
    --out_caption ../submissions/val40_caption.json \
    --out_vqa ../submissions/val40_vqa.json \
    --limit 40 --quant 4bit --resume \
    >> ../submissions/val40_run.log 2>&1 &
  cd /home/kim/aicity2026
fi

# Track 3: resume anomaly submission fill-in (already covers 100% via
# fallback-on-missing-video, --resume skips item_index rows already in the CSV)
if [ -d "$T3/submissions" ]; then
  csv_rows=$(tail -n +2 "$T3/submissions/submission_qwen25vl_4bit.csv" 2>/dev/null | wc -l)
  if [ "${csv_rows:-0}" -lt 960 ]; then
    log "Track3: resuming make_submission.py --resume ($csv_rows/960 rows so far)"
    cd "$T3/scripts"
    nohup python3 make_submission.py \
      --test_json ../data/test/test.json \
      --media_root ../data/videos \
      --out ../submissions/submission_qwen25vl_4bit.csv \
      --quant 4bit --resume \
      >> ../scratch_full_run.log 2>&1 &
    cd /home/kim/aicity2026
  else
    log "Track3: submission already complete ($csv_rows/960 rows), skipping"
  fi
fi

log "resume_all.sh done dispatching jobs"
