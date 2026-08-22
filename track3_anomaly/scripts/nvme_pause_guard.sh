#!/usr/bin/env bash
# Pause a named job while the NVMe is hot, resume when it cools.
#
# The system guard (/usr/local/bin/thermal-guard.sh) is running and logs
# "CRITICAL: NVMe 72C >= 68C -- no known heavy jobs running", because its
# HEAVY_PATTERNS list is train_yolo.py|finetune_clip.py|embed_gallery.py|
# make_submission.py and the job actually hammering the drive is a video-reading
# inference loop. So the guard sees the temperature and can do nothing about it.
#
# This machine's hard crashes have historically been the QLC root SSD passing
# ~70C, not the CPU or GPU, and a crash here costs hours of GPU work. SIGSTOP is
# reversible and loses no progress, unlike killing the job.
#
# Editing the system guard needs root; this needs nothing. Run it beside any long
# NVMe-heavy job:
#   ./nvme_pause_guard.sh 'psi_mcq_cv.py --n 321'
set -u
PATTERN=${1:?usage: nvme_pause_guard.sh <pgrep -f pattern>}
CRIT=${CRIT:-69}
RESUME=${RESUME:-64}
POLL=${POLL:-15}

temp() {
  cat /sys/class/nvme/nvme0/hwmon*/temp1_input 2>/dev/null | head -1 |
    awk '{printf "%.0f", $1/1000}'
}

paused=0
while true; do
  # Matching by pattern, never by a bare name that this script's own command
  # line would also match -- a self-match here would stop the guard instead of
  # the job.
  mapfile -t pids < <(pgrep -f -- "$PATTERN" | grep -v "^$$\$")
  if [[ ${#pids[@]} -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] job gone, exiting (paused=$paused)"
    exit 0
  fi
  t=$(temp)
  if [[ -z $t ]]; then sleep "$POLL"; continue; fi
  if (( paused == 0 && t >= CRIT )); then
    kill -STOP "${pids[@]}" 2>/dev/null && paused=1
    echo "[$(date +%H:%M:%S)] NVMe ${t}C >= ${CRIT} -- paused ${#pids[@]} pid(s)"
  elif (( paused == 1 && t <= RESUME )); then
    kill -CONT "${pids[@]}" 2>/dev/null && paused=0
    echo "[$(date +%H:%M:%S)] NVMe ${t}C <= ${RESUME} -- resumed"
  fi
  sleep "$POLL"
done
