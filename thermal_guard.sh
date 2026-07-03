#!/usr/bin/env bash
# Polls GPU + NVMe temperature. Steps the GPU power limit down when it runs hot
# (software mitigation -- the NVMe can't be cooled from software, so it just
# gets logged/alerted so a human can react before it hits critical and locks
# up the kernel again).
#
# Runs as a systemd service, logs to journal (see: journalctl -u thermal-guard).

GPU_WARN_C=78
GPU_STEP_DOWN_W=20
GPU_MIN_LIMIT_W=200
# Real crashes have historically correlated with NVMe hitting ~70C, well below
# the drive's own 85C critical spec -- this cheap DRAM-less drive apparently
# can't be trusted past that in practice, so we treat 70C as the real ceiling.
NVME_WARN_C=60
NVME_CRIT_C=68
# Processes that hammer the NVMe (dataset reads / checkpoint writes). Paused
# (SIGSTOP) at critical, resumed (SIGCONT) once back under the warn threshold --
# reversible, no lost training progress, unlike killing them.
HEAVY_PATTERNS="train_yolo.py|finetune_clip.py|embed_gallery.py|make_submission.py"
paused=0

while true; do
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    gpu_limit=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
    nvme_temp=$(nvme smart-log /dev/nvme0 2>/dev/null | grep -oP '^temperature\s*:\s*\K[0-9]+(?=\s*°C)')

    if [ -n "$gpu_temp" ] && [ "$gpu_temp" -ge "$GPU_WARN_C" ] && [ "$gpu_limit" -gt "$GPU_MIN_LIMIT_W" ]; then
        new_limit=$((gpu_limit - GPU_STEP_DOWN_W))
        [ "$new_limit" -lt "$GPU_MIN_LIMIT_W" ] && new_limit=$GPU_MIN_LIMIT_W
        nvidia-smi -pl "$new_limit" >/dev/null 2>&1
        echo "$(date '+%F %T') GPU ${gpu_temp}C >= ${GPU_WARN_C}C -- power limit ${gpu_limit}W -> ${new_limit}W"
    fi

    if [ -n "$nvme_temp" ] && [ "$nvme_temp" -ge "$NVME_CRIT_C" ]; then
        if [ "$paused" -eq 0 ]; then
            pids=$(pgrep -f "$HEAVY_PATTERNS")
            if [ -n "$pids" ]; then
                echo "$(date '+%F %T') CRITICAL: NVMe ${nvme_temp}C >= ${NVME_CRIT_C}C -- pausing: $pids"
                kill -STOP $pids 2>/dev/null
                paused=1
            else
                echo "$(date '+%F %T') CRITICAL: NVMe ${nvme_temp}C >= ${NVME_CRIT_C}C -- no known heavy jobs running"
            fi
        fi
    elif [ -n "$nvme_temp" ] && [ "$nvme_temp" -ge "$NVME_WARN_C" ]; then
        echo "$(date '+%F %T') warning: NVMe ${nvme_temp}C >= ${NVME_WARN_C}C"
    elif [ "$paused" -eq 1 ]; then
        pids=$(pgrep -f "$HEAVY_PATTERNS")
        if [ -n "$pids" ]; then
            echo "$(date '+%F %T') NVMe back under ${NVME_WARN_C}C -- resuming: $pids"
            kill -CONT $pids 2>/dev/null
        fi
        paused=0
    fi

    sleep 15
done
