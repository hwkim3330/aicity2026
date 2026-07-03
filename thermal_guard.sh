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
NVME_WARN_C=72
NVME_CRIT_C=80

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
        echo "$(date '+%F %T') CRITICAL: NVMe ${nvme_temp}C >= ${NVME_CRIT_C}C -- consider pausing disk-heavy jobs now"
    elif [ -n "$nvme_temp" ] && [ "$nvme_temp" -ge "$NVME_WARN_C" ]; then
        echo "$(date '+%F %T') warning: NVMe ${nvme_temp}C >= ${NVME_WARN_C}C"
    fi

    sleep 15
done
