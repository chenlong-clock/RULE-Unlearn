#!/bin/bash
# filepath: /mnt/usercache/zhangchenlong/RL-Unlearning/EasyR1/check_and_run_gpu.sh

TARGET_SCRIPT="run_rwku.sh"
THRESHOLD=5

# 检查一次 GPU 是否空闲
check_gpu_idle_once() {
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    for util in $gpu_util; do
        if [ "$util" -gt "$THRESHOLD" ]; then
            return 1  # Busy
        fi
    done
    return 0  # Idle
}

echo "Starting GPU monitor script. Will check every 10 seconds..."
echo "Will run: $TARGET_SCRIPT when GPU is idle for 1 minute"

# 主循环：每30秒检查一次是否连续6次（每10s一次）都空闲
while true; do
    date
    echo "Checking GPU utilization for 1-minute window..."

    all_idle=true
    for i in {1..6}; do
        if ! check_gpu_idle_once; then
            all_idle=false
            break
        fi
        sleep 10
    done

    if $all_idle; then
        echo "GPU has been idle for 1 minute. Starting the target script..."
        bash "$TARGET_SCRIPT"
        echo "Script execution completed at $(date)"
        exit 0
    else
        echo "GPU not idle for the full minute. Waiting 10 mins before checking again..."
        sleep 600
    fi
done
