#!/bin/bash
set -euo pipefail

# Launch phytoplankton binary training (train_swin_cnn_phyto.py) in a detached
# screen session, mirroring runs/training_scripts/launch_training.sh.
#
# Backbone to train. One of: swin_t | swin_s | efficientnet_b0
# Override without editing:  MODEL_BACKBONE=swin_s ./phytoplankton/scripts/launch_training.sh
MODEL_BACKBONE="${MODEL_BACKBONE:-swin_t}"

case "$MODEL_BACKBONE" in
    swin_t|swin_s|efficientnet_b0) ;;
    *) echo "Unsupported MODEL_BACKBONE='${MODEL_BACKBONE}' (use swin_t, swin_s or efficientnet_b0)"; exit 1 ;;
esac

# Also launch a RandomBackgroundNoise run alongside the baseline. Off by default
# for phytoplankton -- the synthetic-dirt augmentation was calibrated on the
# zooplankton dataset and hasn't been checked against real messy phyto captures.
#   RUN_NOISE=1 ./phytoplankton/scripts/launch_training.sh
RUN_NOISE="${RUN_NOISE:-0}"

GPU_BASELINE="${GPU_BASELINE:-0}"
GPU_NOISE="${GPU_NOISE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_PYTHON=/home/tjhinale/repo/HierNet/hiernet-venv/bin/python3
TRAIN_SCRIPT="$SCRIPT_DIR/train_swin_cnn_phyto.py"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_DIR="$REPO_ROOT/phytoplankton/logs"
mkdir -p "$LOG_DIR"

launch_run() {
    local gpu="$1"
    local noise_flag="$2"
    local variant run_name log_file
    if [ "$noise_flag" = "1" ]; then variant="noise"; else variant="baseline"; fi
    run_name="phyto_${MODEL_BACKBONE}_cnn_${variant}"
    log_file="$LOG_DIR/${run_name}.log"

    if screen -list 2>/dev/null | grep -q "\.${run_name}[[:space:]]"; then
        echo "screen session '${run_name}' is already running -- skipping."
        echo "  reattach: screen -r ${run_name}"
        echo "  stop it first if you want to relaunch: screen -X -S ${run_name} quit"
        return
    fi

    echo "Launching '${run_name}' on GPU ${gpu} -> ${log_file}"
    screen -dmS "$run_name" bash -c "
        cd '${REPO_ROOT}'
        CUDA_VISIBLE_DEVICES=${gpu} \
        MODEL_BACKBONE='${MODEL_BACKBONE}' \
        PHYTO_NOISE_AUG=${noise_flag} \
        NUM_WORKERS=${NUM_WORKERS} \
        ${MAX_EPOCHS:+MAX_EPOCHS=${MAX_EPOCHS}} \
        ${PHYTOPLANKTON_DATA_DIR:+PHYTOPLANKTON_DATA_DIR='${PHYTOPLANKTON_DATA_DIR}'} \
        PYTHONUNBUFFERED=1 '${VENV_PYTHON}' '${TRAIN_SCRIPT}' 2>&1 | tee '${log_file}'
    "
    echo "  reattach: screen -r ${run_name}"
}

echo "Backbone: ${MODEL_BACKBONE} | noise run: ${RUN_NOISE}"
# baseline -> training_result/phytoplankton/<backbone>_cnn_baseline_<date>
launch_run "$GPU_BASELINE" 0
if [ "$RUN_NOISE" = "1" ]; then
    # noise -> training_result/phytoplankton/<backbone>_cnn_noise_<date>
    launch_run "$GPU_NOISE" 1
fi

echo
echo "Active screen sessions:"
screen -list || true
