#!/bin/bash
set -euo pipefail

# Backbone to train. One of: efficientnet_b0 | swin_t | swin_s
# Override without editing:  MODEL_BACKBONE=swin_t ./runs/launch_training.sh
MODEL_BACKBONE="${MODEL_BACKBONE:-efficientnet_b0}"

case "$MODEL_BACKBONE" in
    efficientnet_b0|swin_t|swin_s) ;;
    *) echo "Unsupported MODEL_BACKBONE='${MODEL_BACKBONE}' (use efficientnet_b0, swin_t or swin_s)"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON=/home/tjhinale/repo/HierNet/hiernet-venv/bin/python3
NUM_WORKERS=8
LOG_DIR="$REPO_ROOT/runs/logs"
mkdir -p "$LOG_DIR"

launch_run() {
    local gpu="$1"
    local noise_flag="$2"
    local run_name
    if [ "$noise_flag" = "1" ]; then
        run_name="${MODEL_BACKBONE}_cnn_noise"
    else
        run_name="${MODEL_BACKBONE}_cnn_baseline"
    fi
    local log_file="$LOG_DIR/${run_name}.log"

    if screen -list 2>/dev/null | grep -q "\.${run_name}[[:space:]]"; then
        echo "screen session '${run_name}' is already running -- skipping."
        echo "  reattach: screen -r ${run_name}"
        echo "  stop it first if you want to relaunch: screen -X -S ${run_name} quit"
        return
    fi

    echo "Launching '${run_name}' on GPU ${gpu} -> ${log_file}"
    screen -dmS "$run_name" bash -c "
        cd '${REPO_ROOT}'
        CUDA_VISIBLE_DEVICES=${gpu} MODEL_BACKBONE=${MODEL_BACKBONE} SWIN_CNN_NOISE=${noise_flag} NUM_WORKERS=${NUM_WORKERS} PYTHONUNBUFFERED=1 '${VENV_PYTHON}' runs/training_scripts/train_swin_cnn_mnr.py 2>&1 | tee '${log_file}'
    "
    echo "  reattach: screen -r ${run_name}"
}

echo "Backbone: ${MODEL_BACKBONE}"
launch_run 0 1   # with-noise run, GPU 0 -> training_result/mnr/<backbone>_cnn_noise_<date>
launch_run 1 0   # baseline run,   GPU 1 -> training_result/mnr/<backbone>_cnn_baseline_<date>

echo
echo "Active screen sessions:"
screen -list || true
