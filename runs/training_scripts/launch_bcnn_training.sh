#!/bin/bash
set -euo pipefail

# BCNN backbone to train. One of: densenet121 | resnet50 | vgg16
# Override without editing:  BCNN_BACKBONE=resnet50 ./runs/launch_bcnn_training.sh
BCNN_BACKBONE="${BCNN_BACKBONE:-densenet121}"

# Which GPU to use. The swin runs use GPU 0 by default, so this defaults to 1.
BCNN_GPU="${BCNN_GPU:-1}"

case "$BCNN_BACKBONE" in
    densenet121|resnet50|vgg16) ;;
    *) echo "Unsupported BCNN_BACKBONE='${BCNN_BACKBONE}' (use densenet121, resnet50 or vgg16)"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PYTHON=/home/tjhinale/repo/HierNet/hiernet-venv/bin/python3
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_DIR="$REPO_ROOT/runs/logs"
mkdir -p "$LOG_DIR"

launch_run() {
    local gpu="$1"
    local noise_flag="$2"
    local run_name
    if [ "$noise_flag" = "1" ]; then
        run_name="bcnn_${BCNN_BACKBONE}_noise"
    else
        run_name="bcnn_${BCNN_BACKBONE}_baseline"
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
        CUDA_VISIBLE_DEVICES=${gpu} BCNN_BACKBONE=${BCNN_BACKBONE} BCNN_NOISE=${noise_flag} NUM_WORKERS=${NUM_WORKERS} PYTHONUNBUFFERED=1 '${VENV_PYTHON}' runs/training_scripts/train_bcnn_mnr.py 2>&1 | tee '${log_file}'
    "
    echo "  reattach: screen -r ${run_name}"
}

echo "BCNN backbone: ${BCNN_BACKBONE} | GPU: ${BCNN_GPU}"
launch_run "$BCNN_GPU" 1   # with-noise run -> training_result/mnr/bcnn_<backbone>_noise_<date>
launch_run "$BCNN_GPU" 0   # baseline run  -> training_result/mnr/bcnn_<backbone>_baseline_<date>

echo
echo "Active screen sessions:"
screen -list || true
