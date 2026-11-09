#!/usr/bin/env bash
# Explicit opt-in submission; a 12-task array, capped at one concurrent GPU.
#SBATCH --job-name=pendulum-basic
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11%1
#SBATCH --output=pendulum-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
PYTHON_BIN="${PENDULUM_PYTHON:-.venv/bin/python}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?}"
if [[ ! "$TASK_INDEX" =~ ^[0-9]+$ ]] || (( TASK_INDEX > 11 )); then
  echo "Expected array index 0..11, got $TASK_INDEX" >&2
  exit 2
fi
TRAIN_SEED=$((TASK_INDEX / 4))
MODEL_INDEX=$((TASK_INDEX % 2))
DAMP_INDEX=$(((TASK_INDEX / 2) % 2))
MODELS=(lehami lewm)
DAMPINGS=(0p0 0p2)
MODEL_NAME="${MODELS[$MODEL_INDEX]}"
DATA_ROOT="${PENDULUM_DATA_DIR:-data/basic}"
DATA_FILE="${DATA_ROOT}/pendulum_b${DAMPINGS[$DAMP_INDEX]}.h5"
RUN_DIR="runs/server_${SLURM_ARRAY_JOB_ID}/${MODEL_NAME}_b${DAMPINGS[$DAMP_INDEX]}_seed${TRAIN_SEED}"
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
"$PYTHON_BIN" -m dpbench.train --data "$DATA_FILE" --model "$MODEL_NAME" --output "$RUN_DIR" \
  --device cuda --steps "${PENDULUM_STEPS:-5000}" --batch-size 32 --seed "$TRAIN_SEED" \
  --train-frames 81 --unroll 8 --lr 5e-5 --weight-decay 1e-3 \
  --val-every 250 --val-windows 256 --warmup 20 --threads 4 --workers 0
"$PYTHON_BIN" -m dpbench.evaluate --data "$DATA_FILE" --checkpoint "$RUN_DIR/best.pt" \
  --output "$RUN_DIR/evaluation" --device cuda --seed "$TRAIN_SEED" --horizons 4 8 16 32 \
  --labels 4096 --val-labels 1024 --probe-steps 500 --probes ridge mlp --threads 4
