#!/usr/bin/env bash
#SBATCH --job-name=visual-damping-compare
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%1
#SBATCH --output=visual-damping-compare-%A_%a.log
set -euo pipefail

# Explicit submission only. Each array member handles one method and seed.
# For three seeds: sbatch --array=0-8%1 with COMPARISON_SEEDS="0 1 2".
cd "${PENDULUM_CODE_DIR:?Set the verified comparison code directory}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the project Python executable}"
RUN_ROOT="${COMPARISON_RUN_ROOT:?Set a new output root}"
METHODS=(lehami lewm hjepa)
read -r -a SEEDS <<< "${COMPARISON_SEEDS:-0}"
INDEX="${SLURM_ARRAY_TASK_ID:?Submit as an explicit Slurm array}"
SEED_INDEX=$((INDEX / 3))
if (( SEED_INDEX >= ${#SEEDS[@]} )); then
  echo "Array index exceeds declared seeds" >&2
  exit 2
fi
METHOD="${METHODS[$((INDEX % 3))]}"
SEED="${SEEDS[$SEED_INDEX]}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$PYTHON_BIN" scripts/run_comparison.py --execute --device cuda --gpus 0,1 \
  --methods "$METHOD" --seeds "$SEED" --route "${COMPARISON_ROUTE:-staged}" \
  --data-a "${PENDULUM_DATA_A:?Set paired undamped data}" \
  --data-single "${PENDULUM_DATA_SINGLE:?Set paired b0.2 data}" \
  --data-multi "${PENDULUM_DATA_MULTI:?Set paired six-condition data}" \
  --output "$RUN_ROOT/${COMPARISON_ROUTE:-staged}/${METHOD}_seed${SEED}"
