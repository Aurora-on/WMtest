#!/usr/bin/env bash
#SBATCH --job-name=passive-efficient3-fresh-s0
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=efficient3-fresh-%j.log
set -euo pipefail

cd "${PENDULUM_CODE_DIR:?Set the frozen code snapshot directory}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the verified Python environment}"
DATA_FILE="${PENDULUM_DATA:?Set the seen train/validation HDF5 path}"
RUN_ROOT="${PENDULUM_RUN_ROOT:?Set a fresh experiment directory}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
# Fresh model and Adam state at seed0; no checkpoint loading or migration.
exec "$PYTHON_BIN" -m dpbench.train_dissipative \
  --data "$DATA_FILE" \
  --config configs/dissipative_passive_efficient_3substeps.json \
  --output "$RUN_ROOT/training" --device cuda --execute
