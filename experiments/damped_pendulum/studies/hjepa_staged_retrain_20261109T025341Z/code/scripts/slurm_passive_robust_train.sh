#!/usr/bin/env bash
#SBATCH --job-name=passive-robust-s0
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=robust-train-%j.log
set -euo pipefail
cd "${PENDULUM_CODE_DIR:?Set the validated frozen code snapshot directory}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the verified Python environment}"
DATA_FILE="${PENDULUM_DATA:?Set the seen train/validation HDF5 path}"
RUN_ROOT="${PENDULUM_RUN_ROOT:?Set the parent of new training output directories}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
TRAIN_ARGS=(--data "$DATA_FILE" --config configs/dissipative_passive_robust.json
            --output "$RUN_ROOT/seed0_${SLURM_JOB_ID}" --device cuda --execute)
if [[ -n "${PENDULUM_RESUME:-}" ]]; then
  TRAIN_ARGS+=(--resume "$PENDULUM_RESUME")
fi
exec "$PYTHON_BIN" -m dpbench.train_dissipative "${TRAIN_ARGS[@]}"
