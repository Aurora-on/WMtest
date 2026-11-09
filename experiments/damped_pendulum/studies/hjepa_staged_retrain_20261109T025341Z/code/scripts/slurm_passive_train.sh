#!/usr/bin/env bash
#SBATCH --job-name=passive-jepa-s0
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the verified Python environment}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
"$PYTHON_BIN" -m dpbench.train_dissipative \
  --data ../data/passive_family_v1/seen.h5 \
  --config configs/dissipative_passive_family.json \
  --output "../runs/seed0_${SLURM_JOB_ID}" --device cuda --execute
