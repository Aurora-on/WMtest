#!/usr/bin/env bash
#SBATCH --job-name=passive-profile
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/profile-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the verified Python environment}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
"$PYTHON_BIN" scripts/profile_dissipative.py \
  --data ../data/passive_family_v1/seen.h5 \
  --config configs/dissipative_passive_family.json \
  --output "profile-${SLURM_JOB_ID}.json" --device cuda --warmup 2 --steps 4 --execute
