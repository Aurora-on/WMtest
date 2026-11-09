#!/usr/bin/env bash
#SBATCH --job-name=passive-data
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/data-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the verified Python environment}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/usr/bin/time -v "$PYTHON_BIN" -m dpbench.passive_family \
  --config configs/passive_family.json --generate --output ../data/passive_family_v1
