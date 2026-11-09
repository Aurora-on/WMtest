#!/usr/bin/env bash
#SBATCH --job-name=dp-matched-profile
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=matched-profile-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
PYTHON_BIN="${PENDULUM_PYTHON:-.venv/bin/python}"
"$PYTHON_BIN" scripts/server_matched_profile.py --output "runs/server_matched_${SLURM_JOB_ID}"
/usr/bin/time -p "$PYTHON_BIN" -m dpbench.generate --config configs/basic.json \
  --output "data/server_generated_${SLURM_JOB_ID}" > "simulation_${SLURM_JOB_ID}.log" 2>&1
