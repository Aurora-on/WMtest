#!/usr/bin/env bash
#SBATCH --job-name=pendulum-holdout
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=holdout-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}/code"
export PYTHONDONTWRITEBYTECODE=1
/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python -B -m dpbench.evaluate_holdout --study ../study --output ../results --device cuda --batch-size 16 --threads 4 --horizons 4 8 16 32
