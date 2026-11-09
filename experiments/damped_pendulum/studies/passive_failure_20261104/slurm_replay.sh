#!/usr/bin/env bash
#SBATCH --job-name=passive-failure-replay
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/weizhixin/ssd/dissipative_passive_review_20261104/replay-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/dissipative_passive_review_20261104/code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
"$PYTHON_BIN" scripts/replay_passive_failure.py \
 --data /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/data/passive_family_v1/seen.h5 \
 --config configs/dissipative_passive_family.json \
 --resume /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/runs/seed0_47541/last.pt \
 --output "/home/weizhixin/ssd/dissipative_passive_review_20261104/replay_${SLURM_JOB_ID}" \
 --device cuda --max-updates 60 --execute
