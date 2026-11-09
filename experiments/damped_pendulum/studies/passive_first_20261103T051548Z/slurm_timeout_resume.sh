#!/usr/bin/env bash
#SBATCH --job-name=passive-timeout-resume
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/weizhixin/ssd/dissipative_passive_20261103T051548Z/code/logs/timeout-resume-%j.log
set -euo pipefail
PARENT_JOB=47541
JOB_RECORD=$(scontrol show job "$PARENT_JOB" -o 2>&1) || {
  printf '%s\n' 'Parent state unavailable; no training started.' "$JOB_RECORD"
  exit 0
}
if [[ " $JOB_RECORD " != *" JobState=TIMEOUT "* ]]; then
  printf '%s\n' 'Parent was not confirmed TIMEOUT; no training started.' "$JOB_RECORD"
  exit 0
fi
cd /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/code
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
[[ -f ../runs/seed0_47541/last.pt ]] || { printf '%s\n' 'No recovery checkpoint; no training started.'; exit 1; }
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
exec "$PYTHON_BIN" -m dpbench.train_dissipative \
  --data ../data/passive_family_v1/seen.h5 \
  --config configs/dissipative_passive_family.json \
  --resume ../runs/seed0_47541/last.pt \
  --output "../runs/seed0_resume_${SLURM_JOB_ID}" --device cuda --execute
