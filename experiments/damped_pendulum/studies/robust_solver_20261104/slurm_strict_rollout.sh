#!/usr/bin/env bash
#SBATCH --job-name=robust-strict-4250
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/weizhixin/ssd/robust_solver_20261104_v2/strict-4250-%j.log
set -euo pipefail
RUN_ROOT=/home/weizhixin/ssd/robust_solver_20261104_v2
cd "$RUN_ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python "$RUN_ROOT/audits/audit_strict_robust_rollouts.py" \
 --checkpoint "$RUN_ROOT/continuation/last.pt" --config configs/dissipative_passive_robust.json \
 --data /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/data/passive_family_v1/seen.h5 \
 --output "$RUN_ROOT/strict_rollout_4250" --device cuda --batch-size 32 --steps 320 --expected-episodes 384 --execute
