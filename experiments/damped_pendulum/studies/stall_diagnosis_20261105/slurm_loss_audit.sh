#!/usr/bin/env bash
#SBATCH --job-name=stall-loss-audit
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/stall-loss-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/code
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python \
 /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/audit_loss_gradients.py \
 --checkpoint /home/weizhixin/ssd/efficient3_fresh_20261104T101014Z/training/best.pt \
 --data /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/data/passive_family_v1/seen.h5 \
 --output /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/stall_loss_gradient_audit.json
