#!/usr/bin/env bash
#SBATCH --job-name=stall-renderer-audit
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --output=/home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/renderer-initializer-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python \
 studies/stall_diagnosis_20261105/check_renderer_initializer.py \
 --device cuda \
 --checkpoint /home/weizhixin/ssd/efficient3_fresh_20261104T101014Z/training/best.pt \
 --teacher-root /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/teacher_diagnostic \
 --windows /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/stall_rest_windows.npz \
 --output /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/renderer_initializer
