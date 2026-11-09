#!/usr/bin/env bash
#SBATCH --job-name=comp-lewm
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/weizhixin/ssd/official_comparison_launch_20261108T092500Z/lewm-%j.log
set -euo pipefail
ROOT=/home/weizhixin/ssd/official_comparison_launch_20261108T092500Z
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
cd "$ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$PYTHON_BIN" scripts/profile_comparison_cuda.py --method lewm \
 --data-a /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 \
 --data-single /home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run/data_single/damping.h5 \
 --data-multi /home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run/data_multi/seen.h5 \
 --output "$ROOT/profiles/lewm_${SLURM_JOB_ID}.json" --all-stage-horizons
exec "$PYTHON_BIN" scripts/run_comparison.py --execute --device cuda --gpus 0 \
 --methods lewm --seeds 0 --route staged \
 --data-a /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 \
 --data-single /home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run/data_single/damping.h5 \
 --data-multi /home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run/data_multi/seen.h5 \
 --output "$ROOT/runs/lewm"
