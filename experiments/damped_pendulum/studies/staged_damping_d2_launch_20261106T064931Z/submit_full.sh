#!/usr/bin/env bash
#SBATCH --job-name=lehami-d2-staged
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/pipeline-%j.log
set -euo pipefail
export PENDULUM_CODE_DIR=/home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/code
export PENDULUM_PYTHON=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
export PENDULUM_DATA_A=/home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5
export PENDULUM_RESUME_A=/home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/runs/lehami_4000_10000/last.pt
export PENDULUM_RUN_ROOT=/home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run
export PENDULUM_DAMPING_STEPS=20000
exec bash "$PENDULUM_CODE_DIR/scripts/slurm_staged_damping.sh"
