#!/usr/bin/env bash
#SBATCH --job-name=undamped-a4000-visualize
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/animation-eval-%j.log
set -euo pipefail
RUN_ROOT=/home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
cd "$RUN_ROOT/code"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec "$PYTHON_BIN" "$RUN_ROOT/audits/evaluate_for_animation.py" \
  --code-root "$RUN_ROOT/code" --checkpoint "$RUN_ROOT/training/last.pt" \
  --data "$RUN_ROOT/data/undamped_v1/undamped.h5" \
  --output "$RUN_ROOT/animation_eval4000_v1" --device cuda
