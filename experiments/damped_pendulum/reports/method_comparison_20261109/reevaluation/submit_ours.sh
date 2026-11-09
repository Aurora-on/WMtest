#!/bin/bash
#SBATCH --job-name=ours-matched-readout
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/weizhixin/ssd/official_comparison_launch_20261108T092500Z/ours-readout-%j.log
set -euo pipefail
ROOT=/home/weizhixin/ssd/official_comparison_launch_20261108T092500Z
OLD=/home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
cd "$ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
"$PYTHON_BIN" scripts/evaluate_comparison.py --checkpoint "$OLD/milestones/undamped_step20000/last.pt" --data /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 --output "$ROOT/ours_matched_20261109/a/evaluation" --device cuda --horizons 8 16 32
"$PYTHON_BIN" scripts/evaluate_comparison.py --checkpoint "$OLD/milestones/damping_single_step20000/last.pt" --data "$OLD/data_single/damping.h5" --output "$ROOT/ours_matched_20261109/single/evaluation" --device cuda --horizons 8 16 32
"$PYTHON_BIN" scripts/evaluate_comparison.py --checkpoint "$OLD/milestones/damping_multi_step20000/last.pt" --data "$OLD/data_multi/seen.h5" --output "$ROOT/ours_matched_20261109/multi/evaluation" --device cuda --horizons 8 16 32
