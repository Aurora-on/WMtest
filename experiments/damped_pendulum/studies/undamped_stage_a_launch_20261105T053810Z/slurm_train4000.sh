#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --job-name=undamped-a-fresh4000-s0
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/train-%j.log
set -euo pipefail
RUN_ROOT=/home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
cd "$RUN_ROOT/code"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
"$PYTHON_BIN" scripts/qualify_undamped_stage_a.py   --config configs/dissipative_undamped_stage_a_v1.json   --data "$RUN_ROOT/data/undamped_v1/undamped.h5"   --output "$RUN_ROOT/cuda_qualification.json"
exec "$PYTHON_BIN" -m dpbench.train_dissipative   --config configs/dissipative_undamped_stage_a_v1.json   --data "$RUN_ROOT/data/undamped_v1/undamped.h5"   --output "$RUN_ROOT/training" --device cuda --stop-after-step 4000 --execute
