#!/usr/bin/env bash
#SBATCH --job-name=hjepa-retrain-single
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/weizhixin/ssd/hjepa_retrain_20261109T024138Z/logs/single-%j.log
set -euo pipefail
ROOT=/home/weizhixin/ssd/hjepa_retrain_20261109T024138Z
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$ROOT"
"$PYTHON_BIN" verify_bundle.py
cd "$ROOT/code"
CONFIG=configs/hjepa_retrain_20261109/single.json
DATA=/home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run/data_single/damping.h5
"$PYTHON_BIN" scripts/profile_hjepa_retrain.py --config "$CONFIG" --data "$DATA" \
  --output "$ROOT/profiles/single_${SLURM_JOB_ID}.json"
exec "$PYTHON_BIN" scripts/run_hjepa_retrain.py --config "$CONFIG" --data "$DATA" \
  --data-a /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 --output "$ROOT/runs/single" \
  --device cuda --until-step "${HJEPA_UNTIL_STEP:-20000}" --execute
