#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/weizhixin/ssd/official_comparison_launch_20261108T092500Z
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
cd "$ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
for METHOD in lewm hjepa; do
 "$PYTHON_BIN" scripts/profile_comparison_cuda.py --method "$METHOD" \
 --data-a /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 \
 --data-single /home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run/data_single/damping.h5 \
 --data-multi /home/weizhixin/ssd/staged_damping_d2_launch_20261106T064931Z/run/data_multi/seen.h5 \
 --output "$ROOT/profiles/${METHOD}_probe_in_48044.json" --all-stage-horizons
done
