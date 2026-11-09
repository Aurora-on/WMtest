#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --job-name=undamped-a-data
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data-%j.log
set -euo pipefail
RUN_ROOT=/home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
cd "$RUN_ROOT/code"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
/usr/bin/time -v "$PYTHON_BIN" -m dpbench.passive_family   --config configs/passive_family_smooth_v1.json --undamped-only   --output "$RUN_ROOT/data/undamped_v1" --generate > "$RUN_ROOT/data_generation.json"
"$PYTHON_BIN" "$RUN_ROOT/audits/audit_inputs.py"   --run-root "$RUN_ROOT" > "$RUN_ROOT/input_audit.json"
