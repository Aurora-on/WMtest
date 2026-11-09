#!/usr/bin/env bash
#SBATCH --job-name=efficient-solver-short
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/home/weizhixin/ssd/efficient_solver_20261104_v1/short-%j.log
set -euo pipefail
RUN_ROOT=/home/weizhixin/ssd/efficient_solver_20261104_v1
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
cd "$RUN_ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
EFFICIENT_TEST_DEVICE=cuda "$PYTHON_BIN" -m pytest tests/test_efficient_dissipative_core.py -q > "$RUN_ROOT/cuda_tests.log"
"$PYTHON_BIN" scripts/benchmark_efficient_solver.py \
  --checkpoint /home/weizhixin/ssd/robust_solver_20261104_v2/continuation/last.pt \
  --data /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/data/passive_family_v1/seen.h5 \
  --efficient-config configs/dissipative_passive_efficient.json \
  --original-source-root /home/weizhixin/ssd/robust_solver_20261104_v2/code \
  --output "$RUN_ROOT/benchmark" --device cuda --warmup 1 --timed 3 --execute
