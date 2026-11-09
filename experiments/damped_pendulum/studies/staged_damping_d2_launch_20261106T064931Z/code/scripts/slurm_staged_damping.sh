#!/usr/bin/env bash
#SBATCH --job-name=lehami-d2-staged
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=lehami-d2-staged-%j.log
set -euo pipefail

# A uses the first GPU; after its exact step-20000 snapshot and readout are
# published, single/multi B workers run concurrently, one GPU each.
# Submit explicitly with sbatch; running/testing this repository does not submit.
cd "${PENDULUM_CODE_DIR:?Set the verified frozen experiment code root}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the project Python executable}"
DATA_A="${PENDULUM_DATA_A:?Set the smooth_v1 undamped.h5}"
RUN_ROOT="${PENDULUM_RUN_ROOT:?Set the staged experiment output directory}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
ARGS=(scripts/train_staged_damping.py --execute --device cuda --gpus 0,1
  --data-a "$DATA_A" --output "$RUN_ROOT"
  --damping-steps "${PENDULUM_DAMPING_STEPS:-20000}")
if [[ -n "${PENDULUM_RESUME_A:-}" ]]; then
  ARGS+=(--resume-a "$PENDULUM_RESUME_A")
fi
if [[ -n "${PENDULUM_DATA_SINGLE:-}" ]]; then
  ARGS+=(--data-single "$PENDULUM_DATA_SINGLE")
fi
if [[ -n "${PENDULUM_DATA_MULTI:-}" ]]; then
  ARGS+=(--data-multi "$PENDULUM_DATA_MULTI")
fi
exec "$PYTHON_BIN" "${ARGS[@]}"
