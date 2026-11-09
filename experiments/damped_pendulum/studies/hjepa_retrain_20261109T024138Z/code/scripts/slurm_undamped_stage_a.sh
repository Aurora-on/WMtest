#!/usr/bin/env bash
#SBATCH --job-name=undamped-stage-a-s0
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=undamped-stage-a-%j.log
set -euo pipefail

# One bounded segment only. Submit the next segment after reviewing its predecessor.
# The full 20,000-step LR/curriculum configuration is never shortened by this script.
cd "${PENDULUM_CODE_DIR:?Set the verified frozen code snapshot}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the Python executable}"
DATA_FILE="${PENDULUM_DATA:?Set the generated undamped.h5 path}"
RUN_ROOT="${PENDULUM_RUN_ROOT:?Set a new stage-A run root}"
STOP_STEP="${PENDULUM_STOP_AFTER:-250}"
RESUME_FILE="${PENDULUM_RESUME:-}"

case "$STOP_STEP" in
  250) PREVIOUS_STEP=0 ;;
  1000) PREVIOUS_STEP=250 ;;
  4000) PREVIOUS_STEP=1000 ;;
  10000) PREVIOUS_STEP=4000 ;;
  20000) PREVIOUS_STEP=10000 ;;
  *) echo "PENDULUM_STOP_AFTER must be 250, 1000, 4000, 10000, or 20000" >&2; exit 2 ;;
esac

if [[ "$PREVIOUS_STEP" == 0 ]]; then
  if [[ -n "$RESUME_FILE" ]]; then
    echo "The first stage-A segment must start fresh, without PENDULUM_RESUME" >&2
    exit 2
  fi
else
  if [[ -z "$RESUME_FILE" ]]; then
    echo "This segment requires PENDULUM_RESUME from milestone $PREVIOUS_STEP" >&2
    exit 2
  fi
  "$PYTHON_BIN" - "$RESUME_FILE" "$PREVIOUS_STEP" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if checkpoint.get("step") != int(sys.argv[2]) or checkpoint.get("validation_status") != "completed":
    raise SystemExit("Resume must be the preceding milestone's last.pt with completed validation")
PY
fi

export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
TRAIN_ARGS=(-m dpbench.train_dissipative \
  --config configs/dissipative_undamped_stage_a_v1.json \
  --data "$DATA_FILE" --output "$RUN_ROOT/milestone-$STOP_STEP" \
  --device cuda --stop-after-step "$STOP_STEP" --execute)
if [[ -n "$RESUME_FILE" ]]; then
  TRAIN_ARGS+=(--resume "$RESUME_FILE")
fi
exec "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
