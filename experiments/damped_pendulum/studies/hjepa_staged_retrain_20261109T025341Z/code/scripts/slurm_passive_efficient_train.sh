#!/usr/bin/env bash
#SBATCH --job-name=passive-efficient3-s0
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=efficient3-train-%j.log
set -euo pipefail

cd "${PENDULUM_CODE_DIR:?Set the frozen code snapshot directory}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the verified Python environment}"
DATA_FILE="${PENDULUM_DATA:?Set the seen train/validation HDF5 path}"
RUN_ROOT="${PENDULUM_RUN_ROOT:?Set a fresh experiment directory}"
SOURCE_CHECKPOINT="${PENDULUM_SOURCE_CHECKPOINT:?Set the original robust checkpoint}"
SOURCE_CODE="${PENDULUM_SOURCE_CODE:?Set its original frozen source directory}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
"$PYTHON_BIN" scripts/migrate_efficient_checkpoint.py \
  --checkpoint "$SOURCE_CHECKPOINT" \
  --config configs/dissipative_passive_efficient_3substeps.json \
  --data "$DATA_FILE" --original-source-root "$SOURCE_CODE" \
  --output "$RUN_ROOT/migration" --device cuda --write \
  > "$RUN_ROOT/migration.log"

# The training entry point revalidates the migrated step before any new update.
# This is a new algorithm branch with the original full-budget schedule.
# No physical test evaluation or automatic retry is started by this job.
exec "$PYTHON_BIN" -m dpbench.train_dissipative \
  --data "$DATA_FILE" \
  --config configs/dissipative_passive_efficient_3substeps.json \
  --output "$RUN_ROOT/training" --device cuda \
  --resume "$RUN_ROOT/migration/last.pt" --execute
