#!/usr/bin/env bash
# Submit only after logging into your own account and checking site policy.
#SBATCH --job-name=pendulum-profile
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=profile-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
PYTHON_BIN="${PENDULUM_PYTHON:-.venv/bin/python}"
"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
nvidia-smi
for model in lehami lewm; do
  "$PYTHON_BIN" -m dpbench.train --data data/basic/pendulum_b0p2.h5 --model "$model" \
    --output "runs/server_profile_${SLURM_JOB_ID}_${model}" --device cuda \
    --steps 103 --warmup 3 --batch-size 32 --benchmark
done
for model in lehami lewm; do
  RUN_DIR="runs/server_check_${SLURM_JOB_ID}_${model}"
  "$PYTHON_BIN" -m dpbench.train --data data/basic/pendulum_b0p2.h5 --model "$model" \
    --output "$RUN_DIR" --device cuda --steps 10 --warmup 3 --batch-size 8 --val-windows 32
  "$PYTHON_BIN" -m dpbench.evaluate --data data/basic/pendulum_b0p2.h5 --checkpoint "$RUN_DIR/best.pt" \
    --output "$RUN_DIR/evaluation" --device cuda --labels 128 --val-labels 64 \
    --probe-steps 50 --max-episodes 4 --horizons 4 8 16 32
done
