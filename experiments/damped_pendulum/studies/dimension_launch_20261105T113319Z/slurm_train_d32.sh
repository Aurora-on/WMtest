#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=lehami-d32-10k
#SBATCH --output=/home/weizhixin/ssd/dimension_launch_20261105T113319Z/train-d32-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/dimension_launch_20261105T113319Z/code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/home/weizhixin/ssd/dimension_launch_20261105T113319Z/code
export PENDULUM_CODE_DIR=/home/weizhixin/ssd/dimension_launch_20261105T113319Z/code
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python /home/weizhixin/ssd/dimension_launch_20261105T113319Z/run_training.py --manifest /home/weizhixin/ssd/dimension_launch_20261105T113319Z/launch_manifest.json --dimension 32
