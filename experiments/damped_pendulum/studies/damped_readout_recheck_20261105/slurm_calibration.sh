#!/usr/bin/env bash
#SBATCH --job-name=damped-readout-calibration
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/damped_readout_recheck_20261105/calibration-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/damped_readout_recheck_20261105/export_calibration.py \
 --code-root /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/code \
 --checkpoint /home/weizhixin/ssd/efficient3_fresh_20261104T101014Z/training/best.pt \
 --seen /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/data/passive_family_v1/seen.h5 \
 --eval-results /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/results \
 --readouts /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/results/readouts.pt \
 --output /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/damped_readout_recheck_20261105/calibration --device cuda
