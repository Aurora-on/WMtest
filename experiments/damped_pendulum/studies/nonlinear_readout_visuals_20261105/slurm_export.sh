#!/usr/bin/env bash
#SBATCH --job-name=nonlinear-readout-rollout
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --output=/home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/nonlinear_readout_visuals_20261105/rollout-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/nonlinear_readout_visuals_20261105/export_lehami_rollouts.py \
 --code-root /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/code \
 --code-manifest /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/control_visuals_20261105/code_manifest.json \
 --helper-dir /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/control_visuals_20261105 \
 --probe-summary /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/control_visuals_20261105/readout_diagnostic/summary.json \
 --checkpoint /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/runs/lehami_4000_10000/last.pt \
 --data /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 \
 --output /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/nonlinear_readout_visuals_20261105/rollout --device cuda
