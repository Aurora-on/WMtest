#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=loss-lehami-s0-4k
#SBATCH --output=/home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/lehami-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/code
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python scripts/bootstrap_upstream.py --verify
/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python scripts/qualify_loss_pair.py --config configs/dissipative_loss_pair_lehami_v1.json --data /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 --output /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/lehami_cuda_qualification.json
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python scripts/prepare_lehami_loss_pair.py --arm lehami --python /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python --data /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 --output-root /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/runs --device cuda --stop-after-step 4000 --execute
