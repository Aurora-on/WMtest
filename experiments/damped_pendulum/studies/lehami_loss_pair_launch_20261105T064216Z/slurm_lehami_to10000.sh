#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --job-name=loss-lehami-to10k
#SBATCH --dependency=afterok:47801
#SBATCH --output=/home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/lehami-to10k-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/code
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/guard_resume.py /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/runs/lehami/last.pt
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python -m dpbench.train_dissipative --config configs/dissipative_loss_pair_lehami_v1.json --data /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 --output /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/runs/lehami_4000_10000 --device cuda --resume /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/runs/lehami/last.pt --stop-after-step 10000 --execute
