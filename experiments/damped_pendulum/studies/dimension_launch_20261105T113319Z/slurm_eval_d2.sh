#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=dim-d2-knn-val
#SBATCH --output=/home/weizhixin/ssd/dimension_launch_20261105T113319Z/eval-d2-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/dimension_launch_20261105T113319Z/code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=/home/weizhixin/ssd/dimension_launch_20261105T113319Z/code
/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python -c 'import hashlib,json,sys; from pathlib import Path; m=json.loads(Path(sys.argv[1]).read_text()); assert hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()==m['"'"'evaluator_sha256'"'"']' /home/weizhixin/ssd/dimension_launch_20261105T113319Z/evaluation_code_manifest.json /home/weizhixin/ssd/dimension_launch_20261105T113319Z/evaluate_dimension.py
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python /home/weizhixin/ssd/dimension_launch_20261105T113319Z/evaluate_dimension.py --code-root /home/weizhixin/ssd/dimension_launch_20261105T113319Z/code --code-manifest /home/weizhixin/ssd/dimension_launch_20261105T113319Z/evaluation_code_manifest.json --checkpoint /home/weizhixin/ssd/lehami_loss_pair_launch_20261105T064216Z/runs/lehami_4000_10000/last.pt --data /home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5 --output /home/weizhixin/ssd/dimension_launch_20261105T113319Z/evaluation/d2 --device cuda --expected-q-dim 2 --split val --horizon 32
