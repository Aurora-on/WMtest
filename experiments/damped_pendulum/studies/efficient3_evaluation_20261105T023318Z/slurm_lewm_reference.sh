#!/usr/bin/env bash
#SBATCH --job-name=lewm-historical-reference
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/lewm-%j.log
set -euo pipefail
cd /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/code
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python \
 scripts/diagnose_teacher_forced_readout.py \
 --checkpoint /home/weizhixin/ssd/efficient3_fresh_20261104T101014Z/training/best.pt \
 --readouts /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/results/readouts.pt \
 --seen /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/data/passive_family_v1/seen.h5 \
 --unseen /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/data/passive_family_v1/unseen.h5 \
 --eval-results /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/results \
 --output /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/teacher_diagnostic --device cuda
exec /home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python \
 scripts/evaluate_historical_lewm_reference.py \
 --legacy-root /home/weizhixin/ssd/damped_pendulum_profile_20261028/studies/epoch35_20261028T133556Z \
 --frozen-manifest studies/independent_test_20261029T070556Z/frozen_manifest.json \
 --seen /home/weizhixin/ssd/dissipative_passive_20261103T051548Z/data/passive_family_v1/seen.h5 \
 --seen-sha256 55d2b4687e1a6f743b123ba69e3731a067a5b941817bd9d98738262dfabb1509 \
 --output /home/weizhixin/ssd/efficient3_evaluation_20261105T023318Z/lewm_reference \
 --device cuda --batch-size 32 --threads 4 --execute
