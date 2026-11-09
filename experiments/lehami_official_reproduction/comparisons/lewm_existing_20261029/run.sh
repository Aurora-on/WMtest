#!/usr/bin/env bash
set -euo pipefail
cd /home/weizhixin/ssd/lehami_official_reproduction_20261029/comparisons/lewm_existing_20261029
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export STABLEWM_HOME=/home/weizhixin/ssd/lehami_official_reproduction_20261029/data
date -u +%FT%TZ > started_utc.txt
printf '%s\n' "${SLURM_JOB_ID:?}" > job_id.txt
/home/weizhixin/ssd/lehami_official_reproduction_20261029/.venv/bin/python evaluate_existing.py prepare
/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python evaluate_existing.py lewm
/home/weizhixin/ssd/lehami_official_reproduction_20261029/.venv/bin/python evaluate_existing.py lehami
date -u +%FT%TZ > finished_utc.txt
printf 'PASS\n' > status.txt
