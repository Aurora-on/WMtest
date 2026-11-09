#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
"$PYTHON_BIN" verify_bundle.py
mkdir -p logs
for condition in single multi; do
  job_id=$(sbatch --parsable "submit_${condition}.sh")
  printf '%s %s\n' "$condition" "$job_id" | tee -a submissions.log
done
