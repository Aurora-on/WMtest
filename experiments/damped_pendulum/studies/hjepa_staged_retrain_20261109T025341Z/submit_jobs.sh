#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN=/home/weizhixin/ssd/damped_pendulum_profile_20261028/.venv/bin/python
"$PYTHON_BIN" verify_bundle.py
case "${HJEPA_UNTIL_STEP:-20000}" in
  4000|20000) ;;
  *) echo 'HJEPA_UNTIL_STEP must be 4000 or 20000' >&2; exit 2 ;;
esac
mkdir -p logs
a_id=$(sbatch --parsable submit_a.sh)
printf 'a %s\n' "$a_id" | tee -a submissions.log
if [[ "${HJEPA_UNTIL_STEP:-20000}" == 20000 ]]; then
  for condition in single multi; do
    job_id=$(sbatch --parsable --dependency="afterok:${a_id%%;*}" --kill-on-invalid-dep=yes "submit_${condition}.sh")
    printf '%s %s parent=%s\n' "$condition" "$job_id" "$a_id" | tee -a submissions.log
  done
fi
