#!/usr/bin/env bash
#SBATCH --job-name=lehami-data
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/weizhixin/ssd/lehami_official_reproduction_20261029
#SBATCH --output=/home/weizhixin/ssd/lehami_official_reproduction_20261029/data-%j.log
set -euo pipefail
REPRO_ROOT=/home/weizhixin/ssd/lehami_official_reproduction_20261029
export LEHAMI_PYTHON="$REPRO_ROOT/data-env/bin/python"
export UV="$REPRO_ROOT/setup/uv-nosync"
export STABLEWM_HOME="$REPRO_ROOT/data"
export PYTHONUNBUFFERED=1
mkdir -p "$REPRO_ROOT/metadata" "$STABLEWM_HOME"
"$LEHAMI_PYTHON" - <<'PY'
import importlib.metadata as m
for p,v in {'numpy':'2.4.6','pillow':'11.3.0','h5py':'3.16.0','hdf5plugin':'6.0.0'}.items():
    assert m.version(p)==v,(p,m.version(p),v)
print('Original data dependency versions verified')
PY
cd "$REPRO_ROOT/LeHami"
/usr/bin/time -v bash lehami/data/prepare_conservative_data.sh
"$LEHAMI_PYTHON" "$REPRO_ROOT/validation/audit_data.py" \
    --data "$STABLEWM_HOME/pendulum_multi_v2.h5" \
    --manifest lehami/data/split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json \
    --output "$REPRO_ROOT/metadata/generated_data_audit_${SLURM_JOB_ID}.json"
