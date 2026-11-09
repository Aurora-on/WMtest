#!/usr/bin/env bash
# Shared infrastructure checks only; no submission and no upstream patching.
set -euo pipefail

REPRO_ROOT=/home/weizhixin/ssd/lehami_official_reproduction_20261029
UPSTREAM="$REPRO_ROOT/LeHami"
export LEHAMI_PYTHON="$REPRO_ROOT/.venv/bin/python"
export UV="$REPRO_ROOT/setup/uv-nosync"
export STABLEWM_HOME="$REPRO_ROOT/data"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
unset TRAIN_OVERRIDES
MANIFEST="$UPSTREAM/lehami/data/split_pendulum_multi_v2_episode_srnn_K80_val.15_dt2_seed42.json"

initialize_job() {
    local phase="$1"
    : "${SLURM_JOB_ID:?Run this script as an explicitly submitted Slurm job}"
    [[ -x "$LEHAMI_PYTHON" && -x "$UV" ]] || {
        echo "Isolated Python environment or uv-nosync wrapper is missing" >&2; exit 2;
    }
    RUN_META="$REPRO_ROOT/metadata/${phase}_${SLURM_JOB_ID}"
    export RUN_META JOB_PHASE="$phase"
    mkdir -p "$REPRO_ROOT/metadata"
    mkdir "$RUN_META"  # Existing metadata means this job id was already used.
    "$LEHAMI_PYTHON" - <<'PY'
import json, os, time
from pathlib import Path
p = Path(os.environ['RUN_META']) / 'job.json'
p.write_text(json.dumps({'phase':os.environ['JOB_PHASE'], 'job_id':os.environ['SLURM_JOB_ID'],
    'status':'RUNNING', 'started_unix':time.time(),
    'slurm':{k:v for k,v in os.environ.items() if k in ['SLURM_JOB_ID','SLURM_JOB_NODELIST',
        'SLURM_CPUS_PER_TASK','SLURM_JOB_PARTITION','CUDA_VISIBLE_DEVICES']}}, indent=2)+'\n')
PY
    trap finish_job EXIT
    cd "$UPSTREAM"
}

finish_job() {
    local code=$?
    trap - EXIT
    "$LEHAMI_PYTHON" - "$RUN_META/job.json" "$code" <<'PY'
import json, sys, time
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text()); code=int(sys.argv[2])
d.update(exit_code=code, status='PASS' if code==0 else 'FAILED', finished_unix=time.time())
d['wall_seconds']=d['finished_unix']-d['started_unix']
p.write_text(json.dumps(d,indent=2)+'\n')
PY
    exit "$code"
}

time_command() {
    local name="$1"; shift
    echo "[$(date -u +%FT%TZ)] $name"
    if /usr/bin/time -v -o "$RUN_META/$name.time.txt" "$@" > "$RUN_META/$name.log" 2>&1; then
        echo "[$(date -u +%FT%TZ)] $name finished"
    else
        local code=$?
        echo "$name failed (exit $code); full log: $RUN_META/$name.log" >&2
        tail -n 40 "$RUN_META/$name.log" >&2
        return "$code"
    fi
}

verify_runtime() {
    "$LEHAMI_PYTHON" - "$REPRO_ROOT" "$RUN_META" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, out=map(Path,sys.argv[1:]); source=root/'LeHami'
m=json.loads((root/'metadata/source_manifest.json').read_text())
assert m['lehami_commit']=='f76f65f8b99afb6b7e3063f573645d3e2f290ede'
assert m['lewm_commit']=='8edfeb336732b5f3ce7b8b210d0ba370a09e2cac'
files={}
for entry in m['official_lehami_files']:
    rel=entry['path']
    if not (rel.startswith('lehami/') or '/' not in rel): continue
    data=(source/rel).read_bytes()
    actual=hashlib.sha256(data).hexdigest()
    if actual!=entry['sha256']: raise RuntimeError(f'Official source changed: {rel}')
    files[rel]=actual
lewm={'jepa.py':'41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96',
      'module.py':'0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620'}
for name, expected in lewm.items():
    rel='lehami/models/third_party/leworldmodel/'+name
    actual=hashlib.sha256((source/rel).read_bytes()).hexdigest()
    if actual!=expected: raise RuntimeError(f'Pinned LeWM source changed: {rel}')
    files[rel]=actual
fingerprint=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
record={'status':'PASS','lehami_commit':m['lehami_commit'],'lewm_commit':m['lewm_commit'],
        'runtime_source_fingerprint':fingerprint,'verified_files':files,
        'compatibility_manifest_sha256':hashlib.sha256((root/'setup/generated/compatibility_manifest.json').read_bytes()).hexdigest()}
(out/'sources.json').write_text(json.dumps(record,indent=2)+'\n')
PY
    time_command environment "$LEHAMI_PYTHON" "$REPRO_ROOT/setup/verify_environment.py" \
        --manifest "$REPRO_ROOT/setup/generated/compatibility_manifest.json" \
        --output "$RUN_META/environment.json" --cuda
    "$LEHAMI_PYTHON" - <<'PY'
import torch
assert torch.cuda.device_count()==1, 'Official SIGReg recipe requires exactly one visible GPU'
assert 'L40' in torch.cuda.get_device_name(), torch.cuda.get_device_name()
PY
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RUN_META/gpu.csv"
}

audit_official_data() {
    if "$LEHAMI_PYTHON" "$REPRO_ROOT/validation/audit_data.py" \
        --data "$REPRO_ROOT/data/pendulum_multi_v2.h5" --manifest "$MANIFEST" \
        --output "$RUN_META/data_audit.json" > "$RUN_META/data_audit.log" 2>&1; then
        echo "Official dataset semantic and historical byte checks passed"
    else
        local code=$?
        echo "Dataset audit stopped the job (exit $code). Generated files are retained." >&2
        echo "Inspect $RUN_META/data_audit.json; no manifest or dataset was rewritten." >&2
        tail -n 30 "$RUN_META/data_audit.log" >&2
        return "$code"
    fi
}
