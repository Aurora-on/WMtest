#!/usr/bin/env bash
#SBATCH --job-name=passive-efficient3-eval
#SBATCH --partition=compute
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=efficient3-eval-%j.log
set -euo pipefail

cd "${PENDULUM_CODE_DIR:?Set the frozen evaluation code directory}"
PYTHON_BIN="${PENDULUM_PYTHON:?Set the verified Python environment}"
DATA_ROOT="${PENDULUM_DATA_ROOT:?Set the generated passive-family directory}"
EVAL_ROOT="${PENDULUM_EVAL_ROOT:?Set the new evaluation experiment directory}"
CHECKPOINT="${PENDULUM_CHECKPOINT:?Set the preselected best checkpoint}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" scripts/bootstrap_upstream.py --verify
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os
from pathlib import Path

data = Path(os.environ['PENDULUM_DATA_ROOT'])
root = Path(os.environ['PENDULUM_EVAL_ROOT'])
code = Path(os.environ['PENDULUM_CODE_DIR'])
checkpoint = Path(os.environ['PENDULUM_CHECKPOINT'])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
manifest = json.loads((data / 'manifest.json').read_text())
dataset_hashes = {entry['file']: entry['sha256'] for entry in manifest['datasets']}
for name in ('seen.h5', 'unseen.h5'):
    assert sha(data / name) == dataset_hashes[name], name
assert sha(checkpoint) == os.environ['PENDULUM_CHECKPOINT_SHA256']
source_hashes = json.loads((root / 'source_hashes.json').read_text())
assert all(sha(code / name) == digest for name, digest in source_hashes.items())
(root / 'input_verification.json').write_text(json.dumps({
    'status': 'passed', 'checkpoint_sha256': sha(checkpoint),
    'dataset_sha256': dataset_hashes, 'source_hashes': source_hashes,
    'world_model_updates_enabled': False,
}, indent=2) + '\n')
PY

"$PYTHON_BIN" -m dpbench.evaluate_passive \
  --checkpoint "$CHECKPOINT" \
  --seen "$DATA_ROOT/seen.h5" --unseen "$DATA_ROOT/unseen.h5" \
  --output "$EVAL_ROOT/results" --device cuda --batch-size 32 \
  --labels 20000 --val-labels 4000 --probe-steps 2000 --threads 4 --seed 0 \
  --horizons 1 2 4 8 16 32

"$PYTHON_BIN" - <<'PY'
import hashlib, json, os
from pathlib import Path
checkpoint = Path(os.environ['PENDULUM_CHECKPOINT'])
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == os.environ['PENDULUM_CHECKPOINT_SHA256']
root = Path(os.environ['PENDULUM_EVAL_ROOT'])
result = json.loads((root / 'results/manifest.json').read_text())
assert result['status'] == 'completed' and result['episodes'] == 1408
(root / 'job_completion.json').write_text(json.dumps({
    'status': 'evaluation_completed', 'episodes': result['episodes'],
    'failed_rollouts': result['failed_rollouts'],
    'checkpoint_unchanged': True, 'world_model_updates': 0,
}, indent=2) + '\n')
PY
