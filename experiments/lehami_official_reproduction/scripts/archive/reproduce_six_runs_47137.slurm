#!/usr/bin/env bash
# Submit only after reviewing the completed preflight. Never submits other jobs.
#SBATCH --job-name=lehami-original
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/weizhixin/ssd/lehami_official_reproduction_20261029
#SBATCH --output=/home/weizhixin/ssd/lehami_official_reproduction_20261029/reproduce-%j.log

set -euo pipefail
source /home/weizhixin/ssd/lehami_official_reproduction_20261029/scripts/job_common.sh
initialize_job reproduce
verify_runtime
audit_official_data

"$LEHAMI_PYTHON" - "$REPRO_ROOT" "$RUN_META" <<'PY'
import hashlib, json, sys
from pathlib import Path
root,out=map(Path,sys.argv[1:]); upstream=root/'LeHami'
previous=json.loads((root/'metadata/preflight_pass.json').read_text())
assert previous['status']=='PASS'
source=json.loads((out/'sources.json').read_text()); env=json.loads((out/'environment.json').read_text())
audit=json.loads((out/'data_audit.json').read_text())
for key in ['runtime_source_fingerprint','compatibility_manifest_sha256']:
    assert previous[key]==source[key], f'Preflight/source changed: {key}'
fingerprint=hashlib.sha256(json.dumps({'versions':env['versions'],'python':env['python'],
    'cuda':env['torch_cuda_runtime']},sort_keys=True).encode()).hexdigest()
assert previous['environment_fingerprint']==fingerprint, 'Environment changed since preflight'
assert previous['dataset_sha256']==audit['dataset_sha256'], 'Dataset changed since preflight'
# Both checkpoints AND probe caches must start empty. Nothing is deleted here.
for arm, model, suffix, result in [
    ('a0m','lehami_sg_sep_sreg_joint_ms_conv','_sg_sep_conv_w1_multi_exp067_a0m','exp067_a0m_lehami_w1_multi'),
    ('b0m','lewm_ms_conv','_ms_conv_w1_multi_exp068_b0m','exp068_b0m_lewm_w1_multi')]:
    for tag in ['', '_s1', '_s2']:
        ckpt=root/'data/checkpoints'/f'{model}_pendulum_multi_v2{suffix}{tag}'
        results=upstream/'lehami/experiments/results'/f'{result}{tag}'
        for p in [ckpt,results]:
            if p.exists() and any(p.iterdir()): raise RuntimeError(f'Refusing existing formal output: {p}')
p=upstream/'lehami/experiments/results/latent_volume'
if p.exists() and any(p.iterdir()): raise RuntimeError(f'Refusing existing Jacobian outputs: {p}')
(out/'accepted_preflight.json').write_text(json.dumps(previous,indent=2)+'\n')
(out/'recipe.json').write_text(json.dumps({'entrypoint':'lehami/experiments/conservative_multi_rod.sh',
    'arguments':['--arms','a0m,b0m'],'train_overrides':None,'seeds':[42,1,2],
    'epochs_per_run':100,'expected_final_updates':{'a0m':43900,'b0m':45900},
    'evaluations':'Unchanged official runner including probes, dt/reversal and Jacobian diagnostics'},indent=2)+'\n')
PY

# No TRAIN_OVERRIDES, altered readout, reduced evaluation, or changed seed labels.
# Omitting --seeds preserves the author's unsuffixed seed42 directories and the
# extra diagnostics that are conditioned on that config-seed run.
time_command official_runner srun --ntasks=1 bash lehami/experiments/conservative_multi_rod.sh --arms a0m,b0m
echo "Original six-run script finished. Inspect logs/results before claiming reproduction: $RUN_META"
