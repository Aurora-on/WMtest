#!/usr/bin/env bash
#SBATCH --job-name=lehami-preflight
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/weizhixin/ssd/lehami_official_reproduction_20261029
#SBATCH --output=/home/weizhixin/ssd/lehami_official_reproduction_20261029/preflight-%j.log

set -euo pipefail
source /home/weizhixin/ssd/lehami_official_reproduction_20261029/scripts/job_common.sh
initialize_job preflight
verify_runtime

# Execute the unchanged author generator recipe. Existing files are retained by
# the author script; strict auditing below prevents accepting a different file.
time_command generate_data bash lehami/data/prepare_conservative_data.sh
audit_official_data  # Exit 3 (bytes differ) deliberately stops before training.

export STABLEWM_HOME="$REPRO_ROOT/smoke/data"
mkdir -p "$STABLEWM_HOME"
"$LEHAMI_PYTHON" - "$REPRO_ROOT" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ['pendulum.h5','pendulum_multi_v2.h5']:
    source=root/'data'/name; dest=root/'smoke/data'/name
    if dest.exists() or dest.is_symlink():
        if dest.resolve()!=source.resolve(): raise RuntimeError(f'Unexpected smoke data: {dest}')
    else:
        dest.symlink_to(source)
PY

ARMS=(a0m b0m)
CONFIGS=(phase2_lehami_sg_sep_conv_w1_multi_exp067_a0m phase0_lewm_ms_conv_w1_multi_exp068_b0m)
MODELS=(lehami_sg_sep_sreg_joint_ms_conv lewm_ms_conv)
for i in 0 1; do
    arm="${ARMS[$i]}"
    suffix="_preflight_${arm}_${SLURM_JOB_ID}"
    ckpt_dir="$STABLEWM_HOME/checkpoints/${MODELS[$i]}_pendulum_multi_v2${suffix}"
    [[ ! -e "$ckpt_dir" ]] || { echo "Refusing to overwrite $ckpt_dir" >&2; exit 2; }
    # A complete epoch, including the author's sanity and full validation passes.
    # Only duration and output locations change; this is not a formal result.
    time_command "train_$arm" srun --ntasks=1 "$UV" run python lehami/training/train.py \
        "+experiment=${CONFIGS[$i]}" "manifest=$MANIFEST" \
        trainer.max_epochs=1 "run_suffix=$suffix" \
        "hydra.run.dir=$REPRO_ROOT/smoke/hydra_${SLURM_JOB_ID}_$arm"
    "$LEHAMI_PYTHON" - "$ckpt_dir" "$arm" "$RUN_META" <<'PY'
import json, sys, torch
from pathlib import Path
from omegaconf import OmegaConf
directory, arm, out=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
expected={'a0m':439,'b0m':459}[arm]
last=torch.load(directory/'last.ckpt',map_location='cpu',weights_only=False)
assert last['global_step']==expected, (arm,last['global_step'],expected)
assert last['epoch']==0, last['epoch']
assert (directory/'best.ckpt').is_file()
cfg=OmegaConf.to_container(OmegaConf.load(directory/'config.yaml'),resolve=True)
assert cfg['seed']==42 and cfg['trainer']['max_epochs']==1
assert cfg['trainer']['precision']=='bf16-mixed'
assert cfg['loader']['batch_size']==32
schedulers=last.get('lr_schedulers',[])
assert len(schedulers)==1 and schedulers[0]['last_epoch']==expected, schedulers
assert any('validate/pred_loss' in str(key) for key in last.get('callbacks',{}))
record={'arm':arm,'status':'PASS','checkpoint':str(directory/'last.ckpt'),
        'global_step':last['global_step'],'epoch':last['epoch'],'scheduler':schedulers[0],
        'seed':cfg['seed'],'precision':cfg['trainer']['precision'],
        'note':'One complete epoch; its scheduler horizon is intentionally one epoch. Not a scientific result.'}
(out/f'smoke_{arm}.json').write_text(json.dumps(record,indent=2)+'\n')
PY
    # Exercise all six original readouts with a short rollout. Calibration uses
    # the full original training split; these preflight scores are not results.
    smoke_name="preflight_${SLURM_JOB_ID}_${arm}"
    [[ ! -e "lehami/experiments/results/$smoke_name" ]] || exit 2
    time_command "eval_$arm" srun --ntasks=1 "$UV" run python lehami/experiments/src/run_eval.py \
        --checkpoint "$ckpt_dir/best.ckpt" --manifest "$MANIFEST" \
        --exp-name "$smoke_name" --horizon 10 --rollout-seed 0 --device cuda
done

"$LEHAMI_PYTHON" - "$REPRO_ROOT" "$RUN_META" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path
root,out=map(Path,sys.argv[1:])
source=json.loads((out/'sources.json').read_text())
env=json.loads((out/'environment.json').read_text())
audit=json.loads((out/'data_audit.json').read_text())
for arm in ['a0m','b0m']: assert json.loads((out/f'smoke_{arm}.json').read_text())['status']=='PASS'
fingerprint=hashlib.sha256(json.dumps({'versions':env['versions'],'python':env['python'],
    'cuda':env['torch_cuda_runtime']},sort_keys=True).encode()).hexdigest()
record={'status':'PASS','job_id':os.environ['SLURM_JOB_ID'],'metadata_dir':str(out),
        'completed_unix':time.time(),'runtime_source_fingerprint':source['runtime_source_fingerprint'],
        'compatibility_manifest_sha256':source['compatibility_manifest_sha256'],
        'environment_fingerprint':fingerprint,'dataset_sha256':audit['dataset_sha256'],
        'smoke_updates':{'a0m':439,'b0m':459}}
(out/'preflight_pass.json').write_text(json.dumps(record,indent=2)+'\n')
temporary=root/'metadata'/f'.preflight_pass_{os.environ["SLURM_JOB_ID"]}.json'
temporary.write_text(json.dumps(record,indent=2)+'\n')
temporary.replace(root/'metadata/preflight_pass.json')
PY
echo "Preflight passed. No formal job was submitted. Timings and evidence: $RUN_META"
