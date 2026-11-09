#!/usr/bin/env bash
#SBATCH --job-name=lehami-1seed
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --chdir=/home/weizhixin/ssd/lehami_official_reproduction_20261029
#SBATCH --output=/home/weizhixin/ssd/lehami_official_reproduction_20261029/lehami-one-seed-%j.log
set -euo pipefail
source /home/weizhixin/ssd/lehami_official_reproduction_20261029/scripts/job_common.sh
initialize_job lehami_one_seed
verify_runtime
audit_official_data
RESUME_CHECKPOINT="$($LEHAMI_PYTHON - "$REPRO_ROOT" "$RUN_META" <<'PY'
import json,sys
from pathlib import Path
r,out=map(Path,sys.argv[1:])
d=json.loads((r/'metadata/scope_change_lehami_one_seed.json').read_text())
assert d['models']==['a0m'] and d['seeds']==[42] and d['total_epochs']==100
assert Path(d['resume_checkpoint']).is_file()
(out/'recipe.json').write_text(json.dumps(d,indent=2)+'\n')
print(d['resume_checkpoint'])
PY
)"
# Official resume entry point restores optimizer, scheduler and training loop.
# max_epochs remains the original absolute total of 100.
time_command resume_train srun --ntasks=1 "$UV" run python lehami/training/train.py \
    +experiment=phase2_lehami_sg_sep_conv_w1_multi_exp067_a0m \
    "manifest=$MANIFEST" "resume_from=$RESUME_CHECKPOINT"
time_command official_evaluation srun --ntasks=1 bash \
    lehami/experiments/conservative_multi_rod.sh --arms a0m --seeds default --eval-only
# The official runner gates its joint Jacobian comparison on both model
# checkpoints. Retain the same diagnostics for the requested LeHami arm alone.
for mode in anchor rollout epsball; do
    case "$mode" in
        anchor) extra=() ;;
        rollout) extra=(--horizon 70) ;;
        epsball) extra=(--eps-ball --max-episodes 10 --eps 1e-2,1e-3,1e-4) ;;
    esac
    time_command "jacobian_$mode" srun --ntasks=1 "$UV" run python \
        lehami/experiments/src/eval_jacobian_volume.py --arms a0m "${extra[@]}"
done
echo "LeHami seed42 training and original diagnostics finished: $RUN_META"
