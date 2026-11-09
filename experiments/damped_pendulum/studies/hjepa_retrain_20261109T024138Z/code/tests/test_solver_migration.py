"""Solver-only migration keeps state and makes provenance/config changes explicit."""
import copy
import json
from pathlib import Path
import pytest
import torch
from test_dissipative_training import config, tiny_data, _tiny_execution_config, _assert_nested_equal
from dpbench.train_dissipative import execute_run
from scripts.migrate_solver_checkpoint import migrate, check_config_change, sha256

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('change',['tolerance','learning_rate','history'])
def test_reject_non_solver_change(change):
    old={'model':{},'training':{'steps':20}}
    from dpbench.dissipative_model import resolve_model_config
    saved={'train_config':old,'config':resolve_model_config({})}
    new=copy.deepcopy(old);new['model']['core']={'solver':'backtracking'}
    if change=='tolerance':new['model']['core']['residual_atol']=1e-3
    elif change=='learning_rate':new['training']['steps']=21
    else:new['model']['history']=7
    with pytest.raises(ValueError):check_config_change(saved,new)


def test_actual_migration_preserves_state_and_requires_revalidation(tiny_data,config,tmp_path):
    old=_tiny_execution_config(config,1)
    execute_run(tiny_data,tmp_path/'old',old,stop_after_step=1)
    checkpoint=tmp_path/'old/last.pt';before=sha256(checkpoint)
    target=copy.deepcopy(old)
    target['model']['core'].update(solver='backtracking',newton_iterations=12,max_bisections=4)
    cfgfile=tmp_path/'new.json';cfgfile.write_text(json.dumps(target))
    report=migrate(checkpoint,cfgfile,tiny_data,ROOT,tmp_path/'migration','cpu',False)
    assert not (tmp_path/'migration').exists() and report['solver_changes']
    migrate(checkpoint,cfgfile,tiny_data,ROOT,tmp_path/'migration','cpu',True)
    a=torch.load(checkpoint,weights_only=True);b=torch.load(tmp_path/'migration/last.pt',weights_only=True)
    for key in ('state_dict','optimizer','rng','sampler_next_step','validation_indices'):_assert_nested_equal(a[key],b[key])
    assert b['validation_status']=='not_run' and b['best_score'] is None and sha256(checkpoint)==before
    status=execute_run(tiny_data,tmp_path/'new_run',target,resume=tmp_path/'migration/last.pt')
    assert status['status']=='completed' and status['completed_steps']==3
    assert (tmp_path/'new_run/resume_validation.json').exists()
    end=torch.load(tmp_path/'new_run/last.pt',weights_only=True)
    assert end['solver_migration']['original_checkpoint_sha256']==before
