"""Auditable algorithm migration, then actual revalidation and Adam continuation."""
import copy
import json
from pathlib import Path

import pytest
import torch

from test_dissipative_training import config, tiny_data, _tiny_execution_config, _assert_nested_equal
from dpbench.dissipative_model import resolve_model_config
from dpbench.train_dissipative import execute_run, source_hashes
from scripts.migrate_efficient_checkpoint import (
    ADDED_SOURCES, BASE_SOURCES, check_config_change, migrate, sha256, verify_source_change,
)

ROOT = Path(__file__).resolve().parents[1]


def target_config(old):
    target = copy.deepcopy(old)
    target['model']['core'].update(integrator='frozen_dg', solver='picard', substeps=3)
    return target


@pytest.fixture
def robust_config():
    return json.loads((ROOT / 'configs/dissipative_passive_robust.json').read_text())


@pytest.mark.parametrize('change', ['residual_atol', 'chain_atol', 'newton_iterations',
                                   'max_bisections', 'damping_mode', 'loss', 'schedule',
                                   'substeps', 'source_config'])
def test_reject_unapproved_change(robust_config, change):
    saved = {'train_config': robust_config, 'config': resolve_model_config(robust_config['model'])}
    target = target_config(robust_config)
    if change == 'loss':
        target['model']['motion_weight'] = 2.
    elif change == 'schedule':
        target['training']['dynamics_lr'] *= 2
    elif change == 'source_config':
        saved['config']['dt'] = .2
    else:
        target['model']['core'][change] = {
            'residual_atol': 1e-3, 'chain_atol': 1e-3, 'newton_iterations': 24,
            'max_bisections': 5, 'damping_mode': 'constant', 'substeps': 2,
        }[change]
    with pytest.raises(ValueError):
        check_config_change(saved, target)


def test_source_additions_recorded_and_unapproved_changes_rejected():
    current = source_hashes()
    old = {name: current[name] for name in BASE_SOURCES}
    verified, changes = verify_source_change(old, ROOT, current)
    assert verified == old and set(changes) == ADDED_SOURCES
    assert all(value['kind'] == 'added' and value['before'] is None for value in changes.values())
    modified = dict(current, **{'dpbench/dissipative_data.py': 'unapproved'})
    with pytest.raises(ValueError, match='Unapproved source change'):
        verify_source_change(old, ROOT, modified)
    with pytest.raises(ValueError, match='missing or unexpected'):
        verify_source_change(old, ROOT, {**current, 'dpbench/extra.py': 'unapproved'})
    with pytest.raises(ValueError, match='unsafe relative path'):
        verify_source_change({**old, '../escape.py': 'unapproved'}, ROOT, current)
    bad_old = dict(old, **{'dpbench/train_dissipative.py': 'wrong'})
    with pytest.raises(ValueError, match='does not match checkpoint'):
        verify_source_change(bad_old, ROOT, current)


def test_actual_migration_preserves_state_then_revalidates_and_updates(tiny_data, config, tmp_path):
    old = _tiny_execution_config(config, 1)
    old['model']['core'].update(solver='backtracking', substeps=2,
                                newton_iterations=12, max_bisections=4)
    execute_run(tiny_data, tmp_path / 'old', old, stop_after_step=1)
    checkpoint = tmp_path / 'old/last.pt'
    # Reproduce historical lineage and a pre-efficient source manifest.
    saved = torch.load(checkpoint, weights_only=True)
    saved['solver_migration'] = {'kind': 'explicit_solver_only_migration', 'step': 1}
    saved['source_hashes'] = {name: saved['source_hashes'][name] for name in BASE_SOURCES}
    torch.save(saved, checkpoint)
    before_hash = sha256(checkpoint)
    target = target_config(old)
    cfg_file = tmp_path / 'efficient.json'
    cfg_file.write_text(json.dumps(target))
    migration = tmp_path / 'migration'
    report = migrate(checkpoint, cfg_file, tiny_data, ROOT, migration, 'cpu', False)
    assert not migration.exists()
    assert report['kind'] == 'explicit_integrator_and_backward_migration'
    assert report['previous_migration'] == saved['solver_migration']
    assert report['algorithm_changes']['substeps'] == {'before': 2, 'after': 3}
    assert set(report['source_changes']) == ADDED_SOURCES
    migrate(checkpoint, cfg_file, tiny_data, ROOT, migration, 'cpu', True)
    migrated = torch.load(migration / 'last.pt', weights_only=True)
    for key in ('state_dict', 'optimizer', 'rng', 'step', 'sampler_next_step',
                'validation_indices', 'dataset_sha256'):
        _assert_nested_equal(saved[key], migrated[key])
    assert migrated['validation_status'] == 'not_run' and migrated['validation'] is None
    assert all(migrated[key] is None for key in ('best_score', 'best_step', 'best_checkpoint_path'))
    assert sha256(checkpoint) == before_hash
    with pytest.raises(FileExistsError):
        migrate(checkpoint, cfg_file, tiny_data, ROOT, migration, 'cpu', True)
    resumed = tmp_path / 'new_run'
    status = execute_run(tiny_data, resumed, target, resume=migration / 'last.pt')
    assert status['status'] == 'completed' and status['optimizer_steps_this_run'] == 2
    replay = json.loads((resumed / 'resume_validation.json').read_text())
    assert replay['step'] == 1 and replay['optimizer_steps'] == 0
    final = torch.load(resumed / 'last.pt', weights_only=True)
    assert final['config']['core']['substeps'] == 3 and final['step'] == 3
    assert final['solver_migration']['original_checkpoint_sha256'] == before_hash
    assert final['solver_migration']['previous_migration'] == saved['solver_migration']
    assert any(not torch.equal(saved['state_dict'][key], final['state_dict'][key])
               for key in saved['state_dict'])
    assert all(float(value['step']) == 3 for value in final['optimizer']['state'].values())
