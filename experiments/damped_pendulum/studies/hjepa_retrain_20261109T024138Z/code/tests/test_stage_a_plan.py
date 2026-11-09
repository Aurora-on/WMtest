"""Stage-A executable plan contracts; tiny data and intercepted shell calls only.

No optimizer is constructed, no real trainer is called, and no Slurm job is
submitted. The shell's own checkpoint guard runs against tiny metadata files.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from dpbench.dissipative_model import DissipativeWorldModel, resolve_model_config
from dpbench.passive_family import generate_undamped
from dpbench.train_dissipative import (
    build_optimizer_parameter_groups, configure_trainable_modules, preflight, read_config,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_CONFIG = ROOT / 'configs/dissipative_undamped_stage_a_v1.json'
PARENT_CONFIG = ROOT / 'configs/dissipative_passive_fixed_v1.json'
DATA_CONFIG = ROOT / 'configs/passive_family_smooth_v1.json'
PLAN = ROOT / 'studies/undamped_stage_a_20261105/plan.json'
SCRIPT = ROOT / 'scripts/slurm_undamped_stage_a.sh'


def test_actual_stage_config_only_changes_the_conservative_stage_scope():
    parent = read_config(PARENT_CONFIG)
    stage = read_config(STAGE_CONFIG)
    expected = copy.deepcopy(parent)
    expected['model']['environment_context_dim'] = 0
    expected['model']['core']['damping_mode'] = 'none'
    expected['training'].update(trainable_modules=['encoder', 'initializer', 'potential'],
                                required_data_scope='undamped_stage_a')
    # This exact inheritance check includes every solver/time/LR/loss setting;
    # stage selection cannot silently change more than the intended scope.
    assert stage == expected


def test_plan_time_and_budget_contract_agrees_with_executable_configs():
    stage = read_config(STAGE_CONFIG)
    model = resolve_model_config(stage['model'])
    data = json.loads(DATA_CONFIG.read_text())
    plan = json.loads(PLAN.read_text())
    times = plan['time_invariants']
    assert times['dt_seconds'] == model['dt'] == data['dt'] == .1
    assert times['simulation_substeps'] == data['substeps'] == 20
    assert times['model_substeps'] == model['core']['substeps'] == 3
    assert times['history_frames'] == model['history'] == data['history'] == 21
    assert times['velocity_history_frames'] == model['velocity_history'] == 5
    assert times['train_frames'] == stage['training']['train_frames'] == data['train_frames'] == 81
    assert times['test_frames'] == data['test_frames'] == 341
    for key in ('curriculum', 'val_horizons', 'val_every'):
        assert times[key] == stage['training'][key]
    assert plan['optimization']['full_budget'] == stage['training']['steps'] == 20000
    for key in ('batch_size', 'seed', 'encoder_lr', 'dynamics_lr', 'warmup_steps'):
        assert plan['optimization'][key] == stage['training'][key]
    assert [row['stop_after_step'] for row in plan['milestones']] == [250, 1000, 4000, 10000, 20000]
    assert [row['resume_step'] for row in plan['milestones']] == [None, 250, 1000, 4000, 10000]
    assert plan['data_selection']['groups'] == data['groups']


def test_actual_stage_model_has_exactly_zero_damping_and_only_declared_trainable_modules(monkeypatch):
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: pytest.fail('No optimizer may be created'))
    torch.set_num_threads(2)
    cfg = read_config(STAGE_CONFIG)
    model = DissipativeWorldModel(cfg['model'])
    report = configure_trainable_modules(model, cfg['training'])
    groups = build_optimizer_parameter_groups(model, cfg['training'])
    q = torch.tensor([[0., 0.], [1., -2.], [10., -20.]], dtype=torch.float64)
    p = torch.tensor([[3., -4.], [.1, .2], [0., 1.]], dtype=torch.float64)
    damping = model.core.damping(q, p, context=None)
    assert damping.shape == (3, 2, 2) and torch.count_nonzero(damping) == 0
    assert model.environment_encoder is None
    assert model.core.damping_net is None and model.core.constant_factor is None
    assert report['trainable_modules'] == ['encoder', 'initializer', 'potential']
    grouped = [id(parameter) for group in groups for parameter in group['params']]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert all(not parameter.requires_grad and id(parameter) not in grouped
               for parameter in model.core.input_net.parameters())


def test_tiny_undamped_data_accepts_real_stage_configuration_in_readonly_preflight(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: pytest.fail('No optimizer may be created'))
    data_cfg = json.loads(DATA_CONFIG.read_text())
    # Only episode counts are reduced. Keep the full physical clock, histories,
    # frame counts and numerical reference resolution of the real protocol.
    data_cfg['groups'] = dict(train=4, val=4, test=4)
    output = tmp_path / 'tiny_undamped'
    generated = generate_undamped(data_cfg, output)
    path = output / 'undamped.h5'
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    files_before = sorted(p.name for p in output.iterdir())
    with h5py.File(path, 'r') as handle:
        assert handle.attrs['training_scope'] == 'undamped_stage_a'
        assert handle.attrs['renderer'] == 'smooth_v1'
        assert not bool(handle.attrs['requires_environment_context'])
        assert np.count_nonzero(handle['action'][:]) == 0
    report = preflight(path, read_config(STAGE_CONFIG))
    assert report['mode'] == 'preflight_only' and report['optimizer_created'] is False
    assert report['run_directory_created'] is False and report['test_samples_read'] is False
    assert report['training_scope'] == report['required_data_scope'] == 'undamped_stage_a'
    assert report['trainable_modules'] == ['encoder', 'initializer', 'potential']
    assert report['validation_horizons_seconds'] == [.8, 3.2]
    assert before == (path.stat().st_size, path.stat().st_mtime_ns)
    assert files_before == sorted(p.name for p in output.iterdir())
    assert generated['training_started'] is False


@pytest.fixture
def shell_harness(tmp_path):
    """Intercept Python entrypoints, while executing the real metadata guard."""
    log = tmp_path / 'python_calls.jsonl'
    stub = tmp_path / 'intercepted python'
    stub.write_text(
        '#!' + sys.executable + '\n'
        'import json, os, sys\n'
        'with open(os.environ["STAGE_TEST_ARGV_LOG"], "a") as handle:\n'
        '    handle.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'if sys.argv[1:2] == ["-"]:\n'
        '    source = sys.stdin.read()\n'
        '    sys.argv = sys.argv[1:]\n'
        '    exec(compile(source, "<shell_checkpoint_guard>", "exec"))\n'
        'elif sys.argv[1:3] == ["-m", "dpbench.train_dissipative"]:\n'
        '    pass  # NEVER import or execute the trainer.\n'
        'elif sys.argv[1:3] == ["scripts/bootstrap_upstream.py", "--verify"]:\n'
        '    pass  # No subprocess or network action.\n'
        'else:\n'
        '    raise SystemExit("Unexpected Python entrypoint in dry run")\n'
    )
    stub.chmod(0o755)
    environment = os.environ.copy()
    for key in ('PENDULUM_STOP_AFTER', 'PENDULUM_RESUME'):
        environment.pop(key, None)
    environment.update(PENDULUM_CODE_DIR=str(ROOT), PENDULUM_PYTHON=str(stub),
                       PENDULUM_DATA=str(tmp_path / 'undamped data.h5'),
                       PENDULUM_RUN_ROOT=str(tmp_path / 'run root'),
                       STAGE_TEST_ARGV_LOG=str(log))

    def run(**variables):
        env = dict(environment, **{key: str(value) for key, value in variables.items()})
        result = subprocess.run(['bash', str(SCRIPT)], env=env, text=True, capture_output=True, timeout=30)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        assert not (tmp_path / 'run root').exists()
        return result, calls, env

    return run


def test_shell_default_runs_only_bounded_250_segment_without_changing_schedule(shell_harness):
    result, calls, env = shell_harness()
    assert result.returncode == 0, result.stderr
    command = next(call for call in calls if call[:2] == ['-m', 'dpbench.train_dissipative'])
    assert command[command.index('--stop-after-step') + 1] == '250'
    assert command[command.index('--config') + 1] == 'configs/dissipative_undamped_stage_a_v1.json'
    assert command[command.index('--output') + 1] == str(Path(env['PENDULUM_RUN_ROOT']) / 'milestone-250')
    assert command[command.index('--data') + 1] == env['PENDULUM_DATA']
    assert '--steps' not in command and '--fixed-horizon' not in command and '--resume' not in command
    assert calls.count(command) == 1


@pytest.mark.parametrize('stop,previous', [(1000, 250), (4000, 1000), (10000, 4000), (20000, 10000)])
def test_shell_later_segment_accepts_only_completed_preceding_milestone(shell_harness, tmp_path, stop, previous):
    checkpoint = tmp_path / 'preceding checkpoint.pt'
    torch.save(dict(step=previous, validation_status='completed'), checkpoint)
    result, calls, _ = shell_harness(PENDULUM_STOP_AFTER=stop, PENDULUM_RESUME=checkpoint)
    assert result.returncode == 0, result.stderr
    command = next(call for call in calls if call[:2] == ['-m', 'dpbench.train_dissipative'])
    assert command[command.index('--stop-after-step') + 1] == str(stop)
    assert command[command.index('--resume') + 1] == str(checkpoint)
    assert calls[0] == ['-', str(checkpoint), str(previous)]
    assert '--steps' not in command and '--fixed-horizon' not in command


@pytest.mark.parametrize('stop', ['0', '251', '20001', 'invalid'])
def test_shell_rejects_invalid_stop_before_any_python_execution(shell_harness, stop):
    result, calls, _ = shell_harness(PENDULUM_STOP_AFTER=stop)
    assert result.returncode != 0 and 'PENDULUM_STOP_AFTER' in result.stderr
    assert calls == []


@pytest.mark.parametrize('stop', [1000, 20000])
def test_shell_requires_resume_for_later_milestones(shell_harness, stop):
    result, calls, _ = shell_harness(PENDULUM_STOP_AFTER=stop)
    assert result.returncode != 0 and 'requires PENDULUM_RESUME' in result.stderr
    assert calls == []


def test_shell_rejects_resume_on_first_segment(shell_harness, tmp_path):
    result, calls, _ = shell_harness(PENDULUM_RESUME=tmp_path / 'old.pt')
    assert result.returncode != 0 and 'must start fresh' in result.stderr
    assert calls == []


@pytest.mark.parametrize('metadata', [dict(step=1000, validation_status='completed'),
                                     dict(step=250, validation_status='not_run')])
def test_shell_rejects_wrong_step_or_incomplete_validation_before_training(shell_harness, tmp_path, metadata):
    checkpoint = tmp_path / 'last.pt'
    torch.save(metadata, checkpoint)
    result, calls, _ = shell_harness(PENDULUM_STOP_AFTER=1000, PENDULUM_RESUME=checkpoint)
    assert result.returncode != 0 and 'preceding milestone' in result.stderr
    assert len(calls) == 1 and calls[0][0] == '-'
    assert not any(call[:2] == ['-m', 'dpbench.train_dissipative'] for call in calls)
