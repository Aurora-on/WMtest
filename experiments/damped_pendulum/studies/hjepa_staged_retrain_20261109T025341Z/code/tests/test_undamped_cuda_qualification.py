"""CPU tests of no-optimizer qualification and fail-closed CUDA/Slurm routing."""
from pathlib import Path

import pytest
import torch

from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.train_dissipative import read_config
from scripts.qualify_undamped_stage_a import main, require_cuda_allocation, run_checks

ROOT = Path(__file__).resolve().parents[1]


def tiny_inputs():
    torch.set_num_threads(2)
    torch.manual_seed(5706)
    config = read_config(ROOT / 'configs/dissipative_undamped_stage_a_v1.json')
    config['model'].update(encoder_width=8, sigreg_projections=8)
    config['model']['core']['hidden_dim'] = 8
    model = DissipativeWorldModel(config['model'])
    batch = dict(pixels=torch.rand(2, model.history + 32, 3, 64, 64),
                 action=torch.zeros(2, model.history + 31, 1),
                 dt_frame=torch.full((2, model.history + 31, 1), model.dt))
    return model, batch, config


def test_cpu_helper_checks_real_forward_backward_without_optimizer_or_state_updates(monkeypatch):
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *args, **kwargs: pytest.fail('Optimizer forbidden'))
    model, batch, config = tiny_inputs()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = run_checks(model, batch, config, 'cpu')
    assert result['status'] == 'completed'
    assert result['optimizer_created'] is False and result['optimizer_updates'] == 0
    assert result['checkpoint_saved'] is False
    assert result['initial_state_sha256'] == result['final_state_sha256']
    assert [row['horizon'] for row in result['rows']] == [4, 8, 32]
    assert [row['backward'] for row in result['rows']] == [True, True, False]
    for row in result['rows']:
        assert row['numerical_audit']['core_steps'] == row['horizon']
        assert row['numerical_audit']['maximum_abs_damping'] == 0
        assert row['numerical_audit']['maximum_abs_work'] == 0
        assert row['numerical_audit']['maximum_abs_dissipation'] == 0
        if row['backward']:
            assert all(v['finite'] and v['nonzero'] and v['norm'] > 0
                       for v in row['gradients']['modules'].values())
            assert row['gradients']['frozen_parameters_have_no_gradient']
        else:
            assert row['gradients'] is None and row['backward_seconds'] is None
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('failure', ['damped', 'context', 'modules', 'actions', 'history'])
def test_helper_rejects_wrong_stage_contract(failure):
    model, batch, config = tiny_inputs()
    if failure == 'damped':
        model.core.damping_mode = 'constant'
    elif failure == 'context':
        model.environment_context_dim = 4
    elif failure == 'modules':
        config['training']['trainable_modules'].append('input_port')
    elif failure == 'actions':
        batch['action'][0, 0] = .1
    else:
        batch['pixels'] = batch['pixels'][:, 1:]
    with pytest.raises(ValueError):
        run_checks(model, batch, config, 'cpu')


def test_cli_refuses_login_node_before_reading_data_or_creating_output(tmp_path, monkeypatch):
    for name in ('SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_GPUS_ON_NODE'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *args, **kwargs: pytest.fail('Optimizer forbidden'))
    output = tmp_path / 'qualification.json'
    with pytest.raises(RuntimeError, match='Slurm GPU allocation'):
        main(['--config', str(ROOT / 'configs/dissipative_undamped_stage_a_v1.json'),
              '--data', str(tmp_path / 'absent.h5'), '--output', str(output)])
    assert not output.exists()


def test_allocation_requires_cuda_and_accepts_gpu_index_zero(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '5706')
    monkeypatch.setenv('SLURM_JOB_GPUS', '0')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='available CUDA'):
        require_cuda_allocation()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    device, allocation = require_cuda_allocation()
    assert device.type == 'cuda' and allocation['allocated_gpus'] == '0'


def test_cli_parse_cannot_enable_optimizer_or_cpu_execution():
    with pytest.raises(SystemExit):
        main(['--config', 'config.json', '--data', 'data.h5', '--output', 'result.json', '--execute'])
    with pytest.raises(SystemExit):
        main(['--config', 'config.json', '--data', 'data.h5', '--output', 'result.json', '--device', 'cpu'])
