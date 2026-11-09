"""Module freezing and data-scope contracts; no optimizer steps or training."""
import copy

import h5py
import numpy as np
import pytest
import torch

from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.train_dissipative import (
    DEFAULT_CONFIG, build_optimizer_parameter_groups, config_digest,
    configure_trainable_modules, preflight, read_config, validate_checkpoint,
    validate_config,
)


@pytest.fixture
def config():
    cfg = read_config(DEFAULT_CONFIG)
    cfg['model'].update(encoder_width=8, sigreg_projections=8)
    cfg['model'].setdefault('core', {}).update(hidden_dim=8)
    return cfg


def make_model(config):
    torch.set_num_threads(2)
    torch.manual_seed(5703)
    return DissipativeWorldModel(config['model'])


@pytest.mark.parametrize('selected', [[], None, 'encoder', ['encoder', 'encoder'], ['bad'],
                                      [None], [['encoder']], [''], [True], [float('nan')]])
def test_invalid_module_selection_is_rejected_without_model(config, selected):
    config['training']['trainable_modules'] = selected
    with pytest.raises(ValueError, match='trainable_modules'):
        validate_config(config)


@pytest.mark.parametrize('selected,model_config,match', [
    (['damping'], {'core': {'damping_mode': 'none'}}, 'damping'),
    (['environment'], {'environment_context_dim': 0}, 'environment'),
])
def test_nonexistent_modules_rejected_during_validation_and_configuration(config, selected, model_config, match):
    config['training']['trainable_modules'] = selected
    config['model'].update(model_config)
    with pytest.raises(ValueError, match=match):
        validate_config(config)
    model = make_model(config)
    with pytest.raises(ValueError, match=match):
        configure_trainable_modules(model, config['training'])


def test_undamped_stage_a_backward_freezes_unselected_modules_without_updates(config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('This test must not construct or step an optimizer')
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    config['training']['trainable_modules'] = ['encoder', 'initializer', 'potential']
    config['model'].update(velocity_initializer='motion_gated', motion_loss_mode='normalized_velocity',
                           momentum_weight=.1, environment_context_dim=0)
    config['model']['core'].update(damping_mode='none', potential_mode='raw')
    validate_config(config)
    model = make_model(config)
    # A stale gradient on a module being frozen must not survive selection.
    for parameter in model.core.input_net.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    report = configure_trainable_modules(model, config['training'])
    groups = build_optimizer_parameter_groups(model, config['training'])
    assert report['trainable_modules'] == ['encoder', 'initializer', 'potential']
    assert report['modules']['damping']['available'] is False
    assert report['modules']['environment']['available'] is False
    assert report['modules']['input_port']['trainable'] is False
    ids = [id(parameter) for group in groups for parameter in group['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert not set(ids) & {id(parameter) for parameter in model.core.input_net.parameters()}
    assert report['trainable_parameters'] + report['frozen_parameters'] == report['total_parameters']
    history, horizon = model.history, 2
    rng = torch.Generator().manual_seed(5704)
    batch = dict(pixels=torch.rand(2, history + horizon, 3, 64, 64, generator=rng),
                 action=torch.zeros(2, history + horizon - 1, 1),
                 dt_frame=torch.full((2, history + horizon - 1, 1), model.dt))
    metrics = model.loss(batch, horizon)
    metrics['loss'].backward()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
        elif name != 'core.potential_net.0.4.bias':
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.encoder.net[0].weight.grad.abs().sum() > 0
    assert model.velocity_correction[-1].weight.grad.abs().sum() > 0
    assert model.core.potential_net[0][0].weight.grad.abs().sum() > 0
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


@pytest.mark.parametrize('selected', ['encoder', 'initializer', 'potential', 'damping', 'environment', 'input_port'])
def test_single_module_produces_only_one_nonempty_group(config, selected):
    config['model']['environment_context_dim'] = 2
    config['training']['trainable_modules'] = [selected]
    model = make_model(config)
    report = configure_trainable_modules(model, config['training'])
    groups = build_optimizer_parameter_groups(model, config['training'])
    assert len(groups) == 1
    assert groups[0]['name'] == ('encoder' if selected == 'encoder' else 'dynamics')
    assert groups[0]['params']
    assert report['trainable_modules'] == [selected]
    assert sum(parameter.numel() for parameter in groups[0]['params']) == report['trainable_parameters']


def test_constant_damping_parameter_is_supported(config):
    config['model']['core']['damping_mode'] = 'constant'
    config['model']['environment_context_dim'] = 0
    config['training']['trainable_modules'] = ['damping']
    model = make_model(config)
    report = configure_trainable_modules(model, config['training'])
    groups = build_optimizer_parameter_groups(model, config['training'])
    assert model.core.damping_net is None
    assert len(groups[0]['params']) == 1 and groups[0]['params'][0] is model.core.constant_factor
    assert report['modules']['damping']['parameters'] == model.core.constant_factor.numel()


def test_default_selection_preserves_legacy_order_and_all_trainable_parameters(config):
    model = make_model(config)
    old_encoder = model.encoder_parameters()
    old_dynamics = model.dynamics_parameters()
    report = configure_trainable_modules(model, config['training'])
    groups = build_optimizer_parameter_groups(model, config['training'])
    assert report['mode'] == 'legacy_all' and report['requested_modules'] is None
    assert report['frozen_parameters'] == 0
    assert [id(p) for p in groups[0]['params']] == [id(p) for p in old_encoder]
    assert [id(p) for p in groups[1]['params']] == [id(p) for p in old_dynamics]
    assert groups[0]['lr'] == groups[0]['base_lr'] == config['training']['encoder_lr']
    assert groups[1]['lr'] == groups[1]['base_lr'] == config['training']['dynamics_lr']


def test_parameter_group_contract_rejects_duplicate_and_missing_trainable_tensor(config):
    model = make_model(config)
    configure_trainable_modules(model, config['training'])
    original = model.encoder_parameters
    model.encoder_parameters = lambda: original() + original()[:1]
    with pytest.raises(ValueError, match='partition'):
        build_optimizer_parameter_groups(model, config['training'])
    model.encoder_parameters = lambda: original()[1:]
    with pytest.raises(ValueError, match='partition'):
        build_optimizer_parameter_groups(model, config['training'])


def test_selection_and_data_scope_remain_bound_to_strict_resume(config):
    checkpoint = dict(format_version=1, kind='dissipative', config_sha256=config_digest(config),
                      train_config=copy.deepcopy(config), dataset_sha256='data', source_hashes={'train': 'hash'},
                      device='cpu', step=0, sampler_next_step=0, state_dict={}, optimizer={}, rng={},
                      config={}, validation_indices=[], best_score=None, best_step=None)
    validate_checkpoint(checkpoint, config, 'data', {'train': 'hash'}, 'cpu')
    for addition in ({'trainable_modules': ['encoder', 'initializer', 'potential']},
                     {'required_data_scope': 'undamped_stage_a'}):
        changed = copy.deepcopy(config)
        changed['training'].update(addition)
        with pytest.raises(ValueError, match='config_sha256'):
            validate_checkpoint(checkpoint, changed, 'data', {'train': 'hash'}, 'cpu')
    with pytest.raises(ValueError, match='source_hashes'):
        validate_checkpoint(checkpoint, config, 'data', {'train': 'changed'}, 'cpu')


@pytest.fixture
def data(tmp_path):
    path = tmp_path / 'scope_data.h5'
    episodes, frames = 6, 81
    n = episodes * frames
    with h5py.File(path, 'w') as f:
        f.attrs['train_frames'] = frames
        f.create_dataset('pixels', data=np.zeros((n, 64, 64, 3), dtype=np.uint8))
        f.create_dataset('action', data=np.zeros((n, 1), dtype=np.float32))
        f.create_dataset('dt_frame', data=np.full((n, 1), .1, dtype=np.float32))
        valid = np.ones((episodes, frames), dtype=bool)
        valid[:, -1] = False
        f.create_dataset('valid_transition', data=valid.ravel())
        f.create_dataset('ep_offset', data=np.arange(episodes) * frames)
        f.create_dataset('ep_len', data=np.full(episodes, frames))
        for name, ids in [('train', [0, 1]), ('val', [2, 3]), ('test', [4, 5])]:
            f.create_dataset('splits/' + name, data=np.asarray(ids, dtype=np.int64))
    return path


def test_data_scope_preflight_rejects_unspecified_or_wrong_scope_and_reports_match(config, data, monkeypatch):
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: pytest.fail('No optimizer in preflight'))
    assert preflight(data, config)['training_scope'] == 'unspecified'
    config['training']['required_data_scope'] = 'undamped_stage_a'
    with pytest.raises(ValueError, match='training_scope'):
        preflight(data, config)
    with h5py.File(data, 'r+') as f:
        f.attrs['training_scope'] = 'multi_damping'
    with pytest.raises(ValueError, match='required_data_scope'):
        preflight(data, config)
    with h5py.File(data, 'r+') as f:
        f.attrs['training_scope'] = np.bytes_('undamped_stage_a')
    report = preflight(data, config)
    assert report['training_scope'] == report['required_data_scope'] == 'undamped_stage_a'
    assert report['optimizer_created'] is False
    assert report['test_samples_read'] is False


@pytest.mark.parametrize('scope', ['', ' ', None, [], 1])
def test_invalid_required_scope_is_rejected(config, scope):
    config['training']['required_data_scope'] = scope
    with pytest.raises(ValueError, match='required_data_scope'):
        validate_config(config)
