"""Small CPU fixtures: no training, no held-out test data, no optimizer."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('staged_evaluator', ROOT / 'scripts/evaluate_staged_damping.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_dim = config['q_dim']
        self.encoder = torch.nn.Linear(1, 1)
        self.velocity_correction = torch.nn.Linear(1, 1)
        self.core = torch.nn.Module()
        self.core.potential_net = torch.nn.Linear(1, 1)
        self.core.input_net = torch.nn.Linear(1, 1)
        if config['core']['damping_mode'] != 'none':
            self.core.damping_net = torch.nn.Linear(1, 1)
        if config['environment_context_dim']:
            self.environment_encoder = torch.nn.Linear(1, 4)
        for parameter in self.parameters():
            torch.nn.init.zeros_(parameter)


def saved_checkpoint(stage='a', multi=False):
    context = 4 if multi else 0
    modules = ['encoder', 'initializer', 'potential'] if stage == 'a' else (['damping', 'environment'] if multi else ['damping'])
    scope = 'undamped_stage_a' if stage == 'a' else ('damping_stage_b_multi' if multi else 'damping_stage_b_single')
    result = dict(kind='dissipative', step=20000 if stage == 'a' else 750, validation_status='completed',
        source_hashes={'core': 'fixed'}, dataset_sha256='data',
        config=dict(q_dim=2, objective_mode='lehami', history=21, velocity_history=5, dt=.1,
                    encoder_width=32, velocity_correction_scale=1., velocity_initializer='motion_gated',
                    environment_context_dim=context, target_tail_frames=2, momentum_reference_time=.1,
                    core=dict(damping_mode='none' if stage == 'a' else 'position_velocity',
                              potential_mode='raw', substeps=3, integrator='frozen_dg')),
        train_frames=81,
        train_config=dict(training=dict(trainable_modules=modules, required_data_scope=scope, required_renderer='smooth_v1', train_frames=81)),
        trainability=dict(trainable_modules=modules))
    result['state_dict'] = Model(result['config']).state_dict()
    return result


def make_data(path, damping=(0.,), scope='undamped_stage_a', groups=448):
    frames = 81
    group = np.tile(np.arange(groups), len(damping))
    bvalues = np.repeat(damping, groups)
    phase = np.arange(frames)[None, :] + (group % 2)[:, None] * 2
    theta = phase * .01
    count = len(group)
    with h5py.File(path, 'w') as f:
        f.attrs.update(training_scope=scope, renderer='smooth_v1', mass=1., length=1., gravity=9.81,
                       requires_environment_context=len(damping) > 1, history_required=21, dt=.1, train_frames=81,
                       parent_config_sha256='f' * 64, selected_damping_json=json.dumps(list(damping)))
        f['ep_offset'] = np.arange(count) * frames
        f['ep_len'] = np.full(count, frames)
        f['group_id'] = group
        f['energy_band'] = group % 4
        f['damping'] = bvalues
        f['state'] = np.stack((theta, np.full_like(theta, .1)), -1).reshape(-1, 2)
        f['pixels'] = np.broadcast_to(phase.astype(np.uint8)[..., None, None, None], (count, frames, 1, 1, 3)).reshape(-1, 1, 1, 3)
        f['action'] = np.zeros((count * frames, 1), np.float32)
        f['dt_frame'] = np.full((count * frames, 1), .1, np.float32)
        if groups == 448:
            f['splits/train'] = np.flatnonzero(group < 384)
            f['splits/val'] = np.flatnonzero(group >= 384)
        # Deliberately no test split or unseen data exists.


def pixels_tensor(values, device):
    return torch.as_tensor(np.array(values).copy(), device=device).permute(0, 3, 1, 2).float()


def fake_autonomous(model, images, past_actions, past_dt, actions, dt, diagnostic_names):
    assert images.shape[1] == 21 and past_actions.shape[1] == past_dt.shape[1] == 20
    assert actions.shape == dt.shape and torch.all(actions == 0)
    b, steps = actions.shape[:2]
    result = np.zeros((b, steps + 1, 2 * model.q_dim))
    result[..., 0] = images[:, -1, 0, 0, 0].cpu().numpy()[:, None] * .01 + np.arange(steps + 1)[None] * .01
    result[..., model.q_dim] = .1
    return dict(features=result, latent_energy=np.ones((b, steps + 1)),
                valid_steps=np.full(b, steps), failures=[], diagnostics={})


def calibration(model, path, pairs, batch_size):
    with h5py.File(path, 'r') as f:
        # B is allowed only observed val histories, never training calibration.
        if f.attrs['training_scope'] != 'undamped_stage_a':
            assert set(ep for ep, _ in pairs) <= set(f['splits/val'][:])
        offsets = f['ep_offset'][:]
        states = f['state'][:][np.array([offsets[ep] + t for ep, t in pairs])]
    features = np.zeros((len(pairs), 4))
    features[:, 0], features[:, 2] = states.T
    targets = np.stack((np.sin(states[:, 0]), np.cos(states[:, 0]), states[:, 1]), -1)
    return features, targets


def install_helpers(monkeypatch):
    monkeypatch.setattr(ev, 'evaluation_sources', lambda *a: {})
    monkeypatch.setattr(ev, 'load_frozen_helpers', lambda *a: (Model, calibration, fake_autonomous,
                                                            pixels_tensor, lambda: {'core': 'fixed'}, ()))
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: pytest.fail('No model optimizer allowed'))


def evaluate(tmp_path, stage, monkeypatch, readout=None, multi=False, damping=.2, bad_teacher=False):
    install_helpers(monkeypatch)
    name = stage + ('multi' if multi else 'single')
    data = tmp_path / (name + '.h5')
    saved = saved_checkpoint(stage, multi)
    values = (0.,) if stage == 'a' else (ev.SEEN_DAMPING if multi else (damping,))
    make_data(data, values, saved['train_config']['training']['required_data_scope'])
    if bad_teacher:
        original = calibration
        def wrong(model, path, pairs, batch):
            features, labels = original(model, path, pairs, batch)
            with h5py.File(path, 'r') as f:
                val = set(f['splits/val'][:])
            if all(ep in val for ep, _ in pairs):
                features[:, 0] += 1e6
            return features, labels
        monkeypatch.setattr(ev, 'load_frozen_helpers', lambda *a: (Model, wrong, fake_autonomous,
                                                                pixels_tensor, lambda: {'core': 'fixed'}, ()))
    saved['dataset_sha256'] = ev.sha256(data)
    checkpoint = tmp_path / (name + '.pt')
    torch.save(saved, checkpoint)
    output = tmp_path / (name + '_evaluation')
    args = ['--checkpoint', str(checkpoint), '--data', str(data), '--output', str(output),
            '--device', 'cpu', '--threads', '1', '--stage', stage, '--expected-step', str(saved['step'])]
    if readout:
        args += ['--readout', str(readout)]
    return ev.main(args), output, saved, data


def test_a20k_and_b_single_multi_preserve_readout_and_world_model(tmp_path, monkeypatch):
    a, out_a, saved_a, _ = evaluate(tmp_path, 'a', monkeypatch)
    assert a['checkpoint_step'] == 20000 and a['readout_refit']
    assert a['calibration']['fit_groups'] == 288 and a['calibration']['selection_groups'] == 96
    # Any accidental fit in B fails immediately; only load() is legal.
    monkeypatch.setattr(ev.KNNReadout, '__init__', lambda *a, **kw: pytest.fail('B refit prohibited'))
    for multi in (False, True):
        b, output, saved_b, data = evaluate(tmp_path, 'b', monkeypatch, out_a / 'readout.npz', multi=multi, damping=.1)
        assert b['status'] == 'completed' and b['test_split_accessed'] is False
        assert b['world_model_optimizer_created'] is False and b['world_model_updates'] == 0
        assert b['model_hash_before'] == b['model_hash_after']
        assert b['shared_frozen_matches_A'] and not b['readout_refit']
        assert b['readout_sha256'] == a['readout_sha256']
        assert (output / 'readout.npz').read_bytes() == (out_a / 'readout.npz').read_bytes()
        assert (output / 'readout_provenance.json').read_bytes() == (out_a / 'readout_provenance.json').read_bytes()
        result = b['evaluation']['val']
        assert result['unique_group_count'] == 64
        assert result['record_count'] == (384 if multi else 64)
        assert set(result['rollout']) == {'8', '32'}
        assert result['rollout']['32']['groups']['all']['n_unique_groups'] == 64
        with np.load(output / 'trajectories.npz') as arrays:
            assert arrays['features'].shape == (result['record_count'], 33, 4)
            assert arrays['valid_steps'].min() == 32
            if not multi:
                with np.load(out_a / 'trajectories.npz') as original:
                    np.testing.assert_array_equal(arrays['pred'], original['pred'])
        with np.load(output / 'val/animation_inputs.npz') as arrays:
            assert arrays['truth'].shape == arrays['autonomous_prediction'].shape == (4, 33, 2)
        if multi:
            for value in ev.SEEN_DAMPING:
                group = result['rollout']['32']['groups'][f'damping_{value:g}']
                assert group['n_records'] == group['n_unique_groups'] == 64
                assert (output / f'val/damping_{value:g}/animation_inputs.npz').exists()
        records, physical, groups, _, parent = ev.inspect_data(data, b['training_scope'])
        changed = copy.deepcopy(saved_b)
        changed['state_dict']['encoder.weight'][0, 0] = 1.
        with pytest.raises(ValueError, match='provenance/frozen mechanics'):
            ev.verify_readout_provenance(out_a / 'readout.npz', changed, groups, physical, parent)
        changed = copy.deepcopy(saved_b)
        changed['config']['velocity_correction_scale'] = 2.
        with pytest.raises(ValueError, match='provenance/frozen mechanics'):
            ev.verify_readout_provenance(out_a / 'readout.npz', changed, groups, physical, parent)
    with (out_a / 'readout.npz').open('ab') as f:
        f.write(b'tampered')
    with pytest.raises(ValueError, match='provenance'):
        ev.verify_readout_provenance(out_a / 'readout.npz', saved_b, groups, physical, parent)


def test_checkpoint_exact_step_sources_scope_and_trainability():
    for stage, multi in (('a', False), ('b', False), ('b', True)):
        saved = saved_checkpoint(stage, multi)
        ev.verify_checkpoint(saved, {'core': 'fixed'}, 'data', stage, saved['step'])
        for field, value in [('step', 10), ('validation_status', 'not_run'), ('dataset_sha256', 'wrong'), ('source_hashes', {})]:
            wrong = copy.deepcopy(saved); wrong[field] = value
            with pytest.raises(ValueError):
                ev.verify_checkpoint(wrong, {'core': 'fixed'}, 'data', stage, saved['step'])
        wrong = copy.deepcopy(saved); wrong['trainability']['trainable_modules'] = ['encoder']
        with pytest.raises(ValueError, match='trainability'):
            ev.verify_checkpoint(wrong, {'core': 'fixed'}, 'data', stage, saved['step'])
    with pytest.raises(ValueError, match='exact step20000'):
        ev.verify_checkpoint(saved_checkpoint(), {'core': 'fixed'}, 'data', 'a', 10000)


def test_data_scope_condition_pairing_and_test_access(tmp_path):
    path = tmp_path / 'multi.h5'
    make_data(path, ev.SEEN_DAMPING, 'damping_stage_b_multi')
    rows, _, groups, values, _ = ev.inspect_data(path, 'damping_stage_b_multi')
    assert len(rows['val']) == 384 and len(groups['val']) == 64
    assert values == ev.SEEN_DAMPING
    with pytest.raises(ValueError):
        ev.inspect_data(path, 'damping_stage_b_single')
    with h5py.File(path, 'r+') as f:
        f['damping'][0] = .2
    with pytest.raises(ValueError, match='exactly once'):
        ev.inspect_data(path, 'damping_stage_b_multi')
    with h5py.File(path, 'r') as f:
        with pytest.raises(ValueError, match='never reads test'):
            ev.read_records(f, 'test', 53, values)


def test_future_rgb_never_changes_autonomous_features_and_failed_suffix_is_not_held(tmp_path):
    path = tmp_path / 'tiny.h5'; make_data(path, groups=2)
    rows = [dict(episode=0), dict(episode=1)]
    model = SimpleNamespace(q_dim=2)
    first = ev.collect_rollouts(model, path, rows, 32, 32, 'cpu', fake_autonomous, pixels_tensor, ())
    with h5py.File(path, 'r+') as f:
        for episode in range(2):
            offset = int(f['ep_offset'][episode]); f['pixels'][offset + 21:offset + 81] = 255
    changed = ev.collect_rollouts(model, path, rows, 32, 32, 'cpu', fake_autonomous, pixels_tensor, ())
    np.testing.assert_array_equal(first['features'], changed['features'])
    def invalid(*args, **kwargs):
        result = fake_autonomous(*args, **kwargs); result['valid_steps'][0] = 4
        return result
    with pytest.raises(ValueError, match='suffix must remain NaN'):
        ev.collect_rollouts(model, path, rows, 32, 32, 'cpu', invalid, pixels_tensor, ())


def test_bad_quality_still_writes_completed_report(tmp_path, monkeypatch):
    summary, output, _, _ = evaluate(tmp_path, 'a', monkeypatch, bad_teacher=True)
    assert summary['status'] == 'completed'
    assert not summary['evaluation']['val']['rollout']['32']['teacher_quality_gate_passed']
    assert (output / 'summary.json').exists() and (output / 'readout.npz').exists()


def test_physical_stop_and_failure_metrics_use_truth_and_coverage():
    truth = np.ones((2, 33, 2)); pred = truth.copy(); pred[..., 1] = 0
    pred[1, 5:] = np.nan
    rows = [dict(energy_band=0, group_id=0, damping=.2), dict(energy_band=1, group_id=1, damping=.2)]
    result = ev.summarize_horizon(pred, truth, truth, np.array([32, 4]), rows, 32, 1.,
                                  dict(mass=1., length=1., gravity=9.81))
    assert result['groups']['all']['coverage'] == .5
    assert result['groups']['all']['metrics']['stopped_dynamic_window_fraction'] == 1.
    assert result['groups']['energy_band1']['metrics']['angle_mae_rad'] is None
    assert not result['ranking_eligible']
    truth[..., 1] = 0
    assert ev.motion_metrics(np.nan_to_num(pred), truth, 1.)['dynamic_window_count'] == 0
