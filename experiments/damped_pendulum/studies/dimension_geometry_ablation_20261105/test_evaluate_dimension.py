import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('dimension_evaluator', HERE / 'evaluate_dimension.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_knn_matches_existing_nonlinear_readout_and_preserves_nan_suffix():
    old_dir = ROOT / 'studies/control_visuals_20261105'
    sys.path.insert(0, str(old_dir))
    try:
        old_spec = importlib.util.spec_from_file_location('previous_readout', old_dir / 'compare_readouts.py')
        old = importlib.util.module_from_spec(old_spec); old_spec.loader.exec_module(old)
    finally:
        sys.path.remove(str(old_dir))
    rng = np.random.default_rng(5705)
    x, y = rng.normal(size=(70, 16)), rng.normal(size=(70, 3))
    sx, sy, vx = rng.normal(size=(11, 16)), rng.normal(size=(11, 3)), rng.normal(size=(7, 16))
    expected, k, candidates = old.knn_predictions(x, y, sx, sy, vx)
    readout = ev.KNNReadout(x, y, sx, sy)
    actual, distance = readout.predict(np.r_[vx, np.full((1, 16), np.nan)])
    np.testing.assert_allclose(actual[:-1], ev.targets_to_physics(expected), rtol=0, atol=0)
    assert readout.k == k and readout.candidates == candidates
    assert np.isnan(actual[-1]).all() and np.isnan(distance[-1])
    np.testing.assert_array_equal(readout.x_mean, x.mean(0))
    np.testing.assert_array_equal(readout.x_scale, np.maximum(x.std(0), 1e-6))


def saved_checkpoint(dim):
    return dict(kind='dissipative', step=10000, validation_status='completed',
                source_hashes={'core': 'fixed'}, dataset_sha256='data',
                config=dict(q_dim=dim, objective_mode='lehami', history=21, dt=.1,
                            environment_context_dim=0, target_tail_frames=2, momentum_reference_time=.1,
                            core=dict(damping_mode='none', potential_mode='raw', substeps=3, integrator='frozen_dg')))


@pytest.mark.parametrize('dim', [2, 8, 32])
def test_final_checkpoint_identity(dim):
    saved = saved_checkpoint(dim)
    ev.verify_checkpoint(saved, {'core': 'fixed'}, 'data', dim)
    for field, value in [('step', 6000), ('validation_status', 'not_run'), ('dataset_sha256', 'wrong')]:
        changed = copy.deepcopy(saved); changed[field] = value
        with pytest.raises(ValueError):
            ev.verify_checkpoint(changed, {'core': 'fixed'}, 'data', dim)
    wrong = copy.deepcopy(saved); wrong['config']['objective_mode'] = 'current'
    with pytest.raises(ValueError):
        ev.verify_checkpoint(wrong, {'core': 'fixed'}, 'data')


def test_physical_metrics_wrap_angles_and_stop_uses_true_motion():
    pred = np.array([[[np.pi - .01, 0.]] * 32])
    truth = np.array([[[-np.pi + .01, 1.]] * 32])
    metrics = ev.physical_metrics(pred, truth, 2.)
    assert metrics['angle_mae_rad'] == pytest.approx(.02)
    assert metrics['omega_nrmse'] == pytest.approx(.5)
    motion = ev.motion_metrics(pred, truth, 2.)
    assert motion['stopped_dynamic_window_fraction'] == 1
    assert motion['trajectory_with_stop_fraction'] == 1
    assert motion['dynamic_window_count'] == 23
    assert ev.motion_metrics(pred[:, :8], truth[:, :8], 2.)['stopped_dynamic_window_fraction'] is None
    truth[..., 1] = 0
    assert ev.motion_metrics(pred, truth, 2.)['dynamic_window_count'] == 0
    assert ev.motion_metrics(pred, truth, 2.)['stopped_dynamic_window_fraction'] is None


def test_failure_coverage_retained_and_baselines_use_same_support():
    truth = np.ones((2, 33, 2))
    pred = truth.copy(); pred[1, 5:] = np.nan
    records = [dict(energy_band=0), dict(energy_band=1)]
    result = ev.summarize_horizon(pred, truth, truth, np.array([32, 4]), records, 32, 1.,
                                   dict(mass=1., length=1., gravity=9.81))
    assert result['groups']['all']['coverage'] == .5
    assert result['groups']['all']['metrics']['angle_mae_rad'] == 0
    assert result['groups']['energy_band1']['metrics']['angle_mae_rad'] is None
    assert not result['ranking_eligible']
    assert result['groups']['energy_band1']['baselines_matched_support']['oracle_hold']['angle_mae_rad'] is None


def make_data(path, count=448):
    frames = 81
    phase = np.tile(np.arange(frames, dtype=np.float64), (count, 1)) + (np.arange(count) % 2)[:, None] * 2
    theta = .01 * phase
    with h5py.File(path, 'w') as f:
        f.attrs.update(training_scope='undamped_stage_a', renderer='smooth_v1', mass=1., length=1., gravity=9.81)
        f['ep_offset'] = np.arange(count) * frames
        f['ep_len'] = np.full(count, frames)
        f['group_id'] = np.arange(count)
        f['energy_band'] = np.arange(count) % 4
        f['state'] = np.stack((theta, np.full_like(theta, .1)), -1).reshape(-1, 2)
        f['pixels'] = np.broadcast_to(phase.astype(np.uint8)[..., None, None, None], (count, frames, 1, 1, 3)).reshape(-1, 1, 1, 3)
        f['action'] = np.zeros((count * frames, 1), np.float32)
        f['dt_frame'] = np.full((count * frames, 1), .1, np.float32)
        if count == 448:
            f['splits/train'] = np.arange(384)
            f['splits/val'] = np.arange(384, 448)
        # Intentionally NO splits/test exists: default evaluation must not open it.


def pixels_tensor(values, device):
    return torch.as_tensor(np.array(values).copy(), device=device).permute(0, 3, 1, 2).float()


def fake_autonomous(model, images, past_actions, past_dt, actions, dt, diagnostic_names):
    assert images.shape[1] == 21 and past_actions.shape[1] == past_dt.shape[1] == 20
    assert actions.shape == dt.shape
    b, steps = actions.shape[:2]
    result = np.zeros((b, steps + 1, 2 * model.q_dim))
    result[..., 0] = images[:, -1, 0, 0, 0].cpu().numpy()[:, None] * .01 + np.arange(steps + 1)[None] * .01
    result[..., model.q_dim] = .1
    return dict(features=result, latent_energy=np.ones((b, steps + 1)),
                valid_steps=np.full(b, steps), failures=[], diagnostics={})


def test_future_pixels_cannot_change_autonomous_features(tmp_path):
    path = tmp_path / 'data.h5'; make_data(path, count=2)
    records = [dict(episode=0), dict(episode=1)]
    model = SimpleNamespace(q_dim=2)
    first = ev.collect_rollouts(model, path, records, 32, 32, 'cpu', fake_autonomous, pixels_tensor, ())
    with h5py.File(path, 'r+') as f:
        for ep in range(2):
            start = int(f['ep_offset'][ep]); f['pixels'][start + 21:start + 81] = 255
    changed = ev.collect_rollouts(model, path, records, 32, 32, 'cpu', fake_autonomous, pixels_tensor, ())
    np.testing.assert_array_equal(first['features'], changed['features'])


def test_full_val_pipeline_never_opens_test_and_never_creates_optimizer(tmp_path, monkeypatch):
    data = tmp_path / 'data.h5'; make_data(data)
    class Model(torch.nn.Module):
        def __init__(self, config):
            super().__init__(); self.bias = torch.nn.Parameter(torch.zeros(1)); self.q_dim = config['q_dim']
    def calibration(model, path, pairs, batch_size):
        with h5py.File(path, 'r') as f:
            states = np.array([f['state'][int(f['ep_offset'][ep]) + t] for ep, t in pairs])
        features = np.zeros((len(pairs), 2 * model.q_dim))
        features[:, 0], features[:, model.q_dim] = states.T
        targets = np.stack((np.sin(states[:, 0]), np.cos(states[:, 0]), states[:, 1]), -1)
        return features, targets
    monkeypatch.setattr(ev, 'evaluation_sources', lambda *a: {})
    monkeypatch.setattr(ev, 'load_frozen_helpers', lambda *a: (Model, calibration, fake_autonomous,
                                                            pixels_tensor, lambda: {'core': 'fixed'}, ()))
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: pytest.fail('No model optimizer allowed'))
    saved = saved_checkpoint(2); saved['dataset_sha256'] = ev.sha256(data)
    saved['state_dict'] = Model(saved['config']).state_dict()
    checkpoint = tmp_path / 'final.pt'; torch.save(saved, checkpoint)
    output = tmp_path / 'result'
    result = ev.main(['--checkpoint', str(checkpoint), '--data', str(data), '--output', str(output),
                      '--device', 'cpu', '--threads', '1', '--expected-q-dim', '2'])
    assert result['status'] == 'completed' and result['test_split_accessed'] is False
    assert result['world_model_optimizer_created'] is False and result['world_model_updates'] == 0
    assert result['model_hash_before'] == result['model_hash_after']
    assert result['calibration']['fit_groups'] == 288 and result['calibration']['selection_groups'] == 96
    assert set(result['evaluation']) == {'val'}
    assert result['evaluation']['val']['episode_count'] == 64
    assert set(result['evaluation']['val']['rollout']) == {'8', '32'}
    with np.load(output / 'val/trajectories.npz') as arr:
        assert arr['features'].shape == (64, 33, 4)
        assert arr['teacher_features'].shape == (64, 33, 4)
        assert arr['valid_steps'].min() == 32
    # Arrays are prepared for the same true/teacher/autonomous visualization.
    with np.load(output / 'val/animation_inputs.npz') as arr:
        assert arr['truth'].shape == arr['autonomous_prediction'].shape == (4, 33, 2)
    assert json.loads((output / 'summary.json').read_text())['calibration']['weights'] == 'uniform'
