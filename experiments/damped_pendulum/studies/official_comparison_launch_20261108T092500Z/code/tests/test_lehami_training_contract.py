"""Paired-objective data/selection boundaries; no optimizer or training."""
import copy
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from dpbench.dissipative_data import (
    DissipativeWindowDataset, StepBatchSampler, fixed_validation_indices,
    preflight_dataset,
)
from dpbench.train_dissipative import DEFAULT_CONFIG, preflight, read_config, validate_config, validation
from tests.test_dissipative_data import data_path
from tests.test_dissipative_training import tiny_data
from tests.test_stall_validation import GATE, MotionDataset, MotionModel


def phase_config():
    config = read_config(DEFAULT_CONFIG)
    config['model']['target_tail_frames'] = 2
    config['training'].update(validation_score='position_momentum', validation_motion_weight=0.)
    return config


def test_tail_frames_fit_prefix_and_every_rollout_endpoint_has_two_right_neighbors(data_path):
    with h5py.File(data_path, 'r+') as handle:
        # Nonpermitted prefixes and test transitions are deliberately unusable.
        for episode in range(12):
            start = episode * 24
            handle['action'][start + 20:start + 24] = np.nan
            handle['dt_frame'][start + 20:start + 24] = -1
        handle['action'][10 * 24:] = np.nan
        handle['dt_frame'][10 * 24:] = -1
        handle.create_dataset('state', data=np.full((12 * 24, 2), np.nan))
    data = DissipativeWindowDataset(data_path, 'train', history=5, max_horizon=7,
                                    target_tail_frames=2)
    assert data.window == 14 and len(data) == 8 * 7
    for i, ref in enumerate(data.index):
        for horizon in (1, 7):
            sample = data.get_window(i, horizon=horizon)
            frames = 5 + horizon + 2
            assert sample['pixels'].shape[0] == frames
            assert sample['action'].shape == sample['dt_frame'].shape == (frames - 1, 1)
            assert ref.start_local + frames <= 20
            local_times = sample['pixels'][:, 1, 0, 0] * 255
            torch.testing.assert_close(local_times, torch.arange(ref.start_local,
                                                                 ref.start_local + frames).float())
            endpoint = 5 + horizon - 1
            assert local_times[endpoint + 2] == ref.start_local + endpoint + 2
            assert set(sample) == {'pixels', 'action', 'dt_frame'}
    report = preflight_dataset(data_path, history=5, max_horizon=7, target_tail_frames=2)
    assert report['target_tail_frames'] == 2 and report['window_frames'] == 14
    assert report['target_fields_read'] == [] and not report['test_samples_read']
    data.close()
    with pytest.raises(ValueError, match='target_tail_frames'):
        DissipativeWindowDataset(data_path, 'train', history=5, max_horizon=14,
                                 target_tail_frames=2)


@pytest.mark.parametrize('bad', [True, 1, -1, 2., 3, None])
def test_invalid_tail_cannot_silently_change_indexing(data_path, bad):
    with pytest.raises(ValueError, match='target_tail_frames'):
        DissipativeWindowDataset(data_path, 'train', max_horizon=7, target_tail_frames=bad)


def test_default_sampling_is_unchanged_and_pair_sampling_matches(data_path):
    plain = DissipativeWindowDataset(data_path, 'train', max_horizon=7)
    explicit = DissipativeWindowDataset(data_path, 'train', max_horizon=7, target_tail_frames=0)
    assert plain.index == explicit.index and plain.strata == explicit.strata
    left = DissipativeWindowDataset(data_path, 'train', max_horizon=7, target_tail_frames=2)
    right = DissipativeWindowDataset(data_path, 'train', max_horizon=7, target_tail_frames=2)
    for a, b in ((plain, explicit), (left, right)):
        for step in (0, 999, 1000, 3999, 19999):
            ia, ib = StepBatchSampler(a, 32, 0).indices(step), StepBatchSampler(b, 32, 0).indices(step)
            assert ia == ib
            for key in ('pixels', 'action', 'dt_frame'):
                torch.testing.assert_close(a.batch(ia)[key], b.batch(ib)[key], atol=0, rtol=0)
    lv = DissipativeWindowDataset(data_path, 'val', max_horizon=7, target_tail_frames=2)
    rv = DissipativeWindowDataset(data_path, 'val', max_horizon=7, target_tail_frames=2)
    assert fixed_validation_indices(lv, 12, 1009) == fixed_validation_indices(rv, 12, 1009)
    for dataset in (plain, explicit, left, right, lv, rv):
        dataset.close()


def test_config_counts_supervision_tail_and_disallows_ambiguous_selection():
    config = phase_config()
    assert validate_config(config) is config
    # 5 + 32 fits 37, but two derivative target frames do not.
    config['training']['train_frames'] = 37
    with pytest.raises(ValueError, match='target_tail_frames'):
        validate_config(config)
    for key, value, match in [('validation_score', 'loss', 'validation_score'),
                               ('validation_motion_weight', 1., 'validation_motion_weight')]:
        bad = phase_config()
        bad['training'][key] = value
        with pytest.raises(ValueError, match=match):
            validate_config(bad)
    bad = phase_config()
    bad['model'].pop('target_tail_frames')
    with pytest.raises(ValueError, match='target_tail_frames=2'):
        validate_config(bad)


def test_tail_preflight_reports_actual_sample_span_without_optimizer(tiny_data, monkeypatch):
    monkeypatch.setattr(torch.optim, 'AdamW', lambda *a, **kw: pytest.fail('No optimizer permitted'))
    config = phase_config()
    config['training']['train_frames'] = 41
    report = preflight(tiny_data, config)
    assert report['window_frames'] == 39 and report['target_tail_frames'] == 2
    assert report['validation_score'] == 'position_momentum'
    assert not report['optimizer_created'] and not report['test_samples_read']
    assert report['validation_horizons_seconds'] == [.8, 3.2]


class PhaseDataset:
    index = [SimpleNamespace(mode=0, time_bin=0 if i < 7 else 1) for i in range(9)]

    def batch(self, indices, horizon):
        return dict(pixels=torch.tensor(indices, dtype=torch.float64),
                    action=torch.zeros(len(indices), 1), dt_frame=torch.ones(len(indices), 1))


class PhaseModel(torch.nn.Module):
    config = dict(variance_floor=1e-4)

    def loss(self, batch, horizon):
        values = batch['pixels'].double()
        q = values.square().mean() + horizon
        p = values.mean() * .1
        scalar = values.new_tensor
        return dict(prediction_mse=q, persistence_mse=scalar(100.),
                    target_mean_0=scalar(0.), target_second_moment_0=scalar(4.),
                    target_motion_mse=scalar(.01),
                    phase_position_mse=q, phase_momentum_mse=p, phase_prediction_mse=q+p,
                    # It must never select by training objective or stochastic regularizer.
                    loss=scalar(1e8), sigreg_loss=scalar(1e9))


def test_phase_validation_pools_batch_means_then_equal_weights_groups_and_horizons():
    config = phase_config()
    config['training']['val_horizons'] = [2, 4]
    model, dataset = PhaseModel(), PhaseDataset()
    results = []
    for batch_size in (2, 5):
        config['training']['batch_size'] = batch_size
        results.append(validation(model, dataset, list(range(9)), config, 'cpu'))
    expected_q = np.mean([np.mean(np.arange(7) ** 2), np.mean(np.arange(7, 9) ** 2)]) + 3
    expected_p = .1 * np.mean([np.mean(np.arange(7)), np.mean(np.arange(7, 9))])
    for result in results:
        assert result['gate_passed']
        assert result['phase_position_mse'] == pytest.approx(expected_q)
        assert result['phase_momentum_mse'] == pytest.approx(expected_p)
        assert result['score'] == pytest.approx(expected_q + expected_p)
        assert len(result['groups']) == 4
        assert result['score'] != pytest.approx(np.mean(np.arange(9)**2) + 3 + .1*np.arange(9).mean())


def test_phase_selection_still_rejects_stopped_motion_even_with_low_phase_mse():
    class PhaseMotionModel(MotionModel):
        def loss(self, batch, horizon):
            metrics = super().loss(batch, horizon)
            z = metrics['prediction_mse'].new_zeros(())
            metrics.update(phase_position_mse=z, phase_momentum_mse=z, phase_prediction_mse=z)
            return metrics
    config = phase_config()
    config['training'].update(prediction_motion_gate=copy.deepcopy(GATE), val_horizons=[8])
    data = MotionDataset([1., 1.], [0., 0.])
    report = validation(PhaseMotionModel(), data, [0, 1], config, 'cpu')
    assert report['score'] == 0 and not report['gate_passed']
    row = next(iter(report['groups'].values()))
    assert row['representation_gate_passed'] and not row['prediction_gate_passed']


@pytest.mark.parametrize('problem', ['missing', 'negative', 'inconsistent'])
def test_phase_metrics_must_be_present_nonnegative_and_consistent(problem):
    class BadModel(PhaseModel):
        def loss(self, batch, horizon):
            metrics = super().loss(batch, horizon)
            if problem == 'missing':
                metrics.pop('phase_momentum_mse')
            elif problem == 'negative':
                metrics['phase_momentum_mse'] = torch.tensor(-1.)
            else:
                metrics['phase_prediction_mse'] = torch.tensor(0.)
            return metrics
    with pytest.raises(ValueError, match='phase|Phase'):
        validation(BadModel(), PhaseDataset(), list(range(9)), phase_config(), 'cpu')
