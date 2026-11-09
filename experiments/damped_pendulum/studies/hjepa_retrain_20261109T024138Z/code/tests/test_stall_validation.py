"""Prediction gates must reject stopping and wrong motion without training."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dpbench.train_dissipative import (
    DEFAULT_CONFIG, prediction_motion_diagnostics, preflight, read_config,
    validate_config, validation,
)
from tests.test_dissipative_training import tiny_data


GATE = dict(min_rms_ratio=.5, max_rms_ratio=1.5, max_relative_error=.8,
            min_cosine=.5, max_stalled_fraction=.1)


class MotionDataset:
    def __init__(self, target, prediction, time_bin=0):
        self.target = torch.tensor(target, dtype=torch.float64)
        self.prediction = torch.tensor(prediction, dtype=torch.float64)
        self.index = [SimpleNamespace(mode=0, time_bin=time_bin) for _ in target]

    def batch(self, indices, horizon):
        return dict(pixels=torch.stack((self.target[indices], self.prediction[indices]), -1),
                    action=torch.zeros(len(indices), 1), dt_frame=torch.ones(len(indices), 1))


class MotionModel(torch.nn.Module):
    config = dict(variance_floor=1e-4, motion_scale_floor=.1)

    def loss(self, batch, horizon):
        target, prediction = batch['pixels'].double().unbind(-1)
        active = target.abs() > .1
        pp, tt = prediction.square(), target.square()
        error, cross = (prediction-target).square(), prediction*target
        n = len(target)
        return {
            'prediction_mse': target.new_tensor(.01),
            'target_mean_0': target.new_tensor(0.), 'target_second_moment_0': target.new_tensor(1.),
            'target_motion_mse': tt.mean() * .01, 'persistence_mse': target.new_tensor(.5),
            'window_count': target.new_tensor(n), 'dynamic_window_count': active.sum(),
            'stalled_dynamic_window_count': (active & (pp < .01 * tt)).sum(),
            'dynamic_predicted_velocity_power': (pp * active).mean(),
            'dynamic_target_velocity_power': (tt * active).mean(),
            'dynamic_velocity_error_power': (error * active).mean(),
            'dynamic_velocity_cross_power': (cross * active).mean(),
        }


def settings(batch_size=2):
    config = read_config(DEFAULT_CONFIG)
    config['training'].update(batch_size=batch_size, val_horizons=[8],
                              prediction_motion_gate=copy.deepcopy(GATE),
                              validation_motion_weight=1.)
    return config


@pytest.mark.parametrize('predicted', [0., -1., 3.])
def test_stopped_reversed_or_excessive_motion_cannot_be_best(predicted):
    data = MotionDataset([1., 1.], [predicted, predicted])
    result = validation(MotionModel(), data, [0, 1], settings(), torch.device('cpu'))
    assert not result['gate_passed']
    row = next(iter(result['groups'].values()))
    assert row['representation_gate_passed']  # The old target-only gate passes.
    assert not row['prediction_gate_passed']


def test_good_motion_and_settled_targets_pass_without_forcing_motion():
    data = MotionDataset([1., 0.], [1., 0.])
    result = validation(MotionModel(), data, [0, 1], settings(), torch.device('cpu'))
    assert result['gate_passed']
    assert result['motion_score'] == 0
    static = MotionModel().loss(MotionDataset([0., 0.], [0., 0.]).batch([0, 1], 8), 8)
    gate = prediction_motion_diagnostics({k:float(v) for k,v in static.items()}, GATE)
    assert gate['prediction_gate_passed'] and not gate['prediction_gate_applicable']


def test_prediction_gate_checks_late_dynamic_windows_even_if_representation_gate_is_early():
    data = MotionDataset([1., 1.], [0., 0.], time_bin=2)
    data.index[0].time_bin = 0
    config = settings()
    config['training']['validation_gate_time_bins'] = [0]
    data.prediction[0] = 1
    result = validation(MotionModel(), data, [0, 1], config, torch.device('cpu'))
    assert not result['gate_passed']
    late = result['groups']['h8_mode0_time2']
    assert not late['gate_required'] and not late['prediction_gate_passed']


def test_pooling_is_independent_of_batch_partition_and_static_fraction():
    data = MotionDataset([0., 1., 2., 0., 3., 1., 0.], [0., .8, 2.2, 0., 2., 0., 0.])
    indices = list(range(len(data.index)))
    a = validation(MotionModel(), data, indices, settings(2), torch.device('cpu'))
    b = validation(MotionModel(), data, indices[::-1], settings(5), torch.device('cpu'))
    assert a['score'] == pytest.approx(b['score'], abs=1e-12)
    ar, br = next(iter(a['groups'].values())), next(iter(b['groups'].values()))
    for key in ('window_count', 'dynamic_window_count', 'stalled_dynamic_window_count',
                'predicted_target_velocity_rms_ratio', 'relative_velocity_error', 'velocity_cosine'):
        assert ar[key] == pytest.approx(br[key], abs=1e-12)
    assert ar['window_count'] == 7 and ar['dynamic_window_count'] == 4
    assert ar['stalled_dynamic_fraction'] == .25


def test_missing_diagnostics_and_invalid_thresholds_fail_explicitly():
    with pytest.raises(ValueError, match='missing metrics'):
        prediction_motion_diagnostics({}, GATE)
    for key, value in [('min_rms_ratio', 0), ('max_rms_ratio', .9),
                       ('max_relative_error', 1.), ('min_cosine', float('nan')),
                       ('max_stalled_fraction', True)]:
        config = settings()
        config['training']['prediction_motion_gate'][key] = value
        with pytest.raises(ValueError, match='prediction_motion_gate'):
            validate_config(config)
    for bad in ([], {}, True):
        config = settings()
        config['training']['required_renderer'] = bad
        with pytest.raises(ValueError, match='required_renderer'):
            validate_config(config)


def test_entirely_subthreshold_validation_cannot_silently_pass_motion_gate():
    # Target delta .005 passes legacy .001 gate, but velocity .05 < floor .1.
    data = MotionDataset([.05, .05], [0., 0.])
    result = validation(MotionModel(), data, [0, 1], settings(), torch.device('cpu'))
    assert not result['dynamic_validation_support'] and not result['gate_passed']
    row = next(iter(result['groups'].values()))
    assert row['representation_gate_passed'] and not row['prediction_gate_applicable']


def test_new_recipe_rejects_legacy_pixels_before_any_optimizer(tiny_data):
    config = settings()
    config['training'].update(train_frames=41, required_renderer='smooth_v1')
    with pytest.raises(ValueError, match='required_renderer'):
        preflight(tiny_data, config)


def test_new_smooth_data_model_and_validation_integrate_without_optimizer(tmp_path, monkeypatch):
    from dpbench.dissipative_data import DissipativeWindowDataset, fixed_validation_indices
    from dpbench.dissipative_model import DissipativeWorldModel
    from dpbench.passive_family import generate_family
    from dpbench.train_dissipative import main

    def forbidden(*args, **kwargs):
        pytest.fail('This integration check must not construct an optimizer')

    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    root = Path(__file__).resolve().parents[1]
    data_config = json.loads((root/'configs/passive_family_smoke.json').read_text())
    data_config.update(renderer='smooth_v1', history=21, train_frames=25, test_frames=27)
    generate_family(data_config, tmp_path/'data')
    config = read_config(root/'configs/dissipative_passive_fixed_v1.json')
    config['model'].update(encoder_width=8, sigreg_projections=8)
    config['model']['core']['hidden_dim'] = 4
    config['training'].update(train_frames=25, val_horizons=[2], val_windows=6,
                              batch_size=2, curriculum=[dict(until_step=20000, horizon=2)])
    path = tmp_path/'config.json'
    path.write_text(json.dumps(config))
    report = main(['--config',str(path),'--data',str(tmp_path/'data/seen.h5'),
                   '--output',str(tmp_path/'no_run')])
    assert report['renderer'] == 'smooth_v1' and not report['optimizer_created']
    assert not report['test_samples_read'] and not (tmp_path/'no_run').exists()
    data = DissipativeWindowDataset(tmp_path/'data/seen.h5', 'val', history=21,
                                    max_horizon=2, train_frames=25)
    torch.set_num_threads(2)
    torch.manual_seed(68)
    model = DissipativeWorldModel(config['model'])
    before = {k:v.clone() for k,v in model.state_dict().items()}
    try:
        indices = fixed_validation_indices(data, 6)
        result = validation(model, data, indices, config, torch.device('cpu'))
        assert np.isfinite(result['score']) and len(result['groups']) == 6
        assert sum(row['window_count'] for row in result['groups'].values()) == 6
        model.loss(data.batch(indices[:2], horizon=2), 2)['loss'].backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert all(torch.equal(before[k], v) for k,v in model.state_dict().items())
    finally:
        data.close()
