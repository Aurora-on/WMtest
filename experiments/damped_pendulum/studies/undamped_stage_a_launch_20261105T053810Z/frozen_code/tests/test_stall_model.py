"""No optimizer: regression checks for anti-stall objectives and causal inputs."""
import copy
import json
from pathlib import Path

import pytest
import torch

from dpbench.dissipative_model import DissipativeWorldModel, resolve_model_config


def model(**options):
    torch.set_num_threads(2)
    torch.manual_seed(850)
    config = dict(encoder_width=8, sigreg_projections=8,
                  core=dict(hidden_dim=8, substeps=1))
    config.update(options)
    return DissipativeWorldModel(config)


def batch(horizon=2, history=5):
    rng = torch.Generator().manual_seed(853)
    return dict(pixels=torch.rand(2, history + horizon, 3, 64, 64, generator=rng),
                action=torch.zeros(2, history + horizon - 1, 1),
                dt_frame=torch.full((2, history + horizon - 1, 1), .1))


def test_static_observation_zeroes_fitted_roundoff_and_nonzero_correction():
    m = model(velocity_initializer='motion_gated')
    with torch.no_grad():
        m.velocity_correction[-1].bias.fill_(3.)
    one_image = torch.rand(2, 1, 3, 64, 64)
    images = one_image.expand(-1, 5, -1, -1, -1).clone()
    # Simulate convolution roundoff in different batch positions despite
    # exactly identical raw observed images. Encoded equality would fail.
    enc = torch.tensor([2., -3.])[None, None].expand(2, 5, 2).clone()
    enc += torch.arange(5)[None, :, None] * 2e-6
    m.encode_frames = lambda _: enc
    q, p, _, correction = m.initialize(images, torch.ones(2, 4, 1), torch.full((2, 4, 1), .1))
    assert torch.count_nonzero(p) == 0
    assert torch.count_nonzero(correction) == 0
    assert torch.equal(q, enc[:, -1].double())


def test_motion_gated_quadratic_fit_uses_irregular_seconds_and_is_translation_invariant():
    m = model(velocity_initializer='motion_gated')
    dt = torch.tensor([.05, .13, .07, .2], dtype=torch.float64)[None, :, None]
    times = torch.cat([torch.zeros(1, 1), dt[..., 0].cumsum(1)], 1)
    v = torch.tensor([.3, -.4], dtype=torch.float64)
    acceleration = torch.tensor([.5, .2], dtype=torch.float64)
    q = times[..., None] * v + .5 * times[..., None].square() * acceleration
    images = batch()['pixels'][:1, :5]
    m.encode_frames = lambda _: q
    first = m.initialize(images, torch.zeros(1, 4, 1), dt)
    torch.testing.assert_close(first[1], v + times[:, -1:] * acceleration, atol=1e-12, rtol=0)
    with torch.no_grad():
        m.velocity_correction[-1].weight.fill_(.01)
        m.velocity_correction[-1].bias.fill_(.2)
    original = m.initialize(images, torch.zeros(1, 4, 1), dt)
    m.encode_frames = lambda _: q + 1000
    shifted = m.initialize(images, torch.zeros(1, 4, 1), dt)
    torch.testing.assert_close(shifted[1], original[1], atol=2e-10, rtol=0)
    torch.testing.assert_close(shifted[3], original[3], atol=2e-10, rtol=0)


def test_central_momentum_targets_are_aligned_detached_and_exclude_last_frame():
    m = model(momentum_weight=1.)
    b = batch(horizon=3)
    dt = torch.tensor([.05, .13, .07, .2, .08, .17, .1], dtype=torch.float64)[None, :, None]
    b['dt_frame'] = dt.expand(2, -1, -1)
    times = torch.cat([torch.zeros(2, 1), b['dt_frame'][..., 0].cumsum(1)], 1)
    velocity = torch.tensor([.3, -.4], dtype=torch.float64)
    acc = torch.tensor([.5, .2], dtype=torch.float64)
    q = (times[..., None] * velocity + .5 * times[..., None].square() * acc).requires_grad_()
    p = (velocity + times[:, 4:, None] * acc).requires_grad_()
    predicted, target = m._momentum_targets(b, dict(all_encoded=q, p=p))
    assert predicted.shape == target.shape == (2, 3, 2)
    assert not target.requires_grad
    torch.testing.assert_close(target, p[:, :-1], atol=3e-14, rtol=0)
    # Exact static observed neighborhoods override encoder roundoff as well.
    b['pixels'][:] = b['pixels'][:, :1]
    _, target = m._momentum_targets(b, dict(all_encoded=q, p=p))
    assert torch.count_nonzero(target) == 0


class ZeroRegularizer(torch.nn.Module):
    def forward(self, x):
        return x.sum() * 0


def synthetic_result():
    target = torch.tensor([[[0., 0.], [.2, 0.], [.6, 0.]],
                           [[0., 0.], [0., 0.], [0., 0.]]], dtype=torch.float64, requires_grad=True)
    predicted = torch.zeros_like(target, requires_grad=True)
    result = dict(q=predicted, target=target, all_encoded=target,
                  correction=predicted[:, 0], p=predicted, diagnostics={})
    b = batch()
    b['dt_frame'][:, 4:] = torch.tensor([.2, .4])[:, None]
    return b, result


def test_normalized_velocity_loss_and_dynamic_metrics_use_correct_units_and_reductions():
    m = model(motion_loss_mode='normalized_velocity')
    b, result = synthetic_result()
    m.predict_batch = lambda *_: result
    m.sigreg = ZeroRegularizer()
    values = m.loss(b, 2)
    # One stalled moving window, one static window. Speed is1 in one of2 dims.
    torch.testing.assert_close(values['motion_loss'], torch.tensor(.5, dtype=torch.float64))
    assert values['window_count'] == 2
    assert values['dynamic_window_count'] == 1
    assert values['stalled_dynamic_window_count'] == 1
    torch.testing.assert_close(values['target_velocity_power'], torch.tensor(.25, dtype=torch.float64))
    torch.testing.assert_close(values['dynamic_target_velocity_power'], values['target_velocity_power'])
    assert values['predicted_velocity_power'] == values['velocity_cross_power'] == 0
    assert all(value.ndim == 0 and torch.isfinite(value) for value in values.values())
    dt = b['dt_frame'][:, 4:].double()
    pv = result['q'].diff(dim=1) / dt
    tv = result['target'].diff(dim=1) / dt
    expected = ((pv - tv).square().mean(dim=(1, 2))
                / tv.square().mean(dim=(1, 2)).detach().clamp_min(.1 ** 2)).mean()
    expected_grad = torch.autograd.grad(expected, result['target'], retain_graph=True)[0]
    actual_grad = torch.autograd.grad(values['motion_loss'], result['target'])[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


def test_static_floor_keeps_normalized_loss_and_gradients_finite():
    m = model(motion_loss_mode='normalized_velocity')
    b, result = synthetic_result()
    result['target'] = torch.zeros_like(result['target'], requires_grad=True)
    result['q'] = torch.ones_like(result['q'], requires_grad=True) * .05
    m.predict_batch = lambda *_: result
    m.sigreg = ZeroRegularizer()
    values = m.loss(b, 2)
    assert values['dynamic_window_count'] == values['stalled_dynamic_window_count'] == 0
    assert all(torch.isfinite(value) for value in values.values())
    values['loss'].backward()
    assert torch.isfinite(result['target'].grad).all()


def test_signed_velocity_metrics_detect_wrong_direction_with_equal_motion_power():
    m = model(motion_loss_mode='normalized_velocity')
    b, result = synthetic_result()
    result['q'] = -result['target']
    m.predict_batch = lambda *_: result
    m.sigreg = ZeroRegularizer()
    values = m.loss(b, 2)
    torch.testing.assert_close(values['predicted_velocity_power'], values['target_velocity_power'])
    torch.testing.assert_close(values['velocity_cross_power'], -values['target_velocity_power'])
    torch.testing.assert_close(values['velocity_error_power'], 4 * values['target_velocity_power'])
    assert values['stalled_dynamic_window_count'] == 0


def test_momentum_target_with_no_right_neighbor_is_empty():
    m = model(momentum_weight=1.)
    b = batch(horizon=1)
    encoded = torch.zeros(2, 5, 2)
    predicted, target = m._momentum_targets(b, dict(all_encoded=encoded, p=torch.zeros(2, 1, 2)))
    assert predicted.shape == target.shape == (2, 0, 2)


def test_legacy_formula_and_forward_match_explicit_legacy_with_old_config():
    m = model()
    old_config = copy.deepcopy(m.config)
    for key in ('velocity_initializer', 'motion_loss_mode', 'motion_scale_floor', 'momentum_weight'):
        old_config.pop(key)
    old_config['core'].pop('potential_mode')
    clone = DissipativeWorldModel(old_config)
    clone.load_state_dict(m.state_dict())
    b = batch()
    with torch.no_grad():
        first, second = m.predict_batch(b, 2), clone.predict_batch(b, 2)
    for key in ('q', 'p', 'correction', 'target'):
        assert torch.equal(first[key], second[key])
    m.predict_batch = lambda *_: first
    m.sigreg = ZeroRegularizer()
    values = m.loss(b, 2)
    prediction = (first['q'][:, 1:] - first['target'][:, 1:]).square().mean()
    motion = (first['q'].diff(dim=1) - first['target'].diff(dim=1)).square().mean()
    expected = (prediction + m.config['motion_weight'] * motion
                + m.config['sigreg_weight'] * 0 + m.config['initializer_weight'] * first['correction'].square().mean())
    assert torch.equal(values['loss'], expected)
    assert values['momentum_point_count'] == 0


def test_new_targets_cannot_change_rollout_and_backward_does_not_update_parameters():
    m = model(velocity_initializer='motion_gated', motion_loss_mode='normalized_velocity', momentum_weight=.1)
    b = batch()
    changed = {k: v.clone() for k, v in b.items()}
    changed['pixels'][:, 5:] = 0
    changed['state'] = torch.full((2, 7, 2), 1e20)
    for training in (True, False):
        m.train(training)
        with torch.no_grad():
            first, second = m.predict_batch(b, 2), m.predict_batch(changed, 2)
        assert torch.equal(first['q'], second['q'])
        assert torch.equal(first['p'], second['p'])
        assert not torch.equal(first['target'], second['target'])
    before = {k: v.clone() for k, v in m.state_dict().items()}
    values = m.loss(b, 2)
    assert values['momentum_point_count'] == 4  #2samples x(anchor+first future)
    values['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert all(torch.equal(before[k], value) for k, value in m.state_dict().items())


@pytest.mark.parametrize('filename', ['dissipative_passive_fixed_v1.json', 'dissipative_passive_fixed_v1_softplus.json'])
@pytest.mark.parametrize('horizon', [1, 3])
def test_actual_candidate_config_forward_backward_and_future_isolation(filename, horizon):
    path = Path(__file__).resolve().parents[1] / 'configs' / filename
    config = json.loads(path.read_text())['model']
    config.update(encoder_width=8, sigreg_projections=8)
    config['core']['hidden_dim'] = 8
    torch.set_num_threads(2)
    torch.manual_seed(884)
    m = DissipativeWorldModel(config)
    assert m.config['core']['integrator'] == 'frozen_dg'
    assert m.config['velocity_initializer'] == 'motion_gated'
    b = batch(horizon=horizon, history=21)
    # Keep varying dt inside the actual combined model, not only fit tests.
    b['dt_frame'][:, ::2] = .08
    b['dt_frame'][:, 1::2] = .12
    changed = {k: value.clone() for k, value in b.items()}
    changed['pixels'][:, 21:] = 0
    changed['state'] = torch.full((2, 21 + horizon, 2), 1e20)
    for training in (True, False):
        m.train(training)
        with torch.no_grad():
            first = m.predict_batch(b, horizon)
            second = m.predict_batch(changed, horizon)
        assert torch.equal(first['q'], second['q'])
        assert torch.equal(first['p'], second['p'])
        assert torch.equal(first['context'], second['context'])
        assert not torch.equal(first['target'], second['target'])
    before = {k: value.clone() for k, value in m.state_dict().items()}
    values = m.loss(b, horizon)
    assert values['momentum_point_count'] == 2 * horizon
    assert all(value.ndim == 0 and torch.isfinite(value) for value in values.values())
    values['loss'].backward()
    for name, parameter in m.named_parameters():
        if config['core']['potential_mode'] == 'raw' and name == 'core.potential_net.0.4.bias':
            # The final raw-potential additive constant is a gauge: neither
            # force nor DG depends on it, so no prediction gradient is needed.
            assert parameter.grad is None or torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert all(torch.equal(before[k], value) for k, value in m.state_dict().items())


@pytest.mark.parametrize('option', [dict(motion_scale_floor=0), dict(momentum_weight=-1),
                                   dict(velocity_initializer='unknown'), dict(motion_loss_mode='unknown'),
                                   dict(core=dict(potential_mode='unknown')),
                                   dict(velocity_initializer=[]), dict(motion_loss_mode={}),
                                   dict(core=dict(potential_mode=[])), dict(motion_scale_floor=float('nan')),
                                   dict(momentum_weight=float('nan')), dict(motion_scale_floor=True),
                                   dict(momentum_weight=[]), dict(velocity_initializer=None)])
def test_invalid_new_options_are_rejected(option):
    with pytest.raises(ValueError):
        resolve_model_config(option)
