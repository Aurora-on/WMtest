"""No optimizer: independently check the LeHami-style objective contract."""
import pytest
import torch

from dpbench.dissipative_model import DissipativeWorldModel, resolve_model_config


def make_model(**options):
    torch.set_num_threads(2)
    torch.manual_seed(11903)
    config = dict(
        encoder_width=8, sigreg_projections=8,
        objective_mode="lehami", target_tail_frames=2,
        momentum_reference_time=.1, motion_weight=0., momentum_weight=1.,
        core=dict(hidden_dim=8, substeps=1, damping_mode="none", potential_mode="raw"),
    )
    config.update(options)
    return DissipativeWorldModel(config)


def image_batch(horizon=3, history=5, tail=2):
    frames = history + horizon + tail
    generator = torch.Generator().manual_seed(11907)
    return dict(
        pixels=torch.rand(2, frames, 3, 64, 64, generator=generator),
        action=torch.zeros(2, frames - 1, 1),
        dt_frame=torch.full((2, frames - 1, 1), .1, dtype=torch.float64),
    )


class ZeroRegularizer(torch.nn.Module):
    def forward(self, value):
        return value.sum() * 0


class MomentumOnlyRegularizer(torch.nn.Module):
    """Probe the joint input without relying on stochastic SIGReg behavior."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, value):
        self.seen = value
        return value[..., value.shape[-1] // 2:].square().mean()


def polynomial_case(irregular=True):
    """A known physical-second derivative, including every final target."""
    history, horizon = 5, 3
    batch = image_batch(horizon, history)
    if irregular:
        intervals = torch.tensor([.05, .13, .07, .2, .08, .17, .1, .11, .09],
                                 dtype=torch.float64)
        batch["dt_frame"] = intervals[None, :, None].expand(2, -1, -1).clone()
    times = torch.cat([torch.zeros(2, 1, dtype=torch.float64),
                       batch["dt_frame"][..., 0].cumsum(1)], 1)
    velocity = torch.tensor([[.3, -.4], [.7, .2]], dtype=torch.float64)
    acceleration = torch.tensor([[.5, .2], [-.3, .6]], dtype=torch.float64)
    encoded = (torch.tensor([1., -2.], dtype=torch.float64)
               + times[..., None] * velocity[:, None]
               + .5 * times[..., None].square() * acceleration[:, None]).requires_grad_()
    true_momentum = velocity[:, None] + times[..., None] * acceleration[:, None]
    target = encoded[:, history - 1:history + horizon]
    predicted_q = target.detach().clone()
    predicted_q[:, 1:] += torch.tensor([.12, -.08], dtype=torch.float64)
    predicted_q.requires_grad_()
    predicted_p = (true_momentum[:, history - 1:history + horizon]
                   + torch.tensor([.4, -.2], dtype=torch.float64)).clone().requires_grad_()
    result = dict(q=predicted_q, p=predicted_p, target=target,
                  all_encoded=encoded[:, :history + horizon],
                  supervision_encoded=encoded,
                  correction=torch.full((2, 2), .03, dtype=torch.float64, requires_grad=True),
                  diagnostics={})
    return batch, result, true_momentum


@pytest.mark.parametrize("irregular", [False, True])
def test_full_future_momentum_uses_centered_quadratic_derivatives_in_seconds(irregular):
    model = make_model()
    batch, result, true_p = polynomial_case(irregular)
    model.predict_batch = lambda *_: result
    model.sigreg = ZeroRegularizer()
    values = model.loss(batch, horizon=3)
    position = (result["q"][:, 1:] - result["target"][:, 1:]).square().mean()
    momentum = (.1 * (result["p"][:, 1:] - true_p[:, 5:8])).square().mean()
    expected = position + momentum + .001 * result["correction"].square().mean()
    torch.testing.assert_close(values["phase_position_mse"], position, atol=1e-12, rtol=0)
    torch.testing.assert_close(values["phase_momentum_mse"], momentum, atol=1e-12, rtol=0)
    torch.testing.assert_close(values["phase_prediction_mse"], position + momentum, atol=1e-12, rtol=0)
    torch.testing.assert_close(values["loss"], expected, atol=1e-12, rtol=0)
    # A loss on all three future points includes the last point's two tail
    # images. This catches either stop-gradient or silent final-point omission.
    gradient = torch.autograd.grad(values["loss"], result["supervision_encoded"])[0]
    assert torch.isfinite(gradient).all()
    assert gradient[:, -1].abs().sum() > 0
    assert gradient[:, -2].abs().sum() > 0


def test_reference_time_scales_both_momentum_loss_and_joint_regularizer():
    batch, result, true_p = polynomial_case()
    outputs = []
    for reference_time in (.1, .2):
        model = make_model(momentum_reference_time=reference_time)
        model.predict_batch = lambda *_: result
        regularizer = MomentumOnlyRegularizer()
        model.sigreg = regularizer
        values = model.loss(batch, 3)
        seen = regularizer.seen
        # Valid five-point centers cover all interior history and predicted
        # frames; the extra two tail frames are support, never target centers.
        assert seen.shape == (6, 2, 4)
        # The pinned regularizer uses FP32 projections; the cast must preserve
        # the SG gradient even though mechanical targets/MSE remain FP64.
        assert seen.dtype == torch.float32
        torch.testing.assert_close(seen[..., :2],
                                   result["supervision_encoded"][:, 2:-2].transpose(0, 1).float())
        torch.testing.assert_close(seen[..., 2:],
                                   (reference_time * true_p[:, 2:-2]).transpose(0, 1).float(),
                                   atol=1e-8, rtol=1e-6)
        outputs.append(values)
        gradient = torch.autograd.grad(values["sigreg_loss"], result["supervision_encoded"],
                                       retain_graph=True)[0]
        assert gradient[:, -1].abs().sum() > 0
        assert torch.isfinite(gradient).all()
    torch.testing.assert_close(outputs[1]["phase_momentum_mse"],
                               4 * outputs[0]["phase_momentum_mse"], atol=1e-12, rtol=0)
    torch.testing.assert_close(outputs[1]["sigreg_loss"],
                               4 * outputs[0]["sigreg_loss"], atol=1e-12, rtol=0)


def test_changing_only_final_tail_frame_changes_final_momentum_supervision():
    model = make_model()
    model.sigreg = ZeroRegularizer()
    batch, result, _ = polynomial_case()
    model.predict_batch = lambda *_: result
    before = model.loss(batch, 3)
    changed = dict(result)
    changed["supervision_encoded"] = result["supervision_encoded"].detach().clone()
    changed["supervision_encoded"][:, -1] += .75
    changed["supervision_encoded"].requires_grad_()
    model.predict_batch = lambda *_: changed
    after = model.loss(batch, 3)
    assert torch.equal(before["phase_position_mse"], after["phase_position_mse"])
    assert not torch.allclose(before["phase_momentum_mse"], after["phase_momentum_mse"])


@pytest.mark.parametrize("training", [True, False])
def test_real_model_future_and_tail_frames_never_change_autonomous_rollout(training):
    model = make_model(velocity_initializer="motion_gated")
    model.train(training)
    batch = image_batch(horizon=2)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["pixels"][:, -2:] = 0
    changed["state"] = torch.full((2, 9, 2), 1e20)
    changed["damping"] = torch.full((2,), -1e20)
    with torch.no_grad():
        original = model.predict_batch(batch, 2)
        tail_changed = model.predict_batch(changed, 2)
        changed["pixels"][:, 5:] = 0
        future_changed = model.predict_batch(changed, 2)
    assert original["all_encoded"].shape == (2, 7, 2)
    assert original["supervision_encoded"].shape == (2, 9, 2)
    for result in (tail_changed, future_changed):
        assert torch.equal(original["q"], result["q"])
        assert torch.equal(original["p"], result["p"])
    assert torch.equal(original["target"], tail_changed["target"])
    assert not torch.equal(original["supervision_encoded"], tail_changed["supervision_encoded"])
    assert not torch.equal(original["target"], future_changed["target"])


def test_current_objective_is_unchanged_when_common_diagnostics_gain_tail_frames():
    batch, result, _ = polynomial_case()
    current = make_model(objective_mode="current", target_tail_frames=0,
                         motion_weight=1., momentum_weight=.1,
                         motion_loss_mode="normalized_velocity")
    with_tail = make_model(objective_mode="current", target_tail_frames=2,
                           motion_weight=1., momentum_weight=.1,
                           motion_loss_mode="normalized_velocity")
    for model in (current, with_tail):
        model.predict_batch = lambda *_: result
        model.sigreg = ZeroRegularizer()
    first, second = current.loss(batch, 3), with_tail.loss(batch, 3)
    for key in ("loss", "prediction_loss", "motion_loss", "momentum_loss", "initializer_loss"):
        assert torch.equal(first[key], second[key]), key
    assert "phase_momentum_mse" in second


def test_tiny_real_three_substep_model_has_finite_backward_without_parameter_update():
    model = make_model(
        velocity_initializer="motion_gated",
        core=dict(hidden_dim=8, damping_mode="none", potential_mode="raw", substeps=3,
                  integrator="frozen_dg", solver="picard", picard_iterations=6),
    )
    batch = image_batch(horizon=2)
    batch["dt_frame"][:, ::2] = .08
    batch["dt_frame"][:, 1::2] = .12
    parameters_before = {key: value.clone() for key, value in model.state_dict().items()}
    values = model.loss(batch, horizon=2)
    assert all(value.ndim == 0 and torch.isfinite(value) for value in values.values())
    values["loss"].backward()
    active = [parameter for parameter in model.parameters() if parameter.grad is not None]
    assert active and all(torch.isfinite(parameter.grad).all() for parameter in active)
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.core.potential_net.parameters())
    assert all(torch.equal(parameters_before[key], value) for key, value in model.state_dict().items())


@pytest.mark.parametrize("overrides", [
    dict(objective_mode="invalid"), dict(objective_mode=[]),
    dict(objective_mode="lehami", target_tail_frames=0, motion_weight=0., momentum_weight=1.),
    dict(target_tail_frames=1), dict(target_tail_frames=True),
    dict(momentum_reference_time=0.), dict(momentum_reference_time=float("nan")),
])
def test_invalid_objective_contract_is_rejected(overrides):
    with pytest.raises(ValueError):
        resolve_model_config(overrides)
