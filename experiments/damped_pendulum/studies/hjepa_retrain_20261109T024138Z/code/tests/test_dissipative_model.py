"""No fitting: causality, differentiation, units and checkpoint interface checks."""
import copy

import pytest
import torch

from dpbench.dissipative_model import DissipativeWorldModel, resolve_model_config
from dpbench.models import load_checkpoint


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(407)
    return DissipativeWorldModel(dict(encoder_width=8, sigreg_projections=8,
                                      core=dict(hidden_dim=8, substeps=1)))


def batch(horizon=2, history=5):
    generator = torch.Generator().manual_seed(42)
    return dict(pixels=torch.rand(2, history + horizon, 3, 64, 64, generator=generator),
                action=torch.randn(2, history - 1 + horizon, 1, generator=generator) * .1,
                dt_frame=torch.full((2, history - 1 + horizon, 1), .1))


def test_future_and_physical_labels_do_not_enter_rollout_in_train_or_eval(model):
    original = batch()
    changed = {key: value.clone() for key, value in original.items()}
    changed["pixels"][:, 5:] = 0
    changed["state"] = torch.full((2, 7, 2), 1e20)
    changed["energy"] = torch.full((2, 7), -1e20)
    for training in (True, False):
        model.train(training)
        with torch.no_grad():
            before = model.predict_batch(original, 2)
            after = model.predict_batch(changed, 2)
        assert torch.equal(before["q"], after["q"])
        assert torch.equal(before["p"], after["p"])
        assert not torch.equal(before["target"], after["target"])
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())


def test_loss_backward_has_finite_gradients_and_never_updates_parameters(model):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    values = model.loss(batch(), horizon=2)
    assert all(v.ndim == 0 and torch.isfinite(v) for v in values.values())
    values["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.core.damping_net[-1].weight.grad.abs().sum() > 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert "energy_residual" in values and "solver_residual" in values


def test_causal_quadratic_velocity_uses_seconds_and_irregular_intervals(model):
    dt = torch.tensor([.05, .1, .15, .1], dtype=torch.float32)[None, :, None]
    times = torch.cat([torch.zeros(1, 1), dt[..., 0].cumsum(1)], 1).double()
    v0 = torch.tensor([.3, -.4], dtype=torch.float64)
    acc = torch.tensor([.5, .2], dtype=torch.float64)
    q = 1 + times[..., None] * v0 + .5 * times[..., None].square() * acc
    model.encode_frames = lambda _: q.float()
    actual_q, actual_p, _, _ = model.initialize(torch.zeros(1, 5, 3, 64, 64),
                                              torch.zeros(1, 4, 1), dt)
    torch.testing.assert_close(actual_q, q[:, -1], atol=1e-7, rtol=0)
    torch.testing.assert_close(actual_p, v0 + times[:, -1:] * acc, atol=3e-6, rtol=0)


def test_parameter_groups_and_checkpoint_readout_roundtrip(model, tmp_path):
    perception = {id(p) for p in model.encoder_parameters()}
    dynamics = {id(p) for p in model.dynamics_parameters()}
    assert not perception & dynamics
    assert perception | dynamics == {id(p) for p in model.parameters()}
    path = tmp_path / "model.pt"
    torch.save(dict(kind="dissipative", config=copy.deepcopy(model.config),
                    state_dict=model.state_dict()), path)
    restored, _ = load_checkpoint(path)
    b = batch()
    model.eval()
    args = (b["pixels"][:, :5], b["action"][:, 4:])
    expected = model.rollout_features(*args, past_actions=b["action"][:, :4])
    actual = restored.rollout_features(*args, past_actions=b["action"][:, :4])
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert actual.shape == (2, 3, 4)


def test_inference_inputs_are_explicit(model):
    model.eval()
    b = batch()
    with pytest.raises(ValueError, match="past_actions"):
        model.rollout_features(b["pixels"][:, :5], b["action"][:, 4:])
    b["dt_frame"][0, 0] = 0
    with pytest.raises(ValueError, match="positive"):
        model.loss(b, 2)


@pytest.fixture
def contextual_model():
    torch.set_num_threads(2)
    torch.manual_seed(411)
    return DissipativeWorldModel(dict(history=21, velocity_history=5, environment_context_dim=4,
                                      encoder_width=8, sigreg_projections=8,
                                      core=dict(hidden_dim=8, substeps=1)))


def test_environment_context_is_causal_and_frozen_during_rollout(contextual_model):
    model = contextual_model
    original = batch(history=21)
    changed = {key: value.clone() for key, value in original.items()}
    changed["pixels"][:, 21:] = 0
    changed["damping"] = torch.full((2,), 1e20)
    changed["b"] = torch.full((2,), -1e20)
    changed["state"] = torch.full((2, 23, 2), 1e20)
    seen = []
    hook = model.core.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen.append(kwargs["context"]), with_kwargs=True)
    try:
        for training in (True, False):
            model.train(training)
            seen.clear()
            with torch.no_grad():
                before = model.predict_batch(original, 2)
                after = model.predict_batch(changed, 2)
            assert before["context"].shape == (2, 4)
            assert seen[0] is seen[1] is before["context"]
            assert torch.equal(before["context"], after["context"])
            assert torch.equal(before["q"], after["q"])
            assert torch.equal(before["p"], after["p"])
            assert not torch.equal(before["target"], after["target"])
    finally:
        hook.remove()
    older_history_changed = {key: value.clone() for key, value in original.items()}
    older_history_changed["pixels"][:, :16] = 0
    with torch.no_grad():
        older = model.predict_batch(older_history_changed, 2)
    assert not torch.equal(before["context"], older["context"])
    # Initial q and local polynomial p see the identical final five frames;
    # the correction head is zero-initialized, but context sees all 21.
    assert torch.equal(before["q"][:, 0], older["q"][:, 0])
    assert torch.equal(before["p"][:, 0], older["p"][:, 0])


def test_context_encoder_receives_prediction_gradients_without_updates(contextual_model):
    model = contextual_model
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = model.predict_batch(batch(history=21), 2)
    prediction_loss = (result["q"][:, 1:] - result["target"][:, 1:]).square().mean()
    prediction_loss.backward()
    context_parameters = list(model.environment_encoder.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               and p.grad.abs().sum() > 0 for p in context_parameters)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    perception = {id(p) for p in model.encoder_parameters()}
    dynamics = {id(p) for p in model.dynamics_parameters()}
    assert not perception & dynamics
    assert perception | dynamics == {id(p) for p in model.parameters()}
    assert {id(p) for p in context_parameters} <= dynamics
    clone = DissipativeWorldModel(model.config)
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        expected = model.predict_batch(batch(history=21), 2)
        actual = clone.predict_batch(batch(history=21), 2)
    torch.testing.assert_close(expected["context"], actual["context"], atol=0, rtol=0)
    torch.testing.assert_close(expected["q"], actual["q"], atol=0, rtol=0)


def test_long_context_keeps_velocity_fit_local(contextual_model):
    model = contextual_model
    dt = torch.full((1, 20, 1), .1)
    local_times = torch.arange(5, dtype=torch.float64) * .1
    v0 = torch.tensor([.3, -.4], dtype=torch.float64)
    acceleration = torch.tensor([.5, .2], dtype=torch.float64)
    local_q = (1 + local_times[:, None] * v0
               + .5 * local_times[:, None].square() * acceleration)
    # Older samples deliberately disagree with the local quadratic.
    encoded = torch.cat([torch.full((16, 2), 1000., dtype=torch.float64), local_q], 0)[None]
    model.encode_frames = lambda _: encoded.float()
    q, p, _, _ = model.initialize(torch.zeros(1, 21, 3, 64, 64), torch.zeros(1, 20, 1), dt)
    torch.testing.assert_close(q, local_q[-1:], atol=1e-7, rtol=0)
    torch.testing.assert_close(p, (v0 + .4 * acceleration)[None], atol=3e-6, rtol=0)


def test_old_configuration_defaults_and_history_validation(model):
    old_config = copy.deepcopy(model.config)
    old_config.pop("environment_context_dim")
    old_config.pop("velocity_history")
    restored = DissipativeWorldModel(old_config)
    restored.load_state_dict(model.state_dict())
    assert restored.environment_context_dim == 0 and restored.velocity_history == 5
    assert restored.environment_encoder is None and restored.core.context_dim == 0
    assert resolve_model_config(dict(history=3, velocity_history=3))["history"] == 3
    for invalid in (dict(history=2), dict(history=3, velocity_history=5),
                    dict(velocity_history=2), dict(environment_context_dim=-1)):
        with pytest.raises(ValueError):
            resolve_model_config(invalid)
