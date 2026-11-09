"""CPU fixtures for official-core identity, causality, targets and gradients."""
import json

import pytest
import torch
from torch.nn import functional as F

from dpbench.comparison_visual_baselines import (
    HISTORY, LeHamiComparison, LeWMComparison, build_visual_baseline, default_model_config,
)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def batch(horizon=2, seed=411):
    g = torch.Generator().manual_seed(seed)
    frames = HISTORY+horizon+2
    return {"pixels": torch.rand(2, frames, 3, 64, 64, generator=g),
            "action": torch.zeros(2, frames-1, 1),
            "dt_frame": torch.full((2, frames-1, 1), .1)}


def tiny(kind):
    return build_visual_baseline(kind, default_model_config(kind, tiny=True))


def twins(kind):
    torch.manual_seed(92)
    cfg = default_model_config(kind, tiny=True)
    if kind == "lewm":
        cfg["predictor"]["dropout"] = .1
    left = build_visual_baseline(kind, cfg)
    right = build_visual_baseline(kind, cfg)
    right.load_state_dict(left.state_dict(), strict=True)
    return left, right


def test_real_presets_use_correct_official_dimensions_and_objectives():
    a, b = default_model_config("lehami"), default_model_config("lewm")
    assert (a["latent_dim"], b["latent_dim"]) == (32, 64)
    assert a["sigreg_mode"] == "joint" and a["momentum_weight"] == 1
    assert (a["sigreg"]["weight"], b["sigreg"]["weight"]) == (.25, .09)
    assert b["objective"] == "native_one_step_shifted_contexts"
    assert a["history"] == b["history"] == 21
    assert not a["tiny_fixture"] and not b["tiny_fixture"]


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_official_core_identity_and_json_provenance(kind):
    model = tiny(kind)
    assert type(model.model).__module__ == "jepa"
    assert type(model.sigreg).__module__ == "module"
    expected = "lehami.ham_flow" if kind == "lehami" else "module"
    assert type(model.model.predictor).__module__ == expected
    assert type(model.model.encoder).__name__ == "ConvEncoder"
    p = model.provenance()
    assert p["physical_labels_used_by_world_model"] is False
    assert p["tiny_fixture"] is True
    assert len(p["source_manifest_sha256"]) == 64
    json.dumps(p)
    if kind == "lehami":
        assert not any("damping" in name for name, _ in model.named_parameters())
    else:
        assert "not" in model.__doc__.lower() or "adaptation" in model.__doc__.lower()


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_native_losses_backpropagate_and_never_require_physical_labels(kind):
    torch.manual_seed(13)
    model = tiny(kind).train()
    data = batch()
    captured = []
    hook = model.sigreg.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    out = model.loss(data, horizon=2)
    hook.remove()
    assert all(torch.isfinite(v).all() for v in out.values())
    torch.testing.assert_close(out["loss"], out["prediction_loss"] + model.config["sigreg"]["weight"]*out["sigreg_loss"])
    out["loss"].backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    assert any(g.abs().sum() > 0 for n, g in grads.items() if "encoder" in n)
    assert any(g.abs().sum() > 0 for n, g in grads.items() if "predictor" in n)
    if kind == "lehami":
        # Joint, differentiable targets: this catches the previous q-only arm.
        assert captured[0].shape[-1] == 2*model.latent_dim
        assert captured[0].requires_grad
        assert captured[0][..., model.latent_dim:].abs().sum() > 0
        assert model.history_initializer[-1].weight.grad.abs().sum() > 0
        torch.testing.assert_close(out["prediction_loss"], out["position_loss"]+out["momentum_loss"])
    else:
        assert captured[0].shape[-1] == model.latent_dim


def test_lehami_zero_history_correction_matches_official_causal_sg():
    model = tiny("lehami").eval()
    from lehami.derivative import p_causal_last
    data = batch()
    history = data["pixels"][:, :21]
    with torch.no_grad():
        z = model._encode(history)
        features = model.observed_features(history, data["action"][:, :20])
    torch.testing.assert_close(features[:, :model.latent_dim], z[:, -1])
    torch.testing.assert_close(features[:, model.latent_dim:], p_causal_last(z[:, -4:], 1.))


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_full_history_can_affect_first_autonomous_prediction(kind):
    torch.manual_seed(71)
    model = tiny(kind).eval()
    with torch.no_grad():
        if kind == "lehami":
            model.history_initializer[-1].weight.fill_(.2)
        else:
            # Native AdaLN-zero starts attention gates at zero; make attention
            # active to test architecture visibility rather than its zero init.
            for block in model.model.predictor.transformer.layers:
                block.adaLN_modulation[-1].bias.fill_(.3)
    data = batch()
    a = data["pixels"][:, :21].clone()
    b = a.clone()
    b[:, 0] = 1-a[:, 0]
    action = data["action"][:, :2]
    pa = data["action"][:, :20]
    first = model.rollout_features(a, action, pa)[:, 1]
    changed = model.rollout_features(b, action, pa)[:, 1]
    assert not torch.equal(first, changed)


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_training_first_prediction_has_no_future_batchnorm_leak(kind):
    left, right = twins(kind)
    data = batch(horizon=3)
    altered = data["pixels"].clone()
    altered[:, 21:] = 1-altered[:, 21:]
    if kind == "lehami":
        # Encode every target frame to exercise all running-stat updates too.
        za = left._encode(data["pixels"])
        zb = right._encode(altered)
        init_a = left._initial(za[:, :21], data["action"][:, :20])[0]
        init_b = right._initial(zb[:, :21], data["action"][:, :20])[0]
        a = left._rollout(init_a, data["action"][:, 20:21])[:, 1]
        b = right._rollout(init_b, data["action"][:, 20:21])[:, 1]
    else:
        za = left._encode(data["pixels"][:, :24])
        zb = right._encode(altered[:, :24])
        torch.manual_seed(381)
        a = left._teacher_predictions(za, data["action"], 3)[:, 0, -1]
        torch.manual_seed(381)
        b = right._teacher_predictions(zb, data["action"], 3)[:, 0, -1]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_lewm_one_step_objective_matches_original_core_and_ignores_tail():
    model = tiny("lewm").eval()
    data = batch(horizon=2)
    z = model._encode(data["pixels"][:, :23])
    losses = []
    for j in range(2):
        pred = model.model.predict(z[:, j:j+21], model.model.action_encoder(data["action"][:, j:j+21]))
        losses.append(F.mse_loss(pred, z[:, j+1:j+22]))
    torch.manual_seed(131)
    original = model.loss(data, 2)
    torch.testing.assert_close(original["prediction_loss"], torch.stack(losses).mean())
    data["pixels"][:, -2:] = 0
    # Physical labels cannot affect either objective or initialization.
    data.update(state=torch.randn(2, 25, 2), damping=torch.randn(2), energy=torch.randn(2, 25))
    torch.manual_seed(131)
    changed = model.loss(data, 2)
    torch.testing.assert_close(original["loss"], changed["loss"], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_inference_k0_anchor_matches_observed_and_does_not_update(kind):
    model = tiny(kind).eval()
    data = batch()
    history, actions, pa = data["pixels"][:, :21], data["action"][:, :2], data["action"][:, :20]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    actual = model.rollout_features(history, actions, pa)
    expected = model.observed_features(history, pa)
    assert actual.shape == (2, 3, model.feature_dim)
    torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)
    zero = model.rollout_features(history, actions[:, :0], pa)
    torch.testing.assert_close(zero[:, 0], expected, rtol=0, atol=0)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_contract_rejects_bad_clock_history_old_q_only_and_changed_source_pin(monkeypatch):
    model = tiny("lehami")
    data = batch()
    data["dt_frame"][:, 4] = .05
    with pytest.raises(ValueError, match="intervals"):
        model.loss(data, 2)
    with pytest.raises(RuntimeError, match="eval"):
        model.observed_features(data["pixels"][:, :21])
    model.eval()
    with pytest.raises(ValueError, match="21 RGB"):
        model.observed_features(data["pixels"][:, :5])
    cfg = default_model_config("lehami", tiny=True)
    cfg["sigreg_mode"] = "q"
    with pytest.raises(ValueError, match="joint"):
        LeHamiComparison(cfg)
    import dpbench.comparison_visual_baselines as mod
    monkeypatch.setattr(mod, "PINNED_MANIFEST_SHA256", "0"*64)
    with pytest.raises(RuntimeError, match="manifest changed"):
        tiny("lewm")


def test_loaded_upstream_module_collision_is_rejected(monkeypatch):
    tiny("lewm")
    import module as original
    monkeypatch.setattr(original, "__file__", "/unverified/module.py")
    with pytest.raises(RuntimeError, match="module name collision"):
        tiny("lehami")
