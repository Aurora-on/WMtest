"""H-JEPA source fidelity, causal training boundary and reload checks."""
import copy
import json
import shutil

import pytest
import torch

from dpbench import comparison_hjepa as hj


@pytest.fixture(autouse=True)
def cpu_budget():
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(417)
    yield
    torch.set_num_threads(before)


def make_batch(horizon=2):
    time = 21 + horizon + 2
    return {"pixels": torch.rand(2, time, 3, 64, 64),
            "action": torch.zeros(2, time - 1, 1),
            "dt_frame": torch.full((2, time - 1, 1), .1)}


def test_sources_are_pinned_and_tampering_is_rejected(tmp_path, monkeypatch):
    manifest = hj.verify_hjepa_sources()
    assert manifest["commit"] == hj.COMMIT
    assert manifest["license"] == "MIT"
    copied = tmp_path / "third_party"
    copied.mkdir()
    shutil.copytree(hj.UPSTREAM / "hjepa", copied / "hjepa")
    shutil.copy2(hj.UPSTREAM / "hjepa_source.json", copied / "hjepa_source.json")
    path = copied / "hjepa/hamiltonian_predictor.py"
    path.write_bytes(path.read_bytes() + b"\n# unexpected change\n")
    monkeypatch.setattr(hj, "UPSTREAM", copied)
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        hj.verify_hjepa_sources()


def test_native_losses_have_finite_grads_and_update(tmp_path):
    model = hj.HJEPAComparison(hj.default_hjepa_config(tiny=True)).train()
    batch = make_batch()
    out = model.loss(batch, 2)
    assert {"loss", "prediction_loss", "pic_loss", "iso_loss"} <= set(out)
    torch.testing.assert_close(out["loss"], out["prediction_loss"] + out["pic_loss"] + out["iso_loss"])
    assert all(value.ndim == 0 and torch.isfinite(value) for value in out.values())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    out["loss"].backward()
    for part in (model.encoder, model.history_encoder, model.predictor):
        grads = [p.grad for p in part.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
    param = next(p for p in model.predictor.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    old = param.detach().clone()
    optimizer.step()
    assert not torch.equal(old, param)
    model.eval()
    history = batch["pixels"][:, :21]
    actions = torch.zeros(2, 3, 1)
    before = model.rollout_features(history, actions)
    checkpoint = tmp_path / "local.pt"
    torch.save({"config": model.config, "state_dict": model.state_dict()}, checkpoint)
    restored = torch.load(checkpoint, weights_only=True)
    other = hj.HJEPAComparison(restored["config"]).eval()
    other.load_state_dict(restored["state_dict"], strict=True)
    torch.testing.assert_close(before, other.rollout_features(history, actions), rtol=0, atol=0)
    provenance = json.loads(json.dumps(model.provenance()))
    assert "hjepa_loss" in provenance["executed_official_components"]
    assert provenance["feature_dim"] == 8


def test_training_prediction_has_no_future_image_gradient():
    model = hj.HJEPAComparison(hj.default_hjepa_config(tiny=True)).train()
    batch = make_batch()
    batch["pixels"].requires_grad_(True)
    predicted = []
    hook = model.predictor.register_forward_hook(lambda _m, _a, result: predicted.append(result))
    try:
        out = model.loss(batch, 2)
    finally:
        hook.remove()
    # The first two calls are the native prediction branch; later calls belong
    # to the official observed-context PIC auxiliary loss.
    prediction_grad, = torch.autograd.grad(sum(x.square().sum() for x in predicted[:2]),
                                           batch["pixels"], retain_graph=True)
    assert prediction_grad[:, :21].abs().sum() > 0
    assert prediction_grad[:, 0].abs().sum() > 0  # all21 history frames are used
    assert torch.count_nonzero(prediction_grad[:, 21:]) == 0
    target_grad, = torch.autograd.grad(out["prediction_loss"], batch["pixels"])
    assert target_grad[:, 21:23].abs().sum() > 0  # future frames supervise targets
    assert torch.count_nonzero(target_grad[:, 23:]) == 0  # common SG tail is unused


def test_changed_future_changes_loss_not_autonomous_predictions():
    model = hj.HJEPAComparison(hj.default_hjepa_config(tiny=True)).train()
    batch = make_batch()
    changed = {k: v.clone() for k, v in batch.items()}
    changed["pixels"][:, 21:23] = torch.rand_like(changed["pixels"][:, 21:23])
    seen = []
    hook = model.predictor.register_forward_hook(lambda _m, _a, result: seen.append(result.detach().clone()))
    try:
        out1 = model.loss(batch, 2)
        rollout1 = seen[:2]
        seen.clear()
        out2 = model.loss(changed, 2)
        rollout2 = seen[:2]
    finally:
        hook.remove()
    for a, b in zip(rollout1, rollout2):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.isclose(out1["prediction_loss"], out2["prediction_loss"], atol=1e-8, rtol=1e-6)


def test_inference_anchor_prefix_and_official_transition():
    model = hj.HJEPAComparison(hj.default_hjepa_config(tiny=True)).eval()
    history = make_batch()["pixels"][:, :21]
    past, actions = torch.zeros(2, 20, 1), torch.zeros(2, 4, 1)
    encoded_counts = []
    hook = model.encoder.register_forward_pre_hook(lambda _m, a: encoded_counts.append(a[0].shape[0]))
    try:
        features = model.rollout_features(history, actions, past)
    finally:
        hook.remove()
    assert encoded_counts == [42]  # no image encoding after initialization
    assert features.shape == (2, 5, 8)
    torch.testing.assert_close(features[:, 0], model.observed_features(history, past), rtol=0, atol=0)
    torch.testing.assert_close(features[:, :2], model.rollout_features(history, actions[:, :1], past), rtol=0, atol=0)
    with torch.no_grad():
        s, delta = features[:, 0].chunk(2, -1)
        official_next = model.predictor(s, actions[:, 0], context=torch.cat([s, delta], -1))
    torch.testing.assert_close(features[:, 1, :4], official_next, rtol=0, atol=0)
    torch.testing.assert_close(features[:, 1, 4:], official_next - s, rtol=0, atol=0)
    assert model.rollout_features(history, actions[:, :0], past).shape == (2, 1, 8)
    with torch.no_grad():
        evaluation = model.loss(make_batch(), 2, stage="val")
    assert torch.isfinite(evaluation["loss"])


def test_contract_rejects_malformed_or_privileged_inputs():
    model = hj.HJEPAComparison(hj.default_hjepa_config(tiny=True)).eval()
    batch = make_batch()
    with pytest.raises(ValueError, match="21"):
        model.observed_features(batch["pixels"][:, :20])
    wrong_dt = copy.deepcopy(batch)
    wrong_dt["dt_frame"][:] = .05
    with pytest.raises(ValueError, match="dt_frame"):
        model.loss(wrong_dt, 2)
    with pytest.raises(ValueError, match="all actions"):
        model.rollout_features(batch["pixels"][:, :21], torch.ones(2, 1, 1))
    bad_config = hj.default_hjepa_config(tiny=True)
    bad_config["rank"] = 3
    with pytest.raises(ValueError, match="rank"):
        hj.HJEPAComparison(bad_config)
