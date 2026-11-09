"""Regression checks at the official-model / experimental-protocol boundary."""
import shutil

import pytest
import torch

from dpbench import models
from dpbench.models import build_model


@pytest.fixture(autouse=True)
def cpu_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    yield
    torch.set_num_threads(previous)


def batch_for(model):
    # Distinct torques identify their physical transition index, not just shape.
    return {
        "pixels": torch.rand(2, model.window, 3, 64, 64),
        "action": torch.arange(model.window).float()[None, :, None].repeat(2, 1, 1) / 10,
        "dt_frame": torch.full((2, model.window, 1), .1),
    }


def test_pinned_sources_and_tamper_detection(tmp_path, monkeypatch):
    manifest = models.verify_sources()
    assert {r["repository"]: r["commit"] for r in manifest} == {
        "rominay/LeHami": "f76f65f8b99afb6b7e3063f573645d3e2f290ede",
        "lucas-maes/le-wm": "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac",
    }
    copied = tmp_path / "source_copy"
    shutil.copytree(models.UPSTREAM, copied)
    path = copied / "LeHami/lehami/models/lehami/ham_flow.py"
    path.write_bytes(path.read_bytes() + b"\n# unexpected modification\n")
    monkeypatch.setattr(models, "UPSTREAM", copied)
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        models.verify_sources()


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_training_and_inference_share_physical_anchor_and_torques(kind):
    model = build_model(kind, unroll=3).eval()
    batch = batch_for(model)
    actions_seen, states_seen = [], []
    action_hook = model.model.action_encoder.register_forward_pre_hook(
        lambda module, args: actions_seen.append(args[0].detach().clone()))
    predictor_hook = model.model.predictor.register_forward_pre_hook(
        lambda module, args: states_seen.append(args[0].detach().clone()))
    try:
        # eval mode freezes batch statistics/dropout so differences reflect
        # indexing, rather than the amount of target imagery in a training clip.
        with torch.no_grad():
            model.loss(batch, "val")
        training_actions = actions_seen[:]
        training_anchor = states_seen[0]
        actions_seen.clear(); states_seen.clear()
        features = model.rollout_features(batch["pixels"][:, :5], batch["action"][:, 4:7],
                                          past_actions=batch["action"][:, :4])
    finally:
        action_hook.remove(); predictor_hook.remove()
    assert len(training_actions) == len(actions_seen) == 3
    for step, (train_u, infer_u) in enumerate(zip(training_actions, actions_seen)):
        # The transition into physical frame 5 is driven by action at frame 4.
        # LeWM also needs the two earlier controls for its retained context.
        expected = batch["action"][:, 4+step:5+step] if kind == "lehami" else batch["action"][:, 2+step:5+step]
        torch.testing.assert_close(train_u, expected, rtol=0, atol=0)
        torch.testing.assert_close(infer_u, expected, rtol=0, atol=0)
    torch.testing.assert_close(training_anchor, states_seen[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(features[:, 0], training_anchor.flatten(1), rtol=1e-5, atol=1e-6)
    assert features.shape == (2, 4, 64 if kind == "lehami" else 192)
    if kind == "lewm":
        # The readout retains the complete recurrent state, not merely z_t.
        contexts = features.reshape(2, 4, 3, 64)
        torch.testing.assert_close(contexts[:, 1:, :2], contexts[:, :-1, 1:], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_rollout_is_causal_and_encodes_only_initial_history(kind):
    model = build_model(kind, unroll=3).eval()
    batch = batch_for(model)
    history, future, past = batch["pixels"][:, :5], batch["action"][:, 4:7], batch["action"][:, :4]
    calls = []
    # W=1's encoder and W=3's shared spatial stem each see each raw image once.
    spatial = model.model.encoder if kind == "lehami" else model.model.encoder.stem
    hook = spatial.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    try:
        full = model.rollout_features(history, future, past_actions=past)
    finally:
        hook.remove()
    assert calls == [2 * (4 if kind == "lehami" else 5)]
    prefix = model.rollout_features(history, future[:, :1], past_actions=past)
    torch.testing.assert_close(full[:, :2], prefix, rtol=1e-6, atol=1e-7)
    changed_future = future.clone()
    changed_future[:, 1:] += 10
    changed = model.rollout_features(history, changed_future, past_actions=past)
    torch.testing.assert_close(full[:, :2], changed[:, :2], rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="five RGB frames"):
        model.rollout_features(batch["pixels"][:, :6], future, past_actions=past)
    with pytest.raises(ValueError, match="require past_actions"):
        model.rollout_features(history, future)


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_official_training_objective_has_finite_predictor_gradients(kind):
    model = build_model(kind, unroll=3).train()
    batch = batch_for(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    loss = model.loss(batch)["loss"]
    assert loss.ndim == 0 and torch.isfinite(loss)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in model.model.predictor.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    parameter = next(p for p in model.model.predictor.parameters()
                     if p.grad is not None and torch.count_nonzero(p.grad))
    before = parameter.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    assert not torch.equal(parameter, before)
    assert all(torch.isfinite(p).all() for p in model.parameters())
