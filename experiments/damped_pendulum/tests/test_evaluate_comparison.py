"""CPU fixtures for causal evaluation, grouped calibration and metric support."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dpbench import evaluate_comparison as ev


def make_data(path, damping=(.2,)):
    """RGB encodes a simple trajectory; train/val are identifiable, no test exists."""
    groups, frames = 448, 81
    ids = np.tile(np.arange(groups), len(damping))
    bands = ids % 4
    phase = np.arange(frames)[None, :] + 2 * bands[:, None]
    speed_code = np.broadcast_to((10 + 2 * bands)[:, None], phase.shape)
    split_code = np.broadcast_to(np.where(ids < 384, 0, 255)[:, None], phase.shape)
    pixels = np.stack((phase, speed_code, split_code), -1).astype(np.uint8)
    with h5py.File(path, "w") as f:
        f.attrs.update(renderer="smooth_v1", history_required=21, train_frames=81, dt=.1,
                       mass=1., length=1., gravity=9.81, parent_config_sha256="f" * 64,
                       selected_damping_json=json.dumps(list(damping)))
        f["ep_offset"] = np.arange(len(ids)) * frames
        f["ep_len"] = np.full(len(ids), frames)
        f["group_id"] = ids
        f["energy_band"] = bands
        f["damping"] = np.repeat(damping, groups)
        f["pixels"] = pixels.reshape(-1, 1, 1, 3)
        f["state"] = np.stack((phase * .01, speed_code * .01), -1).reshape(-1, 2)
        f["action"] = np.zeros((len(ids) * frames, 1), np.float32)
        f["dt_frame"] = np.full((len(ids) * frames, 1), .1, np.float32)
        f["splits/train"] = np.flatnonzero(ids < 384)
        f["splits/val"] = np.flatnonzero(ids >= 384)


class PixelModel(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.events = []
        self.provenance = {"kind": "fixture", "feature_dimension": 5}

    def features(self, theta, omega):
        return torch.stack((theta + self.bias, omega, theta ** 2, torch.cos(theta), torch.ones_like(theta)), -1)

    def observed_features(self, history, past_actions=None):
        assert history.shape[1:] == (21, 3, 1, 1)
        assert past_actions.shape == (len(history), 20, 1) and torch.all(past_actions == 0)
        self.events.append(("observed", history[:, -1, 2, 0, 0].detach().cpu().numpy().copy()))
        theta = history[:, -1, 0, 0, 0].double() * 2.55
        omega = history[:, -1, 1, 0, 0].double() * 2.55
        return self.features(theta, omega)

    def rollout_features(self, history, actions, past_actions):
        assert history.shape[1:] == (21, 3, 1, 1)
        assert past_actions.shape == (len(history), 20, 1)
        assert actions.shape[0] == len(history) and actions.shape[-1] == 1
        assert torch.all(actions == 0) and torch.all(past_actions == 0)
        # No future RGB (whose red values are >26) can appear in this call.
        assert float(history[:, :, 0].max()) <= 26 / 255 + 1e-7
        self.events.append(("autonomous", history.detach().cpu().numpy().copy()))
        steps = torch.arange(actions.shape[1] + 1, device=history.device, dtype=torch.float64)
        theta = history[:, -1, 0, 0, 0].double()[:, None] * 2.55 + steps[None] * .01
        omega = history[:, -1, 1, 0, 0].double()[:, None].expand_as(theta) * 2.55
        return self.features(theta, omega)


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "single.h5"
    make_data(path)
    return path


def checkpoint(path, data, kind="comparison"):
    model = PixelModel()
    saved = dict(kind=kind, step=20000, state_dict=model.state_dict(), dataset_sha256=ev.sha256(data))
    if kind == "comparison":
        saved.update(model_config={"kind": "fixture", "history": 21, "dt": .1}, protocol_sha256="a" * 64)
    else:
        saved.update(config={"history": 21, "dt": .1})
    torch.save(saved, path)


def test_multi_group_partition_never_splits_conditions(tmp_path):
    path = tmp_path / "multi.h5"
    make_data(path, ev.SEEN_DAMPING)
    rows, groups, _, values, _ = ev.inspect_data(path, 32)
    fit, selection, fit_ids, selection_ids = ev.partition_training_groups(rows["train"])
    assert len(fit_ids) == 288 and len(selection_ids) == 96
    assert not set(fit_ids) & set(selection_ids)
    assert len(fit) == 288 * 6 and len(selection) == 96 * 6
    assert set(fit_ids + selection_ids) == set(groups["train"])
    for gid in fit_ids:
        assert {r["damping"] for r in fit if r["group_id"] == gid} == set(values)
    assert ev.partition_training_groups(list(reversed(rows["train"])))[2:] == (fit_ids, selection_ids)


def test_autonomous_features_ignore_future_pixels_and_labels(data):
    rows, _, _, _, _ = ev.inspect_data(data, 16)
    model = PixelModel()
    before = ev.collect_autonomous_features(model, data, rows["val"][:2], 16, batch_size=2)
    with h5py.File(data, "r+") as f:
        for r in rows["val"][:2]:
            offset = int(f["ep_offset"][r["episode"]])
            f["pixels"][offset + 21:offset + 81] = 255
            f["state"][offset + 21:offset + 81] = 999
    after = ev.collect_autonomous_features(model, data, rows["val"][:2], 16, batch_size=2)
    np.testing.assert_array_equal(before["features"], after["features"])
    assert not np.array_equal(before["truth"], after["truth"])


def test_observed_features_are_causal_at_each_anchor(data):
    model = PixelModel()
    pair = [(0, 30)]
    before, y = ev.collect_observed_features(model, data, pair, 1)
    with h5py.File(data, "r+") as f:
        f["pixels"][31:81] = 255
        f["state"][31:81] = 999
    after, changed_y = ev.collect_observed_features(model, data, pair, 1)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(y, changed_y)


@pytest.mark.parametrize("kind", ["comparison", "dissipative"])
def test_end_to_end_own_readout_train_only_and_manifest(data, tmp_path, monkeypatch, kind):
    model = PixelModel()
    if kind == "comparison":
        monkeypatch.setattr(ev, "build_comparison_model", lambda config, device: model)
    else:
        class LegacyAdapter(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.model = model
                self.provenance = {"kind": "legacy_fixture"}
            def observed_features(self, *a, **kw):
                return self.model.observed_features(*a, **kw)
            def rollout_features(self, *a, **kw):
                return self.model.rollout_features(*a, **kw)
        monkeypatch.setattr(ev, "DissipativeEvaluationAdapter", LegacyAdapter)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("No optimizer is allowed in physical evaluation"))
    cp, out = tmp_path / "checkpoint.pt", tmp_path / "evaluation"
    checkpoint(cp, data, kind)
    result = ev.evaluate_checkpoint(cp, data, out, batch_size=256)
    assert result["status"] == "completed" and result["parameters_unchanged"]
    assert result["model_hash_before"] == result["model_hash_after"]
    assert not result["test_split_accessed"] and not result["readout_reused_from_other_model"]
    assert not result["future_observations_passed_to_predictor"]
    assert result["calibration"]["fit_groups"] == 288 and result["calibration"]["selection_groups"] == 96
    assert result["calibration"]["feature_dimension"] == 5
    assert result["protocol_sha256"] == (None if kind == "dissipative" else "a" * 64)
    assert len(result["evaluation_protocol_sha256"]) == 64
    validation = result["evaluation"]["val"]
    assert validation["unique_group_count"] == validation["record_count"] == 64
    assert set(validation["rollout"]) == {"8", "16", "32"}
    metrics = validation["rollout"]["32"]["groups"]["damping_0.2"]
    assert metrics["coverage"] == 1 and metrics["metrics"]["angle_mae_rad"] < 1e-6
    first_auto = next(i for i, (event, _) in enumerate(model.events) if event == "autonomous")
    assert all(event == "observed" and np.all(value == 0) for event, value in model.events[:first_auto])
    assert all(np.all(value == 1) for event, value in model.events[first_auto + 1:] if event == "observed")
    with np.load(out / "readout.npz", allow_pickle=False) as z:
        assert z["fit_features"].shape == (288 * 61, 5)
        assert len(set(z["fit_group_ids"])) == 288
        assert len(set(z["selection_group_ids"])) == 96
        assert not set(z["fit_group_ids"]) & set(z["selection_group_ids"])
        assert max(z["fit_group_ids"]) < 384 and max(z["selection_group_ids"]) < 384
    with np.load(out / "val/trajectories.npz", allow_pickle=False) as z:
        assert z["features"].shape == z["teacher_features"].shape == (64, 33, 5)
        assert np.all(z["valid_steps"] == 32)
        np.testing.assert_allclose(z["time"], np.arange(33) * .1)
    manifest = json.loads((out / "complete.json").read_text())
    assert all(ev.sha256(out / name) == value for name, value in manifest["artifacts"].items())


def test_nonfinite_prefix_reduces_coverage_without_holding_state():
    features = np.ones((3, 17, 5))
    features[2, 5, 0] = np.nan
    cleaned, valid, failures = ev.finite_prefix(features)
    assert valid.tolist() == [16, 16, 4]
    assert np.isnan(cleaned[2, 5:]).all() and np.isfinite(cleaned[2, :5]).all()
    assert failures == [{"evaluation_index": 2, "failed_step": 5, "error_type": "NonfiniteFeatures"}]
    truth = np.zeros((3, 17, 2))
    pred = truth.copy()
    truth[0, :, 0], pred[0, :, 0] = np.pi - .1, -np.pi + .1
    truth[0, :, 1], pred[0, :, 1] = 2, .1
    pred[2, 5:] = np.nan
    records = [{"group_id": i, "energy_band": 0, "damping": .2} for i in range(3)]
    report = ev.summarize_horizon(pred, truth.copy(), truth, valid, records, 16, 1.,
                                  {"mass": 1., "length": 1., "gravity": 9.81})
    all_ = report["groups"]["all"]
    assert all_["coverage"] == 2 / 3 and not report["ranking_eligible"]
    assert all_["metrics"]["angle_mae_rad"] == pytest.approx(.1)
    assert all_["metrics"]["omega_rmse_rad_s"] == pytest.approx(np.sqrt(1.9 ** 2 / 2))
    # The static true trajectory does not enter the dynamic-window denominator.
    assert all_["metrics"]["dynamic_window_count"] == 7
    assert all_["metrics"]["stopped_dynamic_window_fraction"] == 1
    assert all_["metrics"]["trajectory_with_stop_fraction"] == 1
    assert all_["physical_energy"]["future_energy_relative_mae_median"] is not None


def test_numerical_exception_isolated_but_structural_error_propagates(data):
    rows, _, _, _, _ = ev.inspect_data(data, 8)
    class Numerical(PixelModel):
        def rollout_features(self, history, actions, past_actions):
            if bool((history[:, -1, 0, 0, 0] < 21 / 255).any()):
                raise FloatingPointError("fixture numerical failure")
            return super().rollout_features(history, actions, past_actions)
    result = ev.collect_autonomous_features(Numerical(), data, rows["val"][:2], 8, 2, feature_dim=5)
    assert result["valid_steps"].tolist() == [-1, 8]
    assert result["failures"][0]["error_type"] == "FloatingPointError"
    assert np.isnan(result["features"][0]).all()
    class Structural(PixelModel):
        def rollout_features(self, history, actions, past_actions):
            raise RuntimeError("fixture shape mismatch")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        ev.collect_autonomous_features(Structural(), data, rows["val"][:2], 8, 2, feature_dim=5)


def test_limits_and_hash_fail_before_model_build(data, tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "build_comparison_model", lambda *a, **kw: pytest.fail("Model must not be built"))
    cp = tmp_path / "checkpoint.pt"
    checkpoint(cp, data)
    with pytest.raises(ValueError, match="at most 60 future"):
        ev.evaluate_checkpoint(cp, data, tmp_path / "too_long", horizons=(61,))
    with pytest.raises(ValueError, match="test is sealed"):
        ev.evaluate_checkpoint(cp, data, tmp_path / "test", split="test")
    with h5py.File(data, "r+") as f:
        f.attrs["new_metadata"] = "different dataset bytes"
    with pytest.raises(ValueError, match="SHA-256 differs"):
        ev.evaluate_checkpoint(cp, data, tmp_path / "wrong_hash")
    assert not (tmp_path / "too_long").exists() and not (tmp_path / "wrong_hash").exists()


def test_checkpoint_comparison_source_identity_and_legacy_mode(data, tmp_path, monkeypatch):
    from dpbench import comparison_protocol
    monkeypatch.setattr(comparison_protocol, "source_manifest", lambda: {"model.py": "a" * 64})
    matched = ev.verify_checkpoint_sources({"kind": "comparison", "source_sha256": {"model.py": "a" * 64}})
    assert matched["strict_source_identity_verified"]
    legacy = ev.verify_checkpoint_sources({"kind": "dissipative", "source_hashes": {"old.py": "b" * 64}})
    assert legacy["status"] == "legacy_compatibility_mode" and not legacy["strict_source_identity_verified"]
    cp = tmp_path / "changed_source.pt"
    checkpoint(cp, data)
    saved = torch.load(cp, weights_only=True)
    saved["source_sha256"] = {"model.py": "b" * 64}
    torch.save(saved, cp)
    monkeypatch.setattr(ev, "build_comparison_model", lambda *a, **kw: pytest.fail("Changed code must fail before construction"))
    with pytest.raises(ValueError, match="source identity differs"):
        ev.evaluate_checkpoint(cp, data, tmp_path / "changed_source_eval")
    assert not (tmp_path / "changed_source_eval").exists()
