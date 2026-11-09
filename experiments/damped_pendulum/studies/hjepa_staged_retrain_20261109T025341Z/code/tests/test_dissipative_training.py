"""Preflight checks plus explicitly bounded CPU update/recovery integration tests."""
import copy
import json
import random

import h5py
import numpy as np
import pytest
import torch

from dpbench.dissipative_data import DissipativeWindowDataset, fixed_validation_indices
from dpbench.train_dissipative import (
    DEFAULT_CONFIG, atomic_torch_save, capture_rng, config_digest, horizon_at_step,
    lr_factor, main, read_config, restore_rng, validate_checkpoint, validate_config,
    validation,
)


@pytest.fixture
def config():
    return read_config(DEFAULT_CONFIG)


@pytest.fixture
def tiny_data(tmp_path):
    path = tmp_path / "inputs.h5"
    episodes, frames = 6, 41
    n = episodes * frames
    with h5py.File(path, "w") as handle:
        handle.attrs["train_frames"] = frames
        handle.create_dataset("pixels", data=np.zeros((n, 64, 64, 3), dtype=np.uint8))
        action = np.zeros((episodes, frames, 1), dtype=np.float32)
        action[1::2, :-1] = .2
        handle.create_dataset("action", data=action.reshape(n, 1))
        handle.create_dataset("dt_frame", data=np.full((n, 1), .1, dtype=np.float32))
        valid = np.ones((episodes, frames), dtype=bool)
        valid[:, -1] = False
        handle.create_dataset("valid_transition", data=valid.ravel())
        handle.create_dataset("ep_offset", data=np.arange(episodes) * frames)
        handle.create_dataset("ep_len", data=np.full(episodes, frames))
        for split, ids in (("train", [0, 1]), ("val", [2, 3]), ("test", [4, 5])):
            handle.create_dataset("splits/" + split, data=np.asarray(ids, dtype=np.int64))
    return path


def test_curriculum_boundaries_and_fixed_override(config):
    expected = {1: 4, 1000: 4, 1001: 8, 4000: 8, 4001: 16,
                10000: 16, 10001: 32, 20000: 32}
    assert {step: horizon_at_step(config, step) for step in expected} == expected
    fixed = read_config(DEFAULT_CONFIG, steps=9, fixed_horizon=8)
    assert horizon_at_step(fixed, 1) == horizon_at_step(fixed, 9) == 8
    with pytest.raises(ValueError):
        horizon_at_step(config, 0)
    with pytest.raises(ValueError):
        horizon_at_step(config, 20001)


def test_learning_rate_schedule_resumes_by_absolute_step(config):
    assert lr_factor(config, 1) == pytest.approx(1 / 500)
    assert lr_factor(config, 500) == pytest.approx(1)
    assert lr_factor(config, 20000) == pytest.approx(.1)
    uninterrupted = [lr_factor(config, step) for step in range(1, 20001)]
    resumed = [lr_factor(config, step) for step in range(9021, 20001)]
    assert uninterrupted[9020:] == resumed


def test_reject_invalid_config(config):
    for key, value in (("batch_size", 1), ("train_frames", 20), ("encoder_lr", float("nan")),
                       ("minimum_target_std", 0), ("val_horizons", [32, 32])):
        bad = copy.deepcopy(config)
        bad["training"][key] = value
        with pytest.raises(ValueError):
            validate_config(bad)


def test_rng_checkpoint_roundtrip_without_updates(tmp_path):
    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    path = tmp_path / "rng.pt"
    atomic_torch_save(path, state)
    restore_rng(torch.load(path, weights_only=True))
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert list(tmp_path.glob("*.tmp")) == []


def test_resume_checks_binding_and_sampler_position(config):
    checkpoint = {
        "format_version": 1, "kind": "dissipative", "config_sha256": config_digest(config),
        "train_config": config, "dataset_sha256": "dataset-a", "source_hashes": {"a": "sha"},
        "device": "cpu", "step": 90, "sampler_next_step": 90,
        "state_dict": {}, "optimizer": {}, "rng": {}, "config": {},
        "validation_indices": [0], "best_score": None, "best_step": None,
    }
    validate_checkpoint(checkpoint, config, "dataset-a", {"a": "sha"}, "cpu")
    for key, value in (("dataset_sha256", "dataset-b"), ("source_hashes", {"a": "changed"}),
                       ("sampler_next_step", 91), ("device", "cuda")):
        bad = copy.deepcopy(checkpoint)
        bad[key] = value
        with pytest.raises(ValueError):
            validate_checkpoint(bad, config, "dataset-a", {"a": "sha"}, "cpu")


def test_default_cli_does_not_create_optimizer_or_run(tiny_data, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Default preflight must not construct an optimizer")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    output = tmp_path / "must_not_exist"
    config = read_config(DEFAULT_CONFIG)
    config["training"]["train_frames"] = 41
    config_path = tmp_path / "config.json"
    import json
    config_path.write_text(json.dumps(config))
    report = main(["--config", str(config_path), "--data", str(tiny_data), "--output", str(output)])
    assert report["training_started"] is False
    assert report["optimizer_created"] is False
    assert report["test_samples_read"] is False
    assert not output.exists()
    assert "preflight_only" in capsys.readouterr().out


class DiagnosticOnlyModel(torch.nn.Module):
    """No parameters and no fitting; exercises validation bookkeeping."""

    def __init__(self, collapsed=False):
        super().__init__()
        self.collapsed = collapsed
        self.config = {"variance_floor": 1e-4}

    def loss(self, batch, horizon):
        # Deliberately consume RNG to verify isolation from subsequent training.
        random.random()
        np.random.random()
        torch.rand(1)
        return {key: torch.tensor(value) for key, value in {
            "loss": 1., "prediction_loss": .2, "motion_loss": .1, "sigreg_loss": .2,
            "target_std_min": .01 if self.collapsed else .5,
            "target_motion_rms": .02, "persistence_mse": .5,
            "prediction_mse": .2, "normalized_prediction_mse": .4,
            "target_mean_0": .2, "target_mean_1": .1,
            "target_second_moment_0": .0401 if self.collapsed else .29,
            "target_second_moment_1": .0101 if self.collapsed else .26,
            "target_motion_mse": .0004,
        }.items()}


def test_fixed_validation_gates_and_rng_isolation(config, tiny_data):
    config["training"]["train_frames"] = 41
    config["training"]["val_windows"] = 8
    dataset = DissipativeWindowDataset(tiny_data, "val", max_horizon=32)
    indices = fixed_validation_indices(dataset, 8)
    model = DiagnosticOnlyModel()
    model.train()
    before = capture_rng()
    result = validation(model, dataset, indices, config, torch.device("cpu"))
    after = capture_rng()
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    torch.testing.assert_close(before["torch_cpu"], after["torch_cpu"], rtol=0, atol=0)
    assert model.training
    assert len(result["groups"]) == 12
    assert result["gate_passed"]
    assert result["score"] == pytest.approx(.8)
    assert not validation(DiagnosticOnlyModel(collapsed=True), dataset, indices, config,
                          torch.device("cpu"))["gate_passed"]
    dataset.close()


def test_validation_pools_moments_before_normalizing(config):
    from types import SimpleNamespace

    class MomentDataset:
        index = [SimpleNamespace(mode=0, time_bin=0) for _ in range(9)]

        def batch(self, indices, horizon):
            return {"pixels": torch.tensor(indices, dtype=torch.float32),
                    "action": torch.zeros(len(indices), 1),
                    "dt_frame": torch.ones(len(indices), 1)}

    class MomentModel(DiagnosticOnlyModel):
        def loss(self, batch, horizon):
            values = batch["pixels"].double()
            mean, second = values.mean(), values.square().mean()
            return {
                "loss": second, "prediction_mse": second * .1,
                "normalized_prediction_mse": second * .1 / values.var(unbiased=False).clamp_min(1e-4),
                "persistence_mse": second, "target_mean_0": mean,
                "target_second_moment_0": second,
                "target_motion_mse": values.new_tensor(.01),
                "solver_residual": values.max(),
            }

    model = MomentModel()
    config["training"]["batch_size"] = 2
    first = validation(model, MomentDataset(), list(range(9)), config, torch.device("cpu"))
    config["training"]["batch_size"] = 5
    second = validation(model, MomentDataset(), list(range(9)), config, torch.device("cpu"))
    assert first["score"] == pytest.approx(second["score"], abs=1e-14)
    assert first["gate_passed"] == second["gate_passed"] is True
    assert all(row["solver_residual"] == 8 for row in first["groups"].values())
    assert all(row["solver_residual"] == 8 for row in second["groups"].values())


def test_preflight_rejects_incompatible_clock(tiny_data, config):
    from dpbench.train_dissipative import preflight
    config["training"]["train_frames"] = 41
    config["model"]["dt"] = .2
    with pytest.raises(ValueError, match="dt differs"):
        preflight(tiny_data, config)


def test_preflight_rejects_hidden_environment_without_context_model(tiny_data, config):
    from dpbench.train_dissipative import preflight
    config["training"]["train_frames"] = 41
    with h5py.File(tiny_data, "r+") as f:
        f.attrs["requires_environment_context"] = True
        f.attrs["history_required"] = 5
        f.create_dataset("regime_id", data=np.arange(6, dtype=np.int64) % 2)
    with pytest.raises(ValueError, match="environment_context_dim"):
        preflight(tiny_data, config)
    config["model"]["environment_context_dim"] = 4
    report = preflight(tiny_data, config)
    assert report["requires_environment_context"] and not report["training_started"]


def _tiny_execution_config(config, val_every):
    cfg = copy.deepcopy(config)
    cfg["model"] = dict(history=21, velocity_history=5, environment_context_dim=4,
                        encoder_width=8, sigreg_projections=4,
                        core=dict(hidden_dim=4, substeps=1))
    cfg["training"].update(steps=3, batch_size=2, train_frames=41, threads=2,
                           warmup_steps=1, val_every=val_every, val_windows=6,
                           val_horizons=[1], minimum_target_std=1e6,
                           curriculum=[dict(until_step=3, horizon=1)])
    return cfg


def _assert_nested_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_nested_equal(expected[key], actual[key])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_nested_equal(left, right)
    else:
        assert expected == actual


@pytest.mark.parametrize("val_every", [1, 3])
def test_real_adam_resume_replays_interrupted_validation(tiny_data, config, tmp_path,
                                                        monkeypatch, val_every):
    import dpbench.train_dissipative as trainer

    cfg = _tiny_execution_config(config, val_every)
    full, interrupted, resumed = (tmp_path / name for name in ("full", "interrupted", "resumed"))
    trainer.execute_run(tiny_data, full, cfg)
    original_validation = trainer.validation

    def interrupt_validation(*args, **kwargs):
        # The just-completed update must already be recoverable, even when
        # the first scheduled validation is also the final training step.
        saved = torch.load(interrupted / "last.pt", weights_only=True)
        assert saved["step"] == val_every
        assert saved["validation"] is None and saved["validation_status"] == "not_run"
        raise RuntimeError("simulated validation interruption")

    monkeypatch.setattr(trainer, "validation", interrupt_validation)
    with pytest.raises(RuntimeError, match="simulated validation interruption"):
        trainer.execute_run(tiny_data, interrupted, cfg)
    checkpoint = torch.load(interrupted / "last.pt", weights_only=True)
    assert checkpoint["optimizer"]["state"]
    assert any(state["exp_avg"].abs().sum() > 0
               for state in checkpoint["optimizer"]["state"].values())
    assert json.loads((interrupted / "status.json").read_text())["status"] == "failed"

    monkeypatch.setattr(trainer, "validation", original_validation)
    status = trainer.execute_run(tiny_data, resumed, cfg, resume=interrupted / "last.pt")
    assert status["status"] == "completed" and status["completed_steps"] == 3
    assert status["optimizer_steps_this_run"] == 3 - val_every
    replay = json.loads((resumed / "resume_validation.json").read_text())
    assert replay["step"] == val_every and replay["optimizer_steps"] == 0
    assert replay["validation_seconds"] > 0 and replay["checkpoint_seconds"] > 0
    expected = torch.load(full / "last.pt", weights_only=True)
    actual = torch.load(resumed / "last.pt", weights_only=True)
    for key in ("state_dict", "optimizer", "rng", "validation", "sampler_next_step"):
        _assert_nested_equal(expected[key], actual[key])
    assert actual["validation_status"] == "completed"
    assert not (resumed / "best.pt").exists()  # No bypass of representation gates.
    rows = [json.loads(line) for line in (full / "history.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(row["step_seconds"] > 0 and row["training_memory"] == {} for row in rows)
    assert rows[0]["checkpoint_seconds"] > 0  # Step one always has a recovery point.
    assert rows[-1]["validation_seconds"] > 0 and rows[-1]["checkpoint_seconds"] > 0
    assert rows[-1]["iteration_seconds"] >= (rows[-1]["step_seconds"]
                                              + rows[-1]["validation_seconds"]
                                              + rows[-1]["checkpoint_seconds"])


def test_validation_keeps_regimes_separate_and_allows_late_settling(config):
    from types import SimpleNamespace

    class RegimeDataset:
        index = [SimpleNamespace(regime_id=r, mode=0, time_bin=t)
                 for r in range(2) for t in range(3)]

        def batch(self, indices, horizon):
            return {"pixels": torch.tensor([self.index[i].time_bin for i in indices]),
                    "action": torch.zeros(len(indices), 1), "dt_frame": torch.ones(len(indices), 1)}

    class SettlingModel(DiagnosticOnlyModel):
        def loss(self, batch, horizon):
            self.collapsed = bool((batch["pixels"] > 0).all())
            return super().loss(batch, horizon)

    config["training"]["validation_gate_time_bins"] = [0]
    model, dataset = SettlingModel(), RegimeDataset()
    report = validation(model, dataset, list(range(6)), config, torch.device("cpu"))
    assert len(report["groups"]) == 12
    assert report["gate_passed"]
    assert any(not row["gate_passed"] for row in report["groups"].values())
    assert sum(row["gate_required"] for row in report["groups"].values()) == 4
    config["training"].pop("validation_gate_time_bins")
    assert not validation(model, dataset, list(range(6)), config, torch.device("cpu"))["gate_passed"]


def test_gate_time_bin_configuration_is_validated(config):
    for value in ([], [True], [3], [0, 0]):
        bad = copy.deepcopy(config)
        bad["training"]["validation_gate_time_bins"] = value
        with pytest.raises(ValueError, match="time_bins"):
            validate_config(bad)


def test_real_adam_bounded_run_preserves_full_schedule_and_resume(tiny_data, config, tmp_path):
    import dpbench.train_dissipative as trainer
    cfg = _tiny_execution_config(config, 1)
    full, bounded, resumed = [tmp_path / name for name in ('unbounded', 'bounded', 'continued')]
    trainer.execute_run(tiny_data, full, cfg)
    status = trainer.execute_run(tiny_data, bounded, cfg, stop_after_step=1)
    assert status['status'] == 'budget_completed'
    assert status['completed_steps'] == 1 and not status['full_budget_completed']
    saved = torch.load(bounded / 'last.pt', weights_only=True)
    assert saved['train_config'] == cfg and saved['validation_status'] == 'completed'
    status = trainer.execute_run(tiny_data, resumed, cfg, resume=bounded / 'last.pt')
    assert status['full_budget_completed'] and status['completed_steps'] == 3
    expected = torch.load(full / 'last.pt', weights_only=True)
    actual = torch.load(resumed / 'last.pt', weights_only=True)
    for key in ('state_dict', 'optimizer', 'rng', 'validation', 'sampler_next_step'):
        _assert_nested_equal(expected[key], actual[key])
    with pytest.raises(ValueError, match='scheduled validation'):
        trainer.execute_run(tiny_data, tmp_path / 'invalid', cfg, stop_after_step=4)
