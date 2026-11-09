"""Synthetic bootstrap tests with explicitly bounded CPU optimizer updates.

The A fixture is a labelled synthetic final-step snapshot, not a trained model.
No production data, server, GPU job, or scientific training is used.
"""
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dpbench.dissipative_data import DissipativeWindowDataset
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.staged_damping import (
    create_damping_seed, shared_state_sha256, transfer_manifest_path,
)
from dpbench.train_dissipative import (
    build_optimizer_parameter_groups, capture_rng, config_digest,
    configure_trainable_modules, execute_run, read_config, runtime_signature,
    source_hashes, validate_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def transfer_case(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(6105)
    config_a = read_config(ROOT / "configs/dissipative_loss_pair_lehami_v1.json")
    config_a["model"].update(history=5, velocity_history=5, q_dim=2,
                             encoder_width=8, sigreg_projections=4)
    config_a["model"]["core"].update(hidden_dim=8, substeps=1)
    config_a["training"].update(batch_size=2, train_frames=12, threads=2,
        warmup_steps=1, val_every=1, val_windows=6, val_horizons=[1],
        curriculum=[dict(until_step=20000, horizon=1)])
    model_a = DissipativeWorldModel(config_a["model"])
    trainability = configure_trainable_modules(model_a, config_a["training"])
    optimizer_a = torch.optim.AdamW(build_optimizer_parameter_groups(model_a, config_a["training"]))
    parameter = next(p for p in model_a.parameters() if p.requires_grad)
    optimizer_a.state[parameter] = dict(step=torch.tensor(20000.),
                                      exp_avg=torch.ones_like(parameter),
                                      exp_avg_sq=torch.ones_like(parameter))
    checkpoint_a = dict(format_version=1, kind="dissipative", config=model_a.config,
        train_config=config_a, config_sha256=config_digest(config_a),
        state_dict=model_a.state_dict(), optimizer=optimizer_a.state_dict(),
        step=20000, sampler_next_step=20000, rng=capture_rng(),
        dataset_sha256="a" * 64, source_hashes=source_hashes(), device="cpu",
        runtime=runtime_signature("cpu"), validation_indices=[0], trainability=trainability,
        validation={"gate_passed": False, "score": 1.0}, validation_status="completed",
        best_score=None, best_step=None, best_checkpoint_path=None)
    source = tmp_path / "A_step20000.pt"
    torch.save(checkpoint_a, source)
    episodes, frames = 6, 12
    rng = np.random.default_rng(6105)
    data = tmp_path / "single.h5"
    with h5py.File(data, "w") as handle:
        handle.attrs.update(train_frames=frames, renderer="smooth_v1",
                            training_scope="damping_stage_b_single", requires_environment_context=False)
        handle.create_dataset("pixels", data=rng.integers(0, 256,
            (episodes * frames, 64, 64, 3), dtype=np.uint8))
        handle.create_dataset("action", data=np.zeros((episodes * frames, 1), dtype=np.float32))
        handle.create_dataset("dt_frame", data=np.full((episodes * frames, 1), .1, dtype=np.float32))
        valid = np.ones((episodes, frames), dtype=bool)
        valid[:, -1] = False
        handle.create_dataset("valid_transition", data=valid.ravel())
        handle.create_dataset("ep_offset", data=np.arange(episodes) * frames)
        handle.create_dataset("ep_len", data=np.full(episodes, frames))
        for split, ids in (("train", [0, 1]), ("val", [2, 3]), ("test", [4, 5])):
            handle.create_dataset("splits/" + split, data=np.asarray(ids, dtype=np.int64))
    config_b = copy.deepcopy(config_a)
    config_b["model"]["core"]["damping_mode"] = "position_velocity"
    config_b["training"].update(steps=1, seed=6106, trainable_modules=["damping"],
        required_data_scope="damping_stage_b_single", curriculum=[dict(until_step=1, horizon=1)])
    return source, data, tmp_path / "seed.pt", config_b


def _multi(case):
    source, data, output, config = case
    config = copy.deepcopy(config)
    config["model"]["environment_context_dim"] = 4
    config["training"].update(trainable_modules=["damping", "environment"],
                               required_data_scope="damping_stage_b_multi")
    with h5py.File(data, "r+") as handle:
        handle.attrs.update(training_scope="damping_stage_b_multi", requires_environment_context=True)
        handle.create_dataset("regime_id", data=np.arange(6, dtype=np.int64) % 2)
    return source, data, output, config


@pytest.mark.parametrize("multi", [False, True])
def test_exact_copy_fresh_optimizer_idempotence_and_one_real_update(transfer_case, multi):
    source, data, output, config = _multi(transfer_case) if multi else transfer_case
    before_source = hashlib.sha256(source.read_bytes()).hexdigest()
    original_rng = capture_rng()
    manifest = create_damping_seed(source, data, output, config, "cpu", expected_a_data_sha256="a" * 64)
    torch.testing.assert_close(capture_rng()["torch_cpu"], original_rng["torch_cpu"], rtol=0, atol=0)
    assert not manifest["reused_existing"] and manifest["optimizer_updates"] == 0
    assert manifest["branch"] == ("multi_environment" if multi else "single_environment")
    saved = torch.load(output, weights_only=True)
    a = torch.load(source, weights_only=True)
    assert saved["step"] == saved["sampler_next_step"] == 0
    assert saved["optimizer"]["state"] == {} and a["optimizer"]["state"]
    assert saved["best_score"] is None and saved["validation_status"] == "not_run"
    validate_checkpoint(saved, config, saved["dataset_sha256"], source_hashes(), "cpu", runtime_signature("cpu"))
    assert set(manifest["shared_state_keys"]) == set(a["state_dict"])
    for name, value in a["state_dict"].items():
        assert torch.equal(value, saved["state_dict"][name]), name
    assert shared_state_sha256(a["state_dict"]) == manifest["shared_state_sha256"]
    seed_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    reused = create_damping_seed(source, data, output, config, "cpu")
    assert reused["reused_existing"] and hashlib.sha256(output.read_bytes()).hexdigest() == seed_hash
    # A completed seed can safely recover a missing sidecar after interruption.
    transfer_manifest_path(output).unlink()
    assert create_damping_seed(source, data, output, config, "cpu")["reused_existing"]
    assert hashlib.sha256(output.read_bytes()).hexdigest() == seed_hash
    model = DissipativeWorldModel(config["model"])
    model.load_state_dict(saved["state_dict"], strict=True)
    configure_trainable_modules(model, config["training"])
    groups = build_optimizer_parameter_groups(model, config["training"])
    assert len(groups) == 1 and groups[0]["name"] == "dynamics"
    optimizer = torch.optim.AdamW(groups, weight_decay=config["training"]["weight_decay"])
    optimizer.load_state_dict(saved["optimizer"])
    q, p = torch.randn(2, 2, dtype=torch.float64), torch.randn(2, 2, dtype=torch.float64)
    context = torch.randn(2, 4, dtype=torch.float64) if multi else None
    torch.testing.assert_close(model.core.damping(q, p, context),
        .01 * torch.eye(2, dtype=torch.float64).expand(2, 2, 2), rtol=1e-14, atol=1e-15)
    dataset = DissipativeWindowDataset(data, "train", history=5, max_horizon=1,
                                      train_frames=12, target_tail_frames=2)
    try:
        batch = dataset.batch(manifest["initial_train_window_indices"], horizon=1)
    finally:
        dataset.close()
    loss = model.loss(batch, 1)
    loss["loss"].backward()
    assert model.core.damping_net[-1].weight.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    optimizer.step()  # Exactly one synthetic CPU update, not a scientific run.
    assert shared_state_sha256(model.state_dict(), manifest["shared_state_keys"]) == manifest["shared_state_sha256"]
    assert any(not torch.equal(model.state_dict()[name], saved["state_dict"][name])
               for name in manifest["trainable_parameter_names"])
    if multi:
        # Initial zero output weights intentionally block context gradients on
        # update one. After that update the new context pathway must be active.
        optimizer.zero_grad(set_to_none=True)
        model.loss(batch, 1)["loss"].backward()
        assert model.environment_encoder[-2].weight.grad.abs().sum() > 0
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_source


@pytest.mark.parametrize("multi", [False, True])
def test_seed_enters_unmodified_strict_trainer(transfer_case, tmp_path, multi):
    source, data, output, config = _multi(transfer_case) if multi else transfer_case
    config["training"].update(steps=3, curriculum=[dict(until_step=3, horizon=1)])
    manifest = create_damping_seed(source, data, output, config, "cpu")
    destination = tmp_path / "two_cpu_updates"
    status = execute_run(data, destination, config, device="cpu", resume=output, stop_after_step=2)
    assert status["status"] == "budget_completed" and status["optimizer_steps_this_run"] == 2
    assert status["configured_total_steps"] == 3 and status["completed_steps"] == 2
    final = torch.load(destination / "last.pt", weights_only=True)
    assert final["step"] == 2 and final["validation_status"] == "completed"
    assert shared_state_sha256(final["state_dict"], manifest["shared_state_keys"]) == manifest["shared_state_sha256"]
    replay = json.loads((destination / "resume_validation.json").read_text())
    assert replay["step"] == 0 and replay["optimizer_steps"] == 0
    assert final["optimizer"]["state"]
    assert all(float(state["step"]) == 2 for state in final["optimizer"]["state"].values())


@pytest.mark.parametrize("case", ["step", "validation", "source_hash", "source_config", "shape",
                                  "loss", "time", "trainability", "learning_rate", "data_hash", "zero_factor"])
def test_reject_invalid_transition_before_writing(transfer_case, case):
    source, data, output, config = transfer_case
    saved = torch.load(source, weights_only=True)
    kwargs = {}
    if case == "step":
        saved["step"] = saved["sampler_next_step"] = 19999
    elif case == "validation":
        saved["validation_status"] = "not_run"
    elif case == "source_hash":
        saved["source_hashes"]["dpbench/train_dissipative.py"] = "changed"
    elif case == "source_config":
        saved["config"]["dt"] = .2
    elif case == "shape":
        saved["state_dict"]["encoder.net.15.weight"] = torch.zeros(1)
    elif case == "loss":
        config["model"]["sigreg_weight"] = .5
    elif case == "time":
        config["model"]["dt"] = .2
    elif case == "trainability":
        config["training"]["trainable_modules"] = ["damping", "potential"]
    elif case == "learning_rate":
        config["training"]["dynamics_lr"] *= 2
    elif case == "data_hash":
        kwargs["expected_a_data_sha256"] = "b" * 64
    else:
        kwargs["damping_init"] = 0.
    torch.save(saved, source)
    with pytest.raises(ValueError):
        create_damping_seed(source, data, output, config, "cpu", **kwargs)
    assert not output.exists() and not transfer_manifest_path(output).exists()


def test_reject_context_and_data_mismatch(transfer_case):
    source, data, output, config = transfer_case
    config["model"]["environment_context_dim"] = 4
    config["training"]["trainable_modules"] = ["damping", "environment"]
    with pytest.raises(ValueError, match="multi-environment"):
        create_damping_seed(source, data, output, config, "cpu")


@pytest.mark.parametrize("tamper", ["state", "rng", "manifest"])
def test_reject_tampered_seed_or_manifest_without_overwrite(transfer_case, tamper):
    source, data, output, config = transfer_case
    create_damping_seed(source, data, output, config, "cpu")
    if tamper == "manifest":
        sidecar = transfer_manifest_path(output)
        manifest = json.loads(sidecar.read_text())
        manifest["source_step"] = 19999
        sidecar.write_text(json.dumps(manifest))
    else:
        saved = torch.load(output, weights_only=True)
        if tamper == "state":
            saved["state_dict"]["core.damping_net.4.bias"].add_(.1)
        else:
            saved["rng"]["torch_cpu"] = torch.roll(saved["rng"]["torch_cpu"], 1)
        torch.save(saved, output)
    corrupted = output.read_bytes()
    with pytest.raises(ValueError, match="pristine transfer|manifest"):
        create_damping_seed(source, data, output, config, "cpu")
    assert output.read_bytes() == corrupted
