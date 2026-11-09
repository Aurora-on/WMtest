"""Real tiny-data qualification of matched-comparison training and resumption.

These two-update CPU runs are software fixtures, not scientific experiments.
"""
import copy
import json
from pathlib import Path
import shutil

import h5py
import pytest
import torch

from dpbench import comparison_protocol as protocol
from dpbench import train_comparison as trainer
from dpbench import passive_family, damping_stage_data


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual_paired_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison_training_data")
    cfg = json.loads((ROOT / "configs/passive_family_smooth_v1.json").read_text())
    cfg.update(groups={"train": 8, "val": 4, "test": 4}, test_frames=81)
    passive_family.generate_undamped(cfg, root / "a")
    damping_stage_data.generate(cfg, root / "single")
    a, b = root / "a/undamped.h5", root / "single/damping.h5"
    assert damping_stage_data.validate_pair(a, b)["status"] == "matched"
    return {"a": a, "single": b}


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture_config(kind="hjepa", stage="a"):
    cfg = protocol.make_config(kind, stage=stage, steps=2, tiny=True)
    cfg["data"]["expected_groups"] = {"train": 8, "val": 4}
    cfg["training"].update(batch_size=2, threads=1, val_every=2,
                           checkpoint_every=1, val_windows=6, val_horizons=[4],
                           curriculum=[{"until_step": 2, "horizon": 4}],
                           warmup_steps=1)
    return cfg


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_tree_equal(left, right)
    else:
        assert a == b


def test_preflight_and_default_cli_never_construct_model_or_optimizer(actual_paired_data, tmp_path, monkeypatch, capsys):
    def forbidden(*a, **kw):
        pytest.fail("Read-only preflight constructed a model or optimizer")
    monkeypatch.setattr(trainer, "build_comparison_model", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    cfg = fixture_config()
    file = tmp_path / "config.json"
    file.write_text(json.dumps(cfg))
    output = tmp_path / "must_not_exist"
    trainer.main(["--config", str(file), "--data", str(actual_paired_data["a"]),
                  "--output", str(output)])
    result = json.loads(capsys.readouterr().out)
    assert not result["optimizer_created"] and not result["training_started"]
    assert not result["test_samples_read"] and result["target_fields_read"] == []
    assert result["model_fields"] == ["pixels", "action", "dt_frame"]
    assert not output.exists()


@pytest.mark.parametrize("kind", ["lehami", "hjepa"])
def test_exact_resume_preserves_parameters_bn_optimizer_rng_and_losses(kind, actual_paired_data, tmp_path):
    cfg, data = fixture_config(kind), actual_paired_data["a"]
    complete = load(trainer.execute_run(data, tmp_path / "complete", cfg))
    first_path = trainer.execute_run(data, tmp_path / "resumed", cfg, stop_after_step=1)
    partial = load(first_path)
    assert partial["step"] == partial["sampler_next_step"] == 1
    resumed = load(trainer.execute_run(data, tmp_path / "resumed", cfg, resume=first_path))
    for key in ("state_dict", "optimizer", "rng", "step", "sampler_next_step",
                "validation_indices", "validation"):
        assert_tree_equal(complete[key], resumed[key])
    history = [json.loads(line) for line in (tmp_path / "resumed/history.jsonl").read_text().splitlines()]
    reference = [json.loads(line) for line in (tmp_path / "complete/history.jsonl").read_text().splitlines()]
    assert [row["step"] for row in history] == [1, 2]
    assert_tree_equal([row["losses"] for row in reference], [row["losses"] for row in history])
    assert any("running_mean" in name for name in complete["state_dict"])
    steps = {int(value["step"]) for value in resumed["optimizer"]["state"].values()}
    assert steps == {2}
    assert json.loads((tmp_path / "resumed/status.json").read_text())["status"] == "completed"


def test_weights_transfer_copies_all_a_state_and_starts_fresh_optimizer(actual_paired_data, tmp_path, monkeypatch):
    a_cfg, b_cfg = fixture_config(), fixture_config(stage="single")
    parent_path = trainer.execute_run(actual_paired_data["a"], tmp_path / "a", a_cfg)
    parent = load(parent_path)
    original_build = trainer.build_comparison_model
    observations = []
    def instrument(config, device):
        model = original_build(config, device)
        original_loss = model.loss
        def loss(batch, horizon, stage="train"):
            if stage == "train" and not observations:
                observations.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return original_loss(batch, horizon, stage=stage)
        model.loss = loss
        return model
    monkeypatch.setattr(trainer, "build_comparison_model", instrument)
    b_path = trainer.execute_run(actual_paired_data["single"], tmp_path / "b", b_cfg,
                                 init_from=parent_path, expected_parent_step=2, stop_after_step=1)
    b = load(b_path)
    assert_tree_equal(observations[0], parent["state_dict"])
    assert b["step"] == 1 and b["parent_transfer"]["step"] == 2
    assert b["parent_transfer"]["optimizer_reset"]
    assert b["parent_transfer"]["sha256"] == protocol.digest(parent_path)
    assert {int(row["step"]) for row in b["optimizer"]["state"].values()} == {1}
    assert b["validation_indices"] != []


@pytest.mark.parametrize("tamper", ["stage", "parent", "pending_validation", "step"])
def test_b_transfer_rejects_wrong_stage_parent_or_endpoint(tamper, actual_paired_data, tmp_path):
    parent_path = trainer.execute_run(actual_paired_data["a"], tmp_path / "a", fixture_config())
    saved = load(parent_path)
    if tamper == "stage":
        saved["train_config"]["data"]["stage"] = "single"
    elif tamper == "parent":
        saved["parent_config_sha256"] = "a" * 64
    elif tamper == "pending_validation":
        saved["validation_status"] = "pending"
    else:
        saved["step"] = 1
    bad = tmp_path / "bad.pt"
    torch.save(saved, bad)
    with pytest.raises(ValueError):
        trainer.execute_run(actual_paired_data["single"], tmp_path / "reject_b", fixture_config(stage="single"),
                            init_from=bad, expected_parent_step=2)
    assert not (tmp_path / "reject_b/history.jsonl").exists()
    assert not (tmp_path / "reject_b/last.pt").exists()


def test_resume_rejects_changed_protocol_sources_and_dataset(actual_paired_data, tmp_path):
    cfg = fixture_config()
    p = trainer.execute_run(actual_paired_data["a"], tmp_path / "a", cfg, stop_after_step=1)
    original = load(p)
    report = protocol.preflight(actual_paired_data["a"], cfg)
    sources = protocol.source_manifest()
    for key in ("dataset_sha256", "protocol_sha256", "source_sha256"):
        altered = copy.deepcopy(original)
        altered[key] = "wrong" if key != "source_sha256" else {}
        with pytest.raises(ValueError, match="identity"):
            trainer.validate_saved(altered, cfg, report, sources)
    altered = copy.deepcopy(original)
    altered["sampler_next_step"] = 0
    with pytest.raises(ValueError, match="progress"):
        trainer.validate_saved(altered, cfg, report, sources)


def test_preflight_rejects_cross_stage_or_changed_group_protocol(actual_paired_data):
    with pytest.raises(ValueError, match="stage scope"):
        protocol.preflight(actual_paired_data["a"], fixture_config(stage="single"))
    wrong_groups = fixture_config()
    wrong_groups["data"]["expected_groups"]["train"] = 12
    with pytest.raises(ValueError, match="group count"):
        protocol.preflight(actual_paired_data["a"], wrong_groups)
    not_fixture = fixture_config()
    not_fixture["fixture_only"] = False
    with pytest.raises(ValueError, match="Tiny software fixture"):
        protocol.validate_config(not_fixture)
    not_fixture["model"]["tiny_fixture"] = False
    with pytest.raises(ValueError, match="384/64"):
        protocol.validate_config(not_fixture)


def test_stage_b_requires_explicit_initialization_or_named_scratch_route(actual_paired_data, tmp_path):
    cfg = fixture_config(stage="single")
    assert cfg["training_route"] == "staged"
    with pytest.raises(ValueError):
        trainer.execute_run(actual_paired_data["single"], tmp_path / "missing_parent", cfg)
    assert not (tmp_path / "missing_parent/last.pt").exists()
    scratch = copy.deepcopy(cfg)
    scratch["training_route"] = "scratch"
    saved = load(trainer.execute_run(actual_paired_data["single"], tmp_path / "scratch", scratch,
                                    stop_after_step=1))
    assert saved["parent_transfer"] is None
    assert saved["train_config"]["training_route"] == "scratch"


@pytest.mark.parametrize("tamper", ["version", "protocol", "declared_target", "nonfinite", "validation"])
def test_b_parent_internal_identity_and_finiteness_are_checked(tamper, actual_paired_data, tmp_path):
    p = trainer.execute_run(actual_paired_data["a"], tmp_path / "a", fixture_config())
    parent = load(p)
    if tamper == "version":
        parent["checkpoint_version"] = 77
    elif tamper == "protocol":
        parent["protocol_sha256"] = "b" * 64
    elif tamper == "declared_target":
        parent["train_config"]["training"]["steps"] = 3
        parent["protocol_sha256"] = protocol.json_digest(parent["train_config"])
    elif tamper == "validation":
        parent["validation"]["score"] = float("nan")
    else:
        value = next(v for v in parent["state_dict"].values() if v.is_floating_point())
        value.view(-1)[0] = float("nan")
    bad = tmp_path / "bad.pt"
    torch.save(parent, bad)
    with pytest.raises(ValueError):
        trainer.execute_run(actual_paired_data["single"], tmp_path / "b", fixture_config(stage="single"),
                            init_from=bad, expected_parent_step=2)
    assert not (tmp_path / "b/last.pt").exists()


def test_preflight_recomputes_declared_parent_hash(actual_paired_data, tmp_path):
    path = tmp_path / "corrupt_parent.h5"
    shutil.copy2(actual_paired_data["a"], path)
    with h5py.File(path, "r+") as f:
        parent = json.loads(f.attrs["parent_config_json"])
        parent["split_seeds"]["train"] += 1
        f.attrs["parent_config_json"] = json.dumps(parent)
    with pytest.raises(ValueError):
        protocol.preflight(path, fixture_config())
