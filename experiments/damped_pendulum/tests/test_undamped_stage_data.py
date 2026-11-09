"""Data preparation only: b=0 selection never becomes a world-model input."""
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from dpbench.dissipative_data import DissipativeWindowDataset, MODEL_FIELDS, preflight_dataset
from dpbench import passive_family as family


ROOT = Path(__file__).resolve().parents[1]


def tiny_config():
    config = json.loads((ROOT / "configs/passive_family_smoke.json").read_text())
    config.update(history=3, train_frames=6, test_frames=8, renderer="smooth_v1")
    return config


def test_formal_undamped_plan_keeps_parent_groups_and_physical_times():
    cfg = json.loads((ROOT / "configs/passive_family_smooth_v1.json").read_text())
    out = family.plan(cfg, undamped_only=True)
    assert out["seen_episodes"] == {"train": 384, "val": 64, "test": 128}
    assert out["seen_frames"] == 79936 and out["unseen_frames"] == 0
    assert out["unseen_episodes"] == {"interpolation": 0, "extrapolation": 0}
    assert out["history_seconds"] == 2.0 and out["test_rollout_seconds"] == 32.0
    assert cfg["dt"] == .1 and cfg["train_frames"] == 81 and cfg["test_frames"] == 341
    assert out["output_files"] == ["undamped.h5"]
    assert out["training_scope"] == "undamped_stage_a"
    assert not out["requires_environment_context"] and not out["writes_files"]
    assert not out["training_started"]


def test_cli_undamped_defaults_to_plan_even_when_output_is_given(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tiny_config()))
    output = tmp_path / "must_not_exist"
    family.main(["--config", str(config_path), "--undamped-only", "--output", str(output)])
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "plan_only" and result["selected_damping"] == [0.0]
    assert not output.exists()


@pytest.fixture
def generated(tmp_path, monkeypatch):
    cfg = tiny_config()
    original_simulate = family.simulate
    calls = []

    def observed_simulation(initial, action, dt, substeps, mass, length, gravity, damping):
        calls.append(damping)
        return original_simulate(initial, action, dt, substeps, mass, length, gravity, damping)

    monkeypatch.setattr(family, "simulate", observed_simulation)
    path = tmp_path / "undamped"
    manifest = family.generate_undamped(cfg, path)
    # Exactly one simulation per split, with no rejected nonzero-b simulations.
    assert calls == [0.0, 0.0, 0.0]
    monkeypatch.setattr(family, "simulate", original_simulate)
    return cfg, path, manifest


def test_subset_is_exactly_paired_with_parent_data_and_has_bound_provenance(generated, tmp_path):
    cfg, path, manifest = generated
    groups = family.sample_groups(cfg)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    parent_sha = hashlib.sha256(canonical).hexdigest()
    assert manifest["parent_config_sha256"] == parent_sha
    assert manifest["training_scope"] == "undamped_stage_a"
    assert manifest["selected_damping"] == [0.0] and len(manifest["datasets"]) == 1
    assert manifest["datasets"][0]["parent_config_sha256"] == parent_sha
    assert {p.name for p in path.glob("*.h5")} == {"undamped.h5"}
    family.generate_family(cfg, tmp_path / "full_parent")
    with h5py.File(path / "undamped.h5") as subset, h5py.File(tmp_path / "full_parent/seen.h5") as parent:
        assert subset.attrs["training_scope"] == "undamped_stage_a"
        assert "training_scope" not in parent.attrs
        assert subset.attrs["parent_config_sha256"] == parent_sha
        assert json.loads(subset.attrs["parent_config_json"]) == cfg
        assert json.loads(subset.attrs["selected_damping_json"]) == [0.0]
        assert not subset.attrs["requires_environment_context"]
        assert subset.attrs["history_required"] == cfg["history"] and subset.attrs["dt"] == cfg["dt"]
        assert subset.attrs["renderer"] == "smooth_v1"
        assert not subset["damping"][:].any() and not subset["regime_id"][:].any()
        assert not subset["action"][:].any() and not subset["dissipation"][:].any()
        split_groups = []
        for split in ("train", "val", "test"):
            ids = subset["splits/" + split][:]
            np.testing.assert_array_equal(subset["group_id"][ids], groups[split]["group_id"])
            np.testing.assert_array_equal(subset["initial_state"][ids], groups[split]["initial"])
            split_groups.append(set(subset["group_id"][ids]))
            expected_frames = cfg["test_frames"] if split == "test" else cfg["train_frames"]
            assert np.all(subset["ep_len"][ids] == expected_frames)
        assert all(not (split_groups[i] & split_groups[j]) for i in range(3) for j in range(i))
        parent_ids = np.flatnonzero(parent["damping"][:] == 0.0)
        for ep, parent_ep in enumerate(parent_ids):
            np.testing.assert_array_equal(subset["group_id"][ep], parent["group_id"][parent_ep])
            off, length = int(subset["ep_offset"][ep]), int(subset["ep_len"][ep])
            parent_off = int(parent["ep_offset"][parent_ep])
            for key in ("pixels", "state", "energy", "dissipation", "dt_frame", "valid_transition"):
                np.testing.assert_array_equal(subset[key][off:off + length], parent[key][parent_off:parent_off + length])


def test_model_windows_work_without_physical_label_datasets(generated, tmp_path):
    cfg, path, _ = generated
    # Deliberately remove every physical label from a temporary copy: training
    # window construction and preflight must still succeed without reading them.
    stripped = tmp_path / "label_free.h5"
    shutil.copyfile(path / "undamped.h5", stripped)
    with h5py.File(stripped, "r+") as handle:
        for key in ("state", "energy", "initial_state", "energy_band", "damping", "damping_id",
                    "work", "dissipation", "reference_residual"):
            del handle[key]
    report = preflight_dataset(stripped, history=cfg["history"], max_horizon=2, train_frames=cfg["train_frames"])
    assert report["target_fields_read"] == [] and not report["test_samples_read"]
    assert not report["requires_environment_context"]
    dataset = DissipativeWindowDataset(stripped, "train", history=cfg["history"], max_horizon=2)
    try:
        assert set(dataset.batch([0, 1])) == set(MODEL_FIELDS) == {"pixels", "action", "dt_frame"}
    finally:
        dataset.close()


def test_absent_zero_rejected_before_directory_creation(tmp_path):
    cfg = tiny_config()
    cfg["seen_damping"] = [.05, .2]
    family.validate_config(cfg)  # The parent is valid; selection itself is not.
    with pytest.raises(ValueError, match="requires damping 0"):
        family.generate_undamped(cfg, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_undamped_generation_refuses_nonempty_and_existing_outputs(generated, tmp_path):
    cfg, path, _ = generated
    with pytest.raises(FileExistsError):
        family.generate_undamped(cfg, path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        family.generate_undamped(cfg, partial)
    assert (partial / "keep.txt").read_text() == "keep"
