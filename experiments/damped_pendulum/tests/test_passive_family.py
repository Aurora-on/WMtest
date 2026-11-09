"""Data-only checks: no model, optimizer, training, or learned readout fitting."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from dpbench.generate import energy
from dpbench.passive_family import generate_family, main, plan, sample_groups, sha256, validate_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return json.loads((ROOT / "configs/passive_family_smoke.json").read_text())


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    config = json.loads((ROOT / "configs/passive_family_smoke.json").read_text())
    path = tmp_path_factory.mktemp("passive_family") / "data"
    return path, config, generate_family(config, path)


def test_formal_plan_counts_and_test_clock():
    config = json.loads((ROOT / "configs/passive_family.json").read_text())
    out = plan(config)
    assert out["seen_episodes"] == {"train": 2304, "val": 384, "test": 768}
    assert out["unseen_episodes"] == {"interpolation": 512, "extrapolation": 128}
    assert out["history_seconds"] == 2.
    assert out["test_rollout_seconds"] == 32.
    assert out["seen_frames"] == 2688 * 81 + 768 * 341
    assert out["unseen_frames"] == 640 * 341
    assert not out["training_started"] and not out["writes_files"]


def test_default_cli_is_no_write_even_with_output(tmp_path, capsys):
    output = tmp_path / "must_not_exist"
    main(["--config", str(ROOT / "configs/passive_family_smoke.json"), "--output", str(output)])
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "plan_only" and not output.exists()


def test_initial_groups_are_balanced_reproducible_and_split_before_expansion(config):
    a, b = sample_groups(config), sample_groups(config)
    unique = []
    for split in ("train", "val", "test"):
        for key in a[split]:
            np.testing.assert_array_equal(a[split][key], b[split][key])
        physical = a[split]["initial"]
        e = energy(physical, config["mass"], config["length"], config["gravity"])
        e /= config["mass"] * config["gravity"] * config["length"]
        for band, (lo, hi) in enumerate(zip(config["energy_edges"], config["energy_edges"][1:])):
            selected = a[split]["energy_band"] == band
            assert selected.sum() == config["groups"][split] // 4
            assert ((e[selected] >= lo) & (e[selected] < hi)).all()
        unique.extend(a[split]["group_id"])
    assert len(set(unique)) == len(unique)
    changed = copy.deepcopy(config)
    changed["split_seeds"]["test"] += 7
    c = sample_groups(changed)
    np.testing.assert_array_equal(a["train"]["initial"], c["train"]["initial"])
    np.testing.assert_array_equal(a["val"]["initial"], c["val"]["initial"])
    assert not np.array_equal(a["test"]["initial"], c["test"]["initial"])


@pytest.mark.parametrize("alter", [
    lambda c: c["interpolation_damping"].__setitem__(0, .2),
    lambda c: c["extrapolation_damping"].__setitem__(0, .15),
    lambda c: c["split_seeds"].__setitem__("test", c["split_seeds"]["train"]),
    lambda c: c["groups"].__setitem__("train", 5),
    lambda c: c.__setitem__("history", c["train_frames"]),
    lambda c: c["seen_damping"].__setitem__(0, -.1),
])
def test_invalid_protocol_fails_before_writes(config, tmp_path, alter):
    alter(config)
    with pytest.raises(ValueError):
        generate_family(config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_generated_split_pairing_hidden_metadata_and_unseen_test_only(generated):
    path, config, manifest = generated
    with h5py.File(path / "seen.h5") as seen, h5py.File(path / "unseen.h5") as unseen:
        assert seen.attrs["requires_environment_context"]
        assert seen.attrs["history_required"] == config["history"]
        groups = {}
        for split in ("train", "val", "test"):
            ids = seen["splits/" + split][:]
            assert len(ids) == len(config["seen_damping"]) * config["groups"][split]
            groups[split] = set(seen["group_id"][ids])
        assert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])
        assert set(unseen["group_id"][:]) == groups["test"]
        assert len(unseen["splits/train"]) == len(unseen["splits/val"]) == 0
        assert unseen.attrs["train_frames"] == 0
        assert len(unseen["evaluation/interpolation"]) == config["groups"]["test"]
        assert len(unseen["evaluation/extrapolation"]) == config["groups"]["test"]
        initial_lookup = {}
        for handle in (seen, unseen):
            np.testing.assert_array_equal(handle["damping_id"][:], handle["regime_id"][:])
            for ep, group in enumerate(handle["group_id"][:]):
                initial = handle["initial_state"][ep]
                if group in initial_lookup:
                    np.testing.assert_array_equal(initial, initial_lookup[group])
                initial_lookup[group] = initial
        assert set(seen["damping"][:]).isdisjoint(set(unseen["damping"][:]))
    assert manifest["plan"]["model_fields"] == ["pixels", "action", "dt_frame"]
    assert not manifest["training_started"]


def test_zero_actions_ragged_padding_energy_and_independent_reference(generated):
    path, config, manifest = generated
    for entry in manifest["datasets"]:
        assert sha256(path / entry["file"]) == entry["sha256"]
        with h5py.File(path / entry["file"]) as f:
            assert not f["action"][:].any() and not f["work"][:].any()
            assert (f["dissipation"][:] >= 0).all()
            for ep, (offset, length) in enumerate(zip(f["ep_offset"][:], f["ep_len"][:])):
                sl = slice(offset, offset + length)
                assert np.isfinite(f["state"][sl]).all()
                assert f["valid_transition"][offset:offset + length - 1].all()
                assert not f["valid_transition"][offset + length - 1]
                assert f["dissipation"][offset + length - 1] == 0.
                e, d = f["energy"][sl], f["dissipation"][sl]
                assert np.max(np.abs(np.diff(e) + d[:-1])) < 1e-6
                if f["damping"][ep] == 0:
                    assert not d.any()
                else:
                    assert e[-1] < e[0]
            assert f["pixels"].dtype == np.uint8 and f["pixels"].shape[1:] == (64, 64, 3)
        refs = [a["independent_reference"] for a in entry["numerical_audit"] if a["split"] == "test"]
        assert len(refs) == 2
        assert all(a["energy_bands"] == 4 and a["frames_each"] == config["test_frames"] for a in refs)
        assert all(a["state_max_abs"] < 1e-5 and a["cumulative_dissipation_max_abs"] < 1e-5 for a in refs)


def test_no_overwrite_including_partial_folder(generated, config, tmp_path):
    path, _, _ = generated
    with pytest.raises(FileExistsError):
        generate_family(config, path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "evidence.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        generate_family(config, partial)
    assert (partial / "evidence.txt").read_text() == "keep"


def test_context_audit_discloses_exact_scope_and_does_not_filter(generated):
    _, _, manifest = generated
    audit = manifest["initial_context_overlap"]
    assert "Not later-window" in audit["scope"]
    assert "without deletion" in audit["policy"]
    assert manifest["datasets"][0]["episodes"] == 24
    assert manifest["datasets"][1]["episodes"] == 8
    ambiguity = manifest["cross_condition_identical_test_contexts"]
    assert "across all seen/interpolation/extrapolation" in ambiguity["scope"]
    assert ambiguity["distinct_initial_groups_affected"] <= 4
    for collision in ambiguity["examples"]:
        assert len(collision["members"]) > 1
        assert len({m["damping"] for m in collision["members"]}) == len(collision["members"])
