"""Safeguards for a new test-only dataset; all writes use pytest temp paths."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from dpbench.data import WindowDataset
from dpbench.generate import generate, make_plan
from dpbench.holdout import (frame_hashes, generate_holdout, overlap_audit, paired_subsets,
                             sample_plan, sha256, verify_frozen, window_key, write_dataset)


@pytest.fixture
def config():
    c = json.loads((Path(__file__).parents[1] / "configs/basic.json").read_text())
    c.update(episodes=16, frames=21, train_frames=17, data_seed=2026102901, group_id_offset=1_000_000)
    return c


def test_sampling_matches_original_distribution_and_draw_order(config):
    actual = sample_plan(config)
    expected = make_plan(config)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    old = sample_plan(dict(config, data_seed=20261028))
    assert not np.array_equal(actual[0], old[0])
    assert not np.array_equal(actual[1], old[1])


def test_test_only_files_pairing_and_no_overwrite(config, tmp_path):
    initial, actions, modes = sample_plan(config)
    for b in (0., .2):
        p = tmp_path / f"b{b}.h5"
        write_dataset(p, config, b, initial, actions, modes, "predeclared")
        with h5py.File(p) as f:
            assert len(f["splits/train"]) == len(f["splits/val"]) == 0
            assert f.attrs["train_frames"] == 0
            assert len(f["splits/test_all"]) == 16
            np.testing.assert_array_equal(f["state"][:].reshape(16, 21, 2)[:, 0], initial)
            np.testing.assert_array_equal(f["action"][:].reshape(16, 21)[:, :-1], actions.astype(np.float32))
            assert not f["valid_transition"][:].reshape(16, 21)[:, -1].any()
            assert (f["action"][:].reshape(16, 21)[:, -1] == 0).all()
        with pytest.raises(ValueError, match="No train windows"):
            WindowDataset(p, "train")
        with pytest.raises((FileExistsError, OSError)):
            write_dataset(p, config, b, initial, actions, modes, "predeclared")


def test_clip_hash_includes_used_actions_and_targets():
    rng = np.random.default_rng(4)
    pixels = rng.integers(0, 256, (21, 4, 4, 3), dtype=np.uint8)
    hashes = frame_hashes(pixels)
    actions = rng.normal(size=(21, 1)).astype(np.float32)
    for spec, start, first_used_action, last_target in [("lewm_clip", 0, 2, 12), ("lehami_clip", 1, 4, 14)]:
        original = window_key(hashes, actions, start, spec)
        changed = actions.copy(); changed[first_used_action] += 1
        assert window_key(hashes, changed, start, spec) != original
        changed = actions.copy(); changed[-1] += 1
        assert window_key(hashes, changed, start, spec) == original
        altered = pixels.copy(); altered[last_target] ^= 1
        assert window_key(frame_hashes(altered), actions, start, spec) != original


def test_known_duplicate_is_caught_and_subsets_stay_paired(config, tmp_path):
    initial, actions, modes = sample_plan(config)
    old, new = tmp_path / "old.h5", tmp_path / "new.h5"
    for path in (old, new):
        write_dataset(path, config, .2, initial, actions, modes, "fixture")
    with h5py.File(old, "r+") as f:
        f.attrs["train_frames"] = 17
        del f["splits/train"]
        f.create_dataset("splits/train", data=[0, 1])
        f.create_dataset("splits/test", data=np.arange(2, 16))
    audit = overlap_audit(old, new)
    assert all(audit["counts"][k] == 16 for k in ("context_4", "context_5", "lewm_clip", "lehami_clip", "initial_state", "full_trajectory"))
    with pytest.raises(ValueError, match="physical duplicate"):
        paired_subsets([audit], 16)
    clean = {"hits": {k: [] for k in audit["hits"]}, "counts": {k: 0 for k in audit["counts"]}}
    dirty = json.loads(json.dumps(clean))
    dirty["hits"]["lewm_clip"] = [{"new_episode": 2}]
    dirty["hits"]["context_4"] = [{"new_episode": 3}]
    subsets = paired_subsets([clean, dirty], 4)
    assert subsets == {"test": [0, 1, 3], "test_strict_context": [0, 1]}


def test_frozen_content_change_is_rejected(tmp_path):
    p = tmp_path / "frozen" / "checkpoint.pt"
    p.parent.mkdir(); p.write_bytes(b"frozen model")
    manifest = {"frozen_directory": "frozen", "runs": [{"files": [{"relative_path": p.name,
                "bytes": p.stat().st_size, "sha256": sha256(p)}]}]}
    verify_frozen(tmp_path, manifest)
    p.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="Frozen artifact changed"):
        verify_frozen(tmp_path, manifest)


def test_generation_checks_provenance_and_seals_paired_test_only_output(config, tmp_path):
    old_config = dict(config, data_seed=20261028)
    old = generate(old_config, tmp_path / "old")
    root = tmp_path / "study"
    (root / "frozen").mkdir(parents=True)
    artifact = root / "frozen" / "fixture.pt"
    artifact.write_bytes(b"unit test artifact; no model evaluation")
    runs, references = [], []
    for entry in old["datasets"]:
        runs.append({"training_data_sha256": entry["sha256"], "files": [
            {"relative_path": artifact.name, "bytes": artifact.stat().st_size, "sha256": sha256(artifact)}]})
        references.append({"path": str(tmp_path / "old" / entry["file"]),
                           "sha256": entry["sha256"], "damping": entry["damping"]})
    manifest_path = root / "frozen_manifest.json"
    manifest_path.write_text(json.dumps({"frozen_directory": "frozen", "runs": runs}))
    protocol = {"frozen_manifest_sha256": sha256(manifest_path), "source_sha256": {},
                "generation": config, "references": references, "overlap_check_scope": "fixture"}
    protocol_path = root / "protocol.json"
    protocol_path.write_text(json.dumps(dict(protocol, generation=old_config)))
    with pytest.raises(ValueError, match="seeds must differ"):
        generate_holdout(root)
    assert not (root / "data").exists()
    protocol_path.write_text(json.dumps(protocol))
    report = generate_holdout(root)
    assert report["status"] == "generated_audited_unscored"
    assert report["protocol_sha256"] == sha256(protocol_path)
    for entry in report["datasets"]:
        assert entry["sha256"] == sha256(root / entry["file"])
        with h5py.File(root / entry["file"]) as f:
            np.testing.assert_array_equal(f["splits/test"][:], report["paired_splits"]["test"])
            assert len(f["splits/train"]) == len(f["splits/val"]) == 0
    with pytest.raises(FileExistsError):
        generate_holdout(root)
    manifest_path.write_text(manifest_path.read_text() + " ")
    with pytest.raises(ValueError, match="Frozen manifest changed"):
        generate_holdout(root)
