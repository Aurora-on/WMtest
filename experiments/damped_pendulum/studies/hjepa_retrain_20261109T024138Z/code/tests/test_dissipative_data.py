"""Data-boundary and deterministic-sampling checks; no optimization or training."""
from collections import Counter

import h5py
import numpy as np
import pytest
import torch

from dpbench.dissipative_data import (
    DissipativeWindowDataset,
    StepBatchSampler,
    fixed_validation_indices,
    preflight_dataset,
)


@pytest.fixture
def data_path(tmp_path):
    """Small schema fixture with no physical labels at all."""
    path = tmp_path / "rgb_actions.h5"
    episodes, frames, prefix = 12, 24, 20
    with h5py.File(path, "w") as f:
        f.attrs["train_frames"] = prefix
        f.create_dataset("ep_len", data=np.full(episodes, frames, dtype=np.int64))
        f.create_dataset("ep_offset", data=np.arange(episodes, dtype=np.int64) * frames)
        f.create_dataset("group_id", data=np.arange(episodes, dtype=np.int64))
        for split, ids in (("train", np.arange(8)), ("val", [8, 9]), ("test", [10, 11])):
            f.create_dataset("splits/" + split, data=np.asarray(ids, dtype=np.int64))
        pix = np.zeros((episodes * frames, 8, 8, 3), dtype=np.uint8)
        actions = np.zeros((episodes, frames, 1), dtype=np.float32)
        for ep in range(episodes):
            pix[ep * frames:(ep + 1) * frames, :, :, 0] = ep
            pix[ep * frames:(ep + 1) * frames, :, :, 1] = np.arange(frames)[:, None, None]
            if ep % 2:
                actions[ep, :-1, 0] = np.arange(1, frames)
        f.create_dataset("pixels", data=pix)
        f.create_dataset("action", data=actions.reshape(-1, 1))
        f.create_dataset("dt_frame", data=np.full((episodes * frames, 1), .1, dtype=np.float32))
        valid = np.ones((episodes, frames), dtype=bool)
        valid[:, -1] = False
        f.create_dataset("valid_transition", data=valid.ravel())
    return path


def test_only_observations_and_causal_transitions(data_path):
    ds = DissipativeWindowDataset(data_path, "train", history=5, max_horizon=7)
    assert len(ds) == 8 * 9
    for index, ref in enumerate(ds.index):
        sample = ds[index]
        assert set(sample) == {"pixels", "action", "dt_frame"}
        assert sample["pixels"].shape == (12, 3, 8, 8)
        assert sample["action"].shape == sample["dt_frame"].shape == (11, 1)
        assert torch.allclose(sample["pixels"][:, 0, 0, 0] * 255, torch.full((12,), float(ref.episode)))
        assert ref.start_local + 12 <= 20
        if ref.mode:
            assert sample["action"][4, 0] == ref.start_local + 5
    short = ds.batch([0, 1], horizon=2)
    assert short["pixels"].shape[:2] == (2, 7)
    assert short["action"].shape[:2] == (2, 6)
    with pytest.raises(ValueError, match="exceeds"):
        ds.get_window(0, horizon=8)
    with pytest.raises(IndexError):
        ds[-1]
    ds.close()


def test_frozen_prefix_and_test_gate(data_path):
    with pytest.raises(ValueError, match="Only train/val"):
        DissipativeWindowDataset(data_path, "test", max_horizon=7)
    with pytest.raises(ValueError, match="exceeds the frozen"):
        DissipativeWindowDataset(data_path, "train", max_horizon=7, train_frames=21)
    with pytest.raises(ValueError, match="shorter"):
        DissipativeWindowDataset(data_path, "train", max_horizon=16)
    with h5py.File(data_path, "r+") as f:
        f["valid_transition"][3] = False
    with pytest.raises(ValueError, match="Invalid/padded"):
        DissipativeWindowDataset(data_path, "train", max_horizon=7)


def test_disjoint_splits_and_groups_enforced(data_path):
    with h5py.File(data_path, "r+") as f:
        f["splits/val"][0] = 0
    with pytest.raises(ValueError, match="splits overlap"):
        preflight_dataset(data_path, max_horizon=7)
    with h5py.File(data_path, "r+") as f:
        f["splits/val"][0] = 8
        f["group_id"][8] = 0
    with pytest.raises(ValueError, match="group IDs overlap"):
        preflight_dataset(data_path, max_horizon=7)


def test_step_sampler_resumes_without_iterator_state(data_path):
    ds = DissipativeWindowDataset(data_path, "train", max_horizon=7)
    sampler = StepBatchSampler(ds, batch_size=12, seed=73)
    expected = sampler.indices(11)
    sampler.indices(800)
    assert sampler.indices(11) == expected
    assert StepBatchSampler(ds, 12, 73).indices(11) == expected
    assert sampler.indices(12) != expected
    counts = Counter((ds.index[i].mode, ds.index[i].time_bin) for i in expected)
    assert set(counts.values()) == {2}
    assert len(counts) == 6
    # Equal probability per episode, rather than proportional to window count.
    counts_ep = Counter(ds.index[i].episode for step in range(400) for i in sampler.indices(step))
    assert set(counts_ep) == set(range(8))
    assert max(counts_ep.values()) - min(counts_ep.values()) < 150
    with pytest.raises(ValueError):
        sampler.indices(-1)


def test_fixed_validation_stratified_unique_and_full_coverage(data_path):
    ds = DissipativeWindowDataset(data_path, "val", max_horizon=7)
    ids = fixed_validation_indices(ds, count=12)
    assert ids == fixed_validation_indices(ds, count=12)
    assert len(ids) == len(set(ids)) == 12
    assert set(ds.index[i].episode for i in ids) == {8, 9}
    assert len(set((ds.index[i].mode, ds.index[i].time_bin) for i in ids)) == 6
    full = fixed_validation_indices(ds, count=10000)
    assert sorted(full) == list(range(len(ds)))
    with pytest.raises(ValueError, match="cover every"):
        fixed_validation_indices(ds, count=2)
    with pytest.raises(ValueError, match="only for training"):
        StepBatchSampler(ds, 12)


def test_preflight_reads_no_targets_or_test_samples(data_path):
    # Test sample fields can be unusable without influencing train/val auditing.
    with h5py.File(data_path, "r+") as f:
        f["action"][10 * 24:] = np.nan
        f["dt_frame"][10 * 24:] = -1
    report = preflight_dataset(data_path, max_horizon=7)
    assert report["target_fields_read"] == []
    assert report["test_samples_read"] is False
    assert report["training_started"] is False
    assert report["split_sizes"] == {"train": 8, "val": 2, "test": 2}
    assert len(report["dataset_sha256"]) == 64


def test_invalid_dt_is_rejected(data_path):
    with h5py.File(data_path, "r+") as f:
        f["dt_frame"][4] = 0
    with pytest.raises(ValueError, match="non-positive dt"):
        DissipativeWindowDataset(data_path, "train", max_horizon=7)


def test_hidden_regimes_balance_sampling_without_becoming_model_inputs(data_path):
    with h5py.File(data_path, "r+") as f:
        f.attrs["requires_environment_context"] = True
        f.attrs["history_required"] = 5
        f["action"][:] = 0
        f.create_dataset("regime_id", data=np.arange(12, dtype=np.int64) % 2)
        # Damping labels are deliberately unusable: the loader must not read them.
        f.create_dataset("damping", data=np.full(12, np.nan))
    ds = DissipativeWindowDataset(data_path, "train", history=5, max_horizon=7)
    ids = StepBatchSampler(ds, batch_size=12, seed=11).indices(5)
    counts = Counter((ds.index[i].regime_id, ds.index[i].time_bin) for i in ids)
    assert len(counts) == 6 and set(counts.values()) == {2}
    assert set(ds.batch(ids)) == {"pixels", "action", "dt_frame"}
    report = preflight_dataset(data_path, history=5, max_horizon=7)
    assert report["requires_environment_context"]
    assert report["sampling_metadata_fields"] == ["regime_id"]
    assert report["target_fields_read"] == []
    with pytest.raises(ValueError, match="requires history=5"):
        DissipativeWindowDataset(data_path, "train", history=6, max_horizon=7)
    ds.close()


def test_context_family_requires_regime_metadata(data_path):
    with h5py.File(data_path, "r+") as f:
        f.attrs["requires_environment_context"] = True
    with pytest.raises(ValueError, match="regime_id"):
        DissipativeWindowDataset(data_path, "train", max_horizon=7)
