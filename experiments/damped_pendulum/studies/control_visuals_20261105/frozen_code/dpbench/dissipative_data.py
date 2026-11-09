"""Read-only, episode-safe RGB/action windows for the dissipative model.

Training and checkpoint selection are restricted to the existing train/val
episode partitions and their declared training prefix. No physical target
dataset is opened. Action mode is derived from the permitted action prefix and
is used only to balance sampling, never returned to the model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


MODEL_FIELDS = ("pixels", "action", "dt_frame")


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class WindowRef:
    """Sampler metadata; never a world-model input or learning target."""

    episode: int
    start: int
    start_local: int
    mode: int
    time_bin: int
    regime_id: int | None = None


def window_stratum(ref):
    """Opaque regime IDs balance environments; they never enter model batches."""
    regime = getattr(ref, "regime_id", None)
    return (ref.mode, ref.time_bin) if regime is None else (regime, ref.mode, ref.time_bin)


def stratum_label(key):
    mode, time_bin = key[-2:]
    prefix = f"regime{key[0]}_" if len(key) == 3 else ""
    return f"{prefix}mode{mode}_time{time_bin}"


def _target_tail_frames(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value not in (0, 2):
        raise ValueError("target_tail_frames must be 0 or 2")
    return int(value)


def _layout(handle, history, max_horizon, train_frames, target_tail_frames=0):
    history = _positive_int(history, "history")
    max_horizon = _positive_int(max_horizon, "max_horizon")
    target_tail_frames = _target_tail_frames(target_tail_frames)
    if history < 2:
        raise ValueError("At least two history frames are required for velocity initialization")
    required = ("pixels", "action", "dt_frame", "ep_offset", "ep_len", "valid_transition", "splits")
    missing = [key for key in required if key not in handle]
    if missing or "train_frames" not in handle.attrs:
        raise ValueError(f"Missing required data fields or train_frames attribute: {missing}")
    declared = _positive_int(handle.attrs["train_frames"], "HDF5 train_frames")
    limit = declared if train_frames is None else _positive_int(train_frames, "train_frames")
    if limit > declared:
        raise ValueError(f"train_frames={limit} exceeds the frozen HDF5 prefix {declared}")
    if limit < history + max_horizon + target_tail_frames:
        raise ValueError("Training prefix is shorter than history + max_horizon + target_tail_frames")

    pixels, action, dt = (handle[key] for key in MODEL_FIELDS)
    if pixels.ndim != 4 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
        raise ValueError("pixels must be uint8 [total_frames, height, width, 3]")
    total = len(pixels)
    if action.ndim != 2 or action.shape[0] != total or action.shape[1] < 1:
        raise ValueError("action must have shape [total_frames, action_dim]")
    if dt.shape != (total, 1) or handle["valid_transition"].shape != (total,):
        raise ValueError("dt_frame or valid_transition has an invalid shape")
    offsets, lengths = handle["ep_offset"][:], handle["ep_len"][:]
    if offsets.ndim != 1 or lengths.shape != offsets.shape or not len(offsets):
        raise ValueError("Invalid episode tables")
    if not np.issubdtype(offsets.dtype, np.integer) or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError("Episode tables must contain integers")
    if np.any(offsets < 0) or np.any(lengths < 1) or np.any(offsets + lengths > total):
        raise ValueError("Episode bounds exceed the stored arrays")
    ordered = np.argsort(offsets)
    if np.any(offsets[ordered][1:] < (offsets + lengths)[ordered][:-1]):
        raise ValueError("Episode frame ranges overlap")

    splits = {}
    for key in ("train", "val", "test"):
        if key not in handle["splits"]:
            raise ValueError(f"Missing split {key}")
        ids = handle[f"splits/{key}"][:]
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"Invalid episode IDs in {key}")
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(offsets)):
            raise ValueError(f"Duplicate or out-of-range episode IDs in {key}")
        splits[key] = ids.astype(np.int64)
    all_ids = np.concatenate(list(splits.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Episode splits overlap")
    if set(all_ids.tolist()) != set(range(len(offsets))):
        raise ValueError("Episode splits must partition the complete episode table")
    if not len(splits["train"]) or not len(splits["val"]):
        raise ValueError("Training and validation splits must both be nonempty")
    if "group_id" in handle:
        groups = handle["group_id"][:]
        if groups.shape != offsets.shape:
            raise ValueError("Invalid group_id shape")
        group_sets = {key: set(groups[ids].tolist()) for key, ids in splits.items()}
        if (group_sets["train"] & group_sets["val"] or group_sets["train"] & group_sets["test"]
                or group_sets["val"] & group_sets["test"]):
            raise ValueError("Episode group IDs overlap across splits")
    return history, max_horizon, limit, declared, offsets, lengths, splits


class DissipativeWindowDataset(Dataset):
    """Train/val prefixes containing history + horizon + target_tail_frames.

    Pixels have shape [T, 3, H, W] in [0, 1]. Action and dt_frame have shape
    [T-1, A] and [T-1, 1]: only transitions joining returned frames are included.
    Thus the terminal padded action is never supplied to training. The anchor
    is frame history-1 and its first future action has that same index. Optional
    tail frames support centered derivative targets only; the model must not
    roll out beyond horizon or use them in its causal initialization.
    """

    def __init__(self, path, split, history=5, max_horizon=32, train_frames=None,
                 target_tail_frames=0):
        if split not in ("train", "val"):
            raise ValueError("Only train/val may be used for training or checkpoint selection")
        self.path = str(Path(path).resolve())
        self.split = split
        self._file = None
        self.index = []
        self.strata = {}
        self.target_tail_frames = _target_tail_frames(target_tail_frames)
        with h5py.File(self.path, "r") as handle:
            (self.history, self.max_horizon, self.train_frames, self.declared_train_frames,
             offsets, lengths, splits) = _layout(handle, history, max_horizon, train_frames,
                                                 self.target_tail_frames)
            self.window = self.history + self.max_horizon + self.target_tail_frames
            self.split_ids = tuple(int(ep) for ep in splits[split])
            self.split_sizes = {key: len(ids) for key, ids in splits.items()}
            self.image_shape = tuple(handle["pixels"].shape[1:])
            self.action_dim = handle["action"].shape[1]
            self.history_required = int(handle.attrs.get("history_required", 0))
            self.requires_environment_context = bool(handle.attrs.get("requires_environment_context", False))
            if self.history_required and self.history != self.history_required:
                raise ValueError(f"Dataset requires history={self.history_required}")
            regimes = None
            if "regime_id" in handle:
                regimes = handle["regime_id"][:]
                if (regimes.shape != offsets.shape or not np.issubdtype(regimes.dtype, np.integer)
                        or np.any(regimes < 0)):
                    raise ValueError("regime_id must be a nonnegative integer per episode")
            if self.requires_environment_context and regimes is None:
                raise ValueError("Environment-conditioned data requires regime_id sampling metadata")
            self.sampling_metadata_fields = ["regime_id"] if regimes is not None else []
            self.episode_prefix_lengths = {}
            for ep in self.split_ids:
                off = int(offsets[ep])
                n = min(int(lengths[ep]), self.train_frames)
                if n < self.window:
                    raise ValueError(f"Episode {ep} cannot provide the configured maximum horizon")
                self.episode_prefix_lengths[ep] = n
                interval = slice(off, off + n - 1)
                valid = handle["valid_transition"][interval]
                act = handle["action"][interval]
                dt = handle["dt_frame"][interval]
                if not np.all(valid):
                    raise ValueError(f"Invalid/padded transition inside episode {ep}'s permitted prefix")
                if not np.isfinite(act).all() or not np.isfinite(dt).all() or not np.all(dt > 0):
                    raise ValueError(f"Non-finite action or non-positive dt in episode {ep}")
                mode = int(np.any(act != 0))
                # Equal bins of permissible anchor offsets, not physics labels.
                starts = np.arange(n - self.window + 1)
                for time_bin, chunk in enumerate(np.array_split(starts, 3)):
                    if not len(chunk):
                        continue
                    regime = None if regimes is None else int(regimes[ep])
                    for start in chunk:
                        ref = WindowRef(ep, off + int(start), int(start), mode, time_bin, regime)
                        bucket = self.strata.setdefault(window_stratum(ref), {}).setdefault(ep, [])
                        bucket.append(len(self.index))
                        self.index.append(ref)
        if not self.index:
            raise ValueError("No valid windows")

    def __len__(self):
        return len(self.index)

    def _horizon(self, horizon):
        horizon = self.max_horizon if horizon is None else _positive_int(horizon, "horizon")
        if horizon > self.max_horizon:
            raise ValueError("Requested horizon exceeds the indexed maximum")
        return horizon

    def get_window(self, i, horizon=None):
        horizon = self._horizon(horizon)
        if isinstance(i, (bool, np.bool_)) or int(i) != i or not 0 <= i < len(self):
            raise IndexError(i)
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        start = self.index[int(i)].start
        stop = start + self.history + horizon + self.target_tail_frames
        pixels = self._file["pixels"][start:stop]
        return {
            "pixels": torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).float().div_(255),
            "action": torch.from_numpy(self._file["action"][start:stop - 1].astype(np.float32)),
            "dt_frame": torch.from_numpy(self._file["dt_frame"][start:stop - 1].astype(np.float32)),
        }

    def __getitem__(self, i):
        return self.get_window(i)

    def batch(self, indices, horizon=None):
        samples = [self.get_window(i, horizon=horizon) for i in indices]
        if not samples:
            raise ValueError("Cannot create an empty batch")
        return {key: torch.stack([sample[key] for sample in samples]) for key in MODEL_FIELDS}

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        return dict(self.__dict__, _file=None)

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()


class StepBatchSampler:
    """Stateless train sampling keyed by (seed, zero-based optimizer step).

    Cycle equally through available regime/mode/time strata, choose an episode
    uniformly inside each stratum, then an anchor uniformly inside the episode.
    Sampling is with replacement, so shorter episodes do not receive less
    weight. Resume requires a step number, not a DataLoader iterator state.
    """

    def __init__(self, dataset, batch_size, seed=0):
        if dataset.split != "train":
            raise ValueError("StepBatchSampler is only for training")
        self.dataset = dataset
        self.batch_size = _positive_int(batch_size, "batch_size")
        if isinstance(seed, bool) or int(seed) != seed or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = int(seed)
        self.strata = tuple(sorted(dataset.strata))

    def indices(self, step):
        if isinstance(step, bool) or int(step) != step or step < 0:
            raise ValueError("step must be a nonnegative integer")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(step), 937]))
        indices = []
        for slot in range(self.batch_size):
            key = self.strata[(int(step) * self.batch_size + slot) % len(self.strata)]
            episodes = self.dataset.strata[key]
            ep = sorted(episodes)[int(rng.integers(len(episodes)))]
            candidates = episodes[ep]
            indices.append(candidates[int(rng.integers(len(candidates)))])
        return np.asarray(indices, dtype=np.int64)[rng.permutation(self.batch_size)].tolist()


def fixed_validation_indices(dataset, count=256, seed=1009):
    """Unique, fixed validation anchors balanced by mode/time and episode.

    Small strata may be exhausted; remaining slots are filled from nonempty
    strata. All available strata must be represented at least once.
    """
    if dataset.split != "val":
        raise ValueError("Checkpoint selection indices must come from validation")
    count = min(_positive_int(count, "count"), len(dataset))
    if count < len(dataset.strata):
        raise ValueError("Validation count must cover every available mode/time stratum")
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    queues = {}
    episodes = {}
    for key, by_ep in sorted(dataset.strata.items()):
        queues[key] = {ep: rng.permutation(ids).tolist() for ep, ids in by_ep.items()}
        episodes[key] = rng.permutation(sorted(by_ep)).tolist()
    result, cursors = [], {key: 0 for key in queues}
    while len(result) < count:
        for key in queues:
            if len(result) == count:
                break
            order = episodes[key]
            for _ in range(len(order)):
                ep = order[cursors[key] % len(order)]
                cursors[key] += 1
                if queues[key][ep]:
                    result.append(queues[key][ep].pop())
                    break
    return result


def preflight_dataset(path, history=5, max_horizon=32, train_frames=None,
                      target_tail_frames=0):
    """Audit layout and allowed inputs without fitting or reading physical targets.

    Only split membership is inspected for test episodes. No test pixels,
    actions, or outcome labels influence the sampler or validation selection.
    """
    datasets = [DissipativeWindowDataset(path, split, history, max_horizon, train_frames,
                                         target_tail_frames)
                for split in ("train", "val")]
    train, val = datasets
    report = {
        "path": str(Path(path).resolve()),
        "dataset_sha256": file_sha256(path),
        "split_disjoint": True,
        "split_audit_scope": (
            "Episode IDs and group IDs are disjoint across the existing splits. "
            "This is not an audit of exact or near-duplicate RGB clips; the original splits are unchanged."
        ),
        "split_sizes": train.split_sizes,
        "declared_train_frames": train.declared_train_frames,
        "train_frames": train.train_frames,
        "history": train.history,
        "history_required": train.history_required,
        "requires_environment_context": train.requires_environment_context,
        "sampling_metadata_fields": train.sampling_metadata_fields,
        "max_horizon": train.max_horizon,
        "target_tail_frames": train.target_tail_frames,
        "window_frames": train.window,
        "action_dim": train.action_dim,
        "image_shape": list(train.image_shape),
        "model_fields": list(MODEL_FIELDS),
        "target_fields_read": [],
        "test_samples_read": False,
        "training_started": False,
        "windows": {ds.split: len(ds) for ds in datasets},
        "strata_windows": {
            ds.split: {stratum_label(key): sum(map(len, eps.values()))
                       for key, eps in sorted(ds.strata.items())}
            for ds in datasets
        },
        "sampling_contract": "Available regime/mode/time strata balanced; episode uniform within strata; RGB/action/dt model inputs only",
        "scope": (
            "Constant physical damping per trajectory tests a special case, not identification of genuine "
            "position/velocity-dependent damping. Opaque regime IDs, when present, are sampling metadata only. "
            "No damping coefficient or other physical target is read for training."
        ),
    }
    return report
