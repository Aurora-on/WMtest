"""Episode-safe windows. Physical labels are never returned to model training."""
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, path, split, window=15, train_frames=None):
        self.path = str(Path(path).resolve())
        self.window = int(window)
        self._file = None
        self.index = []
        with h5py.File(self.path, "r") as f:
            limit = int(f.attrs["train_frames"]) if train_frames is None else int(train_frames)
            for ep in f[f"splits/{split}"][:]:
                n = min(int(f["ep_len"][ep]), limit)
                off = int(f["ep_offset"][ep])
                self.index.extend((int(ep), off + t) for t in range(n - self.window + 1))
        if not self.index:
            raise ValueError(f"No {split} windows of length {window} in {path}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        _, start = self.index[i]
        sl = slice(start, start + self.window)
        return {
            "pixels": torch.from_numpy(self._file["pixels"][sl].transpose(0, 3, 1, 2).copy()).float() / 255.,
            "action": torch.from_numpy(self._file["action"][sl].astype(np.float32)),
            "dt_frame": torch.from_numpy(self._file["dt_frame"][sl].astype(np.float32)),
        }

    def __getstate__(self):
        return dict(self.__dict__, _file=None)

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            self._file.close()


def load_episode(path, episode):
    with h5py.File(path, "r") as f:
        off, n = int(f["ep_offset"][episode]), int(f["ep_len"][episode])
        sl = slice(off, off + n)
        out = {k: f[k][sl] for k in ("pixels", "action", "state", "energy", "work", "dissipation", "valid_transition")}
        out["mode"] = int(f["mode"][episode])
        out["metadata"] = {k: f.attrs[k] for k in ("mass", "length", "gravity", "damping", "dt", "train_frames")}
        return out
