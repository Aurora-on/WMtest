"""Frozen-checkpoint diagnostic; writes JSON only, never a trained checkpoint."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dpbench.comparison_hjepa import HJEPAComparison
from dpbench.dissipative_data import DissipativeWindowDataset, StepBatchSampler

torch.set_num_threads(2)
torch.manual_seed(0)
HERE = Path(__file__).resolve().parent
data_path = HERE / "remote/undamped.h5"
checkpoint = HERE / "remote/runs/hjepa/hjepa_seed0/a/segments/0001/last.pt"
ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert hashlib.sha256(data_path.read_bytes()).hexdigest() == ckpt["dataset_sha256"]
model = HJEPAComparison(ckpt["model_config"])
model.load_state_dict(ckpt["state_dict"])


def spectrum(x):
    x = x.detach().double()
    flat = x.flatten(0, 1)
    centered = flat - flat.mean(0)
    covariance = centered.T @ centered / (len(flat) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    return {
        "mean_coordinate_variance": float(eigenvalues.mean()),
        "participation_rank": float(eigenvalues.sum().square() / eigenvalues.square().sum()),
        "variance_within_trajectory": float((x - x.mean(1, keepdim=True)).square().mean()),
        "variance_between_trajectory_means": float(x.mean(1).var(0, unbiased=False).mean()),
        "mean_square": float(x.square().mean()),
    }


results = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
           "step": ckpt["step"], "batch_size": 32, "horizon": 32,
           "scope": "One deterministic train batch and one validation batch; frozen weights; no test data"}
for split in ("train", "val"):
    dataset = DissipativeWindowDataset(data_path, split, 21, 32, 81, 2)
    indices = (StepBatchSampler(dataset, 32, seed=0).indices(19999)
               if split == "train" else ckpt["validation_indices"][:32])
    batch = dataset.batch(indices, 32)
    dataset.close()
    out = {"indices": indices}
    states_by_mode = {}
    for mode in ("eval", "train"):
        model.load_state_dict(ckpt["state_dict"])
        model.train(mode == "train")
        with torch.no_grad():
            percept, states = model._encode_states(batch["pixels"][:, :53])
            actions = model.action_encoder(batch["action"][:, 20:52])
            predictions = model._predict_states(states[:, 0], states[:, 1], actions)
        target = states[:, 2:]
        predicted_mse = float((predictions[:, 1:] - target).square().mean())
        constant_mse = float((states[:, 1:2] - target).square().mean())
        states_by_mode[mode] = states.detach()
        out[mode] = {
            "percept_spectrum": spectrum(percept[:, 1:]),
            "state_spectrum": spectrum(states[:, 1:]),
            "percept_isotropy_loss": float(model.iso(percept[:, 1:])),
            "prediction_mse": predicted_mse,
            "constant_state_mse": constant_mse,
            "prediction_mse_over_constant": predicted_mse / constant_mse,
            "median_target_step_norm": float((states[:, 2:] - states[:, 1:-1]).norm(dim=-1).median()),
            "median_predicted_step_norm": float((predictions[:, 1:] - predictions[:, :-1]).norm(dim=-1).median()),
        }
    out["batchnorm_train_eval_state_rms_difference"] = float(
        (states_by_mode["eval"] - states_by_mode["train"]).square().mean().sqrt())
    results[split] = out
    print(split, json.dumps(out, indent=2), flush=True)
(HERE / "checkpoint_probe.json").write_text(json.dumps(results, indent=2) + "\n")
