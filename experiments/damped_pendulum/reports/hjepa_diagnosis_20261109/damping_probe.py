"""Inference-only damping ablation of the frozen undamped checkpoint."""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from dpbench.comparison_hjepa import HJEPAComparison
from dpbench.dissipative_data import DissipativeWindowDataset

torch.set_num_threads(2)
checkpoint = torch.load(HERE / "remote/runs/hjepa/hjepa_seed0/a/segments/0001/last.pt",
                        map_location="cpu", weights_only=False)
model = HJEPAComparison(checkpoint["model_config"])
model.load_state_dict(checkpoint["state_dict"])
model.eval()
dataset = DissipativeWindowDataset(HERE / "remote/undamped.h5", "val", 21, 32, 81, 2)
batch = dataset.batch(checkpoint["validation_indices"][:32], 32)
dataset.close()
result = {"scope": "One fixed validation batch, frozen A20000 checkpoint; inference-only removal of damping, not retraining", "n": 32}
with torch.no_grad():
    _, states = model._encode_states(batch["pixels"][:, :53])
    state = states[:, 1]
    velocity = state - states[:, 0]
    grad = model.predictor._grad_h(state, velocity)
    conservative = grad @ model.predictor._J.T
    L = model.predictor._cholesky(torch.cat([state, velocity], -1))
    dissipative = -torch.einsum("bij,bj->bi", L, torch.einsum("bji,bj->bi", L, grad))
    result["anchor_median_norms"] = {"conservative": float(conservative.norm(dim=-1).median()),
                                    "dissipative": float(dissipative.norm(dim=-1).median()),
                                    "combined": float((conservative + dissipative).norm(dim=-1).median())}
    for zero_damping in (False, True):
        previous, state = states[:, 0], states[:, 1]
        predictions = [state]
        for _ in range(32):
            velocity = state - previous
            if zero_damping:
                step = model.predictor._grad_h(state, velocity) @ model.predictor._J.T
            else:
                step = model.predictor.passive(state, velocity)
            previous, state = state, state + model.predictor.dt * step
            predictions.append(state)
        predictions = torch.stack(predictions, 1)
        result["zero_damping" if zero_damping else "original"] = {
            "mse": float((predictions[:, 1:] - states[:, 2:]).square().mean()),
            "median_step_norm": float((predictions[:, 1:] - predictions[:, :-1]).norm(dim=-1).median())}
(HERE / "damping_probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
