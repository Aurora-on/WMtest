"""Recompute diagnostics from the frozen job47111 saved passive rollouts.

Run with the project environment, from any working directory:
  .venv/bin/python studies/lehami_stall_comparison_20261105/audit_custom47111_saved_latents.py

No training, parameter updates, or autonomous rollout is performed. The only
model calculations are derivatives evaluated at already-saved latent states.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from dpbench.models import load_checkpoint  # noqa: E402

OUTPUT = Path(__file__).with_name("custom47111_saved_latent_audit.json")
ROOT = PROJECT / "received/matrix_47111_review_20261029/runs/server_47111"
FROZEN = PROJECT / "studies/independent_test_20261029T070556Z/frozen/runs/server_47111"
WINDOWS = [("0_to_1s", 0, 10), ("7_to_8s", 70, 80), ("31_to_32s", 310, 320)]


def fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"absolute_path": str(path.resolve()), "sha256": digest.hexdigest()}


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def validate_previous_numbers(previous: dict, current: dict) -> dict:
    """Check every original numeric metric before replacing the prior report."""
    count = 0
    max_difference = 0.0
    for name, old in previous.get("runs", {}).items():
        new = current["runs"][name]
        for group in ("windows", "loss_first100mean", "loss_last100mean"):
            old_group, new_group = old[group], new[group]
            pairs = (
                [(old_group[w][k], new_group[w][k]) for w in old_group for k in old_group[w]]
                if group == "windows" else [(old_group[k], new_group[k]) for k in old_group]
            )
            for left, right in pairs:
                difference = abs(float(left) - float(right))
                if not np.isclose(left, right, atol=1e-7, rtol=1e-6):
                    raise AssertionError(f"Previous result changed: {name}/{group}: {left} -> {right}")
                count += 1
                max_difference = max(max_difference, difference)
    return {"compared_numeric_metrics": count, "maximum_absolute_difference": max_difference,
            "absolute_tolerance": 1e-7, "relative_tolerance": 1e-6, "passed": True}


def main() -> None:
    torch.set_num_threads(4)
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    out = {
        "scope": "Retrospective analysis of saved custom-adaptation job47111 passive rollouts only. No training or autonomous rerollout. CPU derivative checks on saved latent states. Old test set, 32 passive episodes per model.",
        "loss": "MSE(q_pred,q_target) + MSE(p_pred,p_SG_target) + 0.25 SIGReg(q_target), averaged over 8 autoregressive steps. All target encoding differentiable.",
        "method": {
            "script": fingerprint(Path(__file__)),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "device": "cpu", "torch_threads": 4,
            "passive_selection": "episode_id % 2 == 0, matching modes=np.arange(n)%2 in dpbench/generate.py:141; this saved test set has 32 passive episodes.",
            "feature_layout": "features[episode, time, coordinate], first 32 coordinates q and last 32 p; no physical-unit interpretation of latent RMS.",
            "prediction_readout": "MLP; selected by probe_names rather than assuming its index.",
            "time_origin": "Saved state index 0 is the rollout anchor (raw video frame 4, simulation time 0.4 seconds). Table times are elapsed from this anchor, with dt=0.1 seconds.",
            "window_definition": "For each [lo,hi), q and p use saved states at indices lo,...,hi-1, while dq,dp and angular increments use transitions k->k+1 for those same k. Therefore state sample times cover [0,0.9], [7,7.9], [31,31.9] seconds and transition intervals cover [0,1], [7,8], [31,32] seconds.",
            "window_indices": {name: {"lo_inclusive": lo, "hi_exclusive": hi} for name, lo, hi in WINDOWS},
            "aggregation": "RMS over all selected episodes, window indices, and latent coordinates; angular increments use shortest wrapped difference in radians. Tanh saturation fractions pool episodes, time, and hidden units.",
            "derivative_check": "Zero physical action is encoded by the frozen action encoder. Compute p_half=p-0.5*dV/dq and dK(p_half/s)/dp_half on saved states with gauge-time step h=1. Compare this to the saved q[k+1]-q[k].",
            "interpretation_limits": "Undamped b=0 truth remains oscillatory, so diminishing predicted motion is spurious. Damped b=0.2 truth naturally slows; its late visual stillness alone is not a failure claim. Both conditions show latent p growth and K-network saturation. These checks identify the mechanism of small predicted q increments, not the original cause of long-horizon p drift. They do not establish that joint SIGReg would constrain predicted-p divergence.",
        },
        "runs": {},
    }
    for folder in sorted(ROOT.glob("lehami*")):
        features_path = folder / "evaluation/trajectories.npz"
        checkpoint_path = FROZEN / folder.name / "best.pt"
        history_path = folder / "history.jsonl"
        with np.load(features_path) as saved:
            assert np.isclose(float(saved["dt"]), .1)
            mask = saved["episode_ids"] % 2 == 0
            mlp_index = list(saved["probe_names"]).index("mlp")
            z = saved["features"][mask]
            y = saved["predictions"][mask, mlp_index]
            truth = saved["truth"][mask]
            episode_ids = saved["episode_ids"][mask].tolist()
        difference = np.diff(z, axis=1)
        model, checkpoint = load_checkpoint(checkpoint_path)
        flow = model.model.predictor.flow
        kinetic = flow.H_net.K_net
        history = [json.loads(line) for line in history_path.read_text().splitlines()]
        record = {
            "inputs": {"checkpoint": fingerprint(checkpoint_path),
                       "features": fingerprint(features_path),
                       "training_history": fingerprint(history_path)},
            "checkpoint_step": checkpoint["step"],
            "passive_episodes": int(mask.sum()),
            "passive_episode_ids": episode_ids,
            "log_p_scale_exp_mean": float(flow._get_p_scale().mean().detach()),
            "windows": {},
        }
        for label, lo, hi in WINDOWS:
            q = torch.tensor(z[:, lo:hi, :32].reshape(-1, 32), requires_grad=True)
            p = torch.tensor(z[:, lo:hi, 32:].reshape(-1, 32), requires_grad=True)
            u = model.model.action_encoder(torch.zeros((q.shape[0], 1, 1))).squeeze(1)
            dv = torch.autograd.grad(flow.H_net.V(q, u).sum(), q)[0]
            p_half = (p - .5 * dv).detach().requires_grad_(True)
            scaled_p = p_half / flow._get_p_scale().detach()
            dk = torch.autograd.grad(flow.H_net.K(scaled_p).sum(), p_half)[0]
            first = kinetic[1](kinetic[0](scaled_p)).detach()
            second = kinetic[3](kinetic[2](first)).detach()
            angle_prediction = np.angle(np.exp(1j * np.diff(y, axis=1)[:, lo:hi, 0]))
            angle_truth = np.angle(np.exp(1j * np.diff(truth, axis=1)[:, lo:hi, 0]))
            record["windows"][label] = {
                "q_rms": rms(q.detach()), "p_rms": rms(p.detach()),
                "dq_rms": rms(difference[:, lo:hi, :32]),
                "dp_rms": rms(difference[:, lo:hi, 32:]),
                "mlp_pred_angle_increment_rms_rad": rms(angle_prediction),
                "true_angle_increment_rms_rad": rms(angle_truth),
                "dK_dp_at_halfstep_rms": rms(dk.detach()),
                "saved_dq_vs_Kgrad_max_abs": float(np.max(np.abs(
                    difference[:, lo:hi, :32].reshape(-1, 32) - dk.detach().numpy()))),
                "kinetic_first_tanh_abs_above_099_fraction": float((first.abs() > .99).float().mean()),
                "kinetic_second_tanh_abs_above_099_fraction": float((second.abs() > .99).float().mean()),
            }
        for name, rows in [("loss_first100mean", history[:100]), ("loss_last100mean", history[-100:])]:
            record[name] = {key: float(np.mean([row[key] for row in rows]))
                            for key in ["loss", "pos_loss", "mom_loss", "sigreg_loss"]}
        out["runs"][folder.name] = record
    out["reproduction_check"] = validate_previous_numbers(previous, out)
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), **out["reproduction_check"]}))


if __name__ == "__main__":
    main()
