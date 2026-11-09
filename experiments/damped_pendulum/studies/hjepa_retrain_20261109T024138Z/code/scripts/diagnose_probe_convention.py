"""Paired causal/centered SG readout diagnostic; never trains a world model.

Both probes use identical sampled labels, stay inside the checkpoint's training
frame cap, and decode the identical previously saved autonomous rollout states.
The centered features use neighboring calibration images only, never future
images from an evaluated rollout. Results are a diagnostic, not a new matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch

from dpbench.evaluate import decode_state, fit_readout, pixels_tensor, targets
from dpbench.models import load_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def collect_pair(model, path, split, maximum, seed, cap, batch_size=32):
    from lehami.derivative import p_causal_last, p_centered_at

    device = next(model.parameters()).device
    with h5py.File(path, "r") as f:
        # t+2 <= cap-1: both future calibration frames remain inside train cap.
        pool = [(int(ep), t) for ep in f["splits/" + split][:]
                for t in range(4, min(int(f["ep_len"][ep]), cap) - 2)]
        if len(pool) < maximum:
            raise ValueError(f"Only {len(pool)} eligible {split} pairs for {maximum} labels")
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(len(pool), size=maximum, replace=False))
        pairs = [pool[i] for i in chosen]
        causal, centered, ys = [], [], []
        fd_dt = float(model.cfg.model.fd_dt)
        for start in range(0, len(pairs), batch_size):
            clips, labels = [], []
            for ep, t in pairs[start:start + batch_size]:
                off = int(f["ep_offset"][ep])
                clips.append(pixels_tensor(f["pixels"][off+t-4:off+t+3], "cpu"))
                labels.append(targets(f["state"][off+t:off+t+1])[0])
            images = torch.stack(clips).to(device)
            z = model._encode(images)  # t-4 ... t+2; q_t is at index 4.
            q = z[:, 4]
            xc = torch.cat((q, p_causal_last(z[:, 1:5], fd_dt)), -1)
            xs = torch.cat((q, p_centered_at(z, 4, fd_dt)), -1)
            if start == 0:
                native = model.rollout_features(
                    images[:, :5], torch.zeros((len(images), 0, 1), device=device))[:, 0]
                torch.testing.assert_close(xc, native, rtol=1e-5, atol=1e-6)
            causal.append(xc.cpu().numpy())
            centered.append(xs.cpu().numpy())
            ys.extend(labels)
        return {"causal": np.concatenate(causal), "centered": np.concatenate(centered)}, np.asarray(ys), pairs


def scores(probe, features, truth, episode_ids, modes, dt, horizons):
    pred = decode_state(probe.predict(features))
    rows = []
    for mode_id, mode in [(0, "passive"), (1, "driven")]:
        mask = modes[episode_ids] == mode_id
        if not mask.any():
            continue
        for horizon in horizons:
            steps = round(horizon / dt)
            if not np.isclose(steps * dt, horizon) or steps >= pred.shape[1]:
                raise ValueError(f"Saved rollout cannot cover horizon {horizon}")
            p, y = pred[mask, :steps + 1], truth[mask, :steps + 1]
            angle = np.arctan2(np.sin(p[..., 0] - y[..., 0]), np.cos(p[..., 0] - y[..., 0]))
            angle_rmse = np.sqrt(np.mean(angle[:, 1:] ** 2, axis=1))
            omega_rmse = np.sqrt(np.mean((p[:, 1:, 1] - y[:, 1:, 1]) ** 2, axis=1))
            rows.append(dict(mode=mode, horizon_s=horizon, episodes=int(mask.sum()),
                             angle_rmse_rad=float(angle_rmse.mean()),
                             omega_rmse_rad_s=float(omega_rmse.mean()),
                             anchor_angle_error_rad=float(np.abs(angle[:, 0]).mean()),
                             anchor_omega_error_rad_s=float(np.abs(p[:, 0, 1] - y[:, 0, 1]).mean())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", type=int, default=4096)
    parser.add_argument("--val-labels", type=int, default=1024)
    parser.add_argument("--probe-steps", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--horizons", type=float, nargs="+", default=[8., 32.])
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing diagnosis: {args.output}")
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    if model.kind != "lehami" or model.cfg.model.enc_window != 1:
        raise ValueError("This controlled diagnostic supports the W1 LeHami arm only")
    data_hash = sha256(args.data)
    if data_hash != checkpoint.get("dataset_sha256"):
        raise ValueError("Dataset hash differs from checkpoint")
    with h5py.File(args.data, "r") as f:
        dt = float(f.attrs["dt"])
        if not np.isclose(dt, model.dt, rtol=1e-7) or not np.allclose(f["dt_frame"][:], model.dt, rtol=1e-7):
            raise ValueError("Diagnostic requires the fixed reference-rate dataset (h=1)")
        modes = f["mode"][:]
        test_ids = f["splits/test"][:]
        with np.load(args.rollouts) as saved:
            features = saved["features"].copy()
            truth = saved["truth"].copy()
            episode_ids = saved["episode_ids"].copy()
            if not np.isclose(float(saved["dt"]), dt):
                raise ValueError("Saved rollout dt differs from dataset")
        if features.shape[:2] != truth.shape[:2] or features.shape[-1] != 64:
            raise ValueError("Expected LeHami qp features aligned to truth")
        if set(episode_ids.tolist()) - set(test_ids.tolist()):
            raise ValueError("Saved rollouts include non-test episodes")
        # Bind every stored ground-truth row to this dataset, and the first
        # anchor to this exact model; no new autonomous rollout is performed.
        for i, ep in enumerate(episode_ids):
            off = int(f["ep_offset"][ep])
            np.testing.assert_allclose(truth[i], f["state"][off+4:off+4+truth.shape[1]], rtol=0, atol=0)
        off = int(f["ep_offset"][episode_ids[0]])
        h = pixels_tensor(f["pixels"][off:off+5], args.device)[None]
        anchor = model.rollout_features(h, torch.zeros((1, 0, 1), device=args.device))[0, 0].cpu().numpy()
        # Saved inference used CUDA; CPU convolution and reduction order can
        # differ by a few 1e-4. Record the discrepancy and enforce a bounded
        # numerical check in addition to the exact dataset/checkpoint binding.
        anchor_difference = float(np.max(np.abs(anchor - features[0, 0])))
        np.testing.assert_allclose(anchor, features[0, 0], rtol=1e-3, atol=1e-3)

    cap = int(checkpoint["train_frames"])
    x, y, train_pairs = collect_pair(model, args.data, "train", args.labels, args.seed, cap)
    xv, yv, val_pairs = collect_pair(model, args.data, "val", args.val_labels, args.seed, cap)
    print(json.dumps({"status": "features_collected", "seconds": time.perf_counter()-started,
                      "train_labels": len(y), "val_labels": len(yv)}), flush=True)
    result, weights = {}, {}
    for convention in ["causal", "centered"]:
        probe, info = fit_readout(x[convention], y, xv[convention], yv, "mlp", args.probe_steps, args.seed)
        result[convention] = dict(probe_info=info, aggregate=scores(probe, features, truth, episode_ids, modes, dt, args.horizons))
        weights[convention] = probe.serialize()
    out = dict(checkpoint=str(args.checkpoint), checkpoint_step=checkpoint["step"],
               world_model_seed=checkpoint["seed"], probe_seed=args.seed, data_sha256=data_hash,
               saved_rollouts_sha256=sha256(args.rollouts), train_frames=cap,
               train_pairs=train_pairs, val_pairs=val_pairs, results=result,
               seconds=time.perf_counter()-started, device=args.device,
               saved_cuda_vs_local_anchor_max_abs=anchor_difference,
               interpretation="Paired probe-only diagnostic on identical frozen rollouts; independent world-model seeds=1. Both probes use the identical labels with t+2 inside train cap. Differences from the old table also include the newly shared eligible sampling pool; compare these paired arms.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"diagnosis.json").write_text(json.dumps(out, indent=2) + "\n")
    torch.save(weights, args.output/"readouts.pt")
    print(json.dumps({k: v for k, v in out.items() if k not in ("train_pairs", "val_pairs")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
