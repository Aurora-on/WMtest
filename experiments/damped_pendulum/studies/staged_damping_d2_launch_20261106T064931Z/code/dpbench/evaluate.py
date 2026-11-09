"""Frozen physical readouts on causal latent state; autonomous test rollouts.

Readout labels are a separate diagnostic budget, not world-model supervision.
The energy metric uses reference W,D and is a trajectory diagnostic, NOT a
claim of exact energy balance of the model's own discrete latent update.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from dpbench.data import load_episode
from dpbench.generate import energy
from dpbench.models import load_checkpoint, synchronize


def pixels_tensor(pixels, device):
    return torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).float().to(device) / 255.


def targets(state):
    return np.column_stack((np.sin(state[:, 0]), np.cos(state[:, 0]), state[:, 1]))


def calibration_features(model, path, split, maximum, seed, train_frames=None, batch_size=32):
    with h5py.File(path, "r") as f:
        ids = f["splits/"+split][:]
        cap = int(f.attrs["train_frames"]) if train_frames is None else int(train_frames)
        pairs = [(int(ep), t) for ep in ids for t in range(4, min(int(f["ep_len"][ep]), cap))]
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(pairs), size=min(maximum, len(pairs)), replace=False))
    pairs = [pairs[i] for i in chosen]
    feature_chunks, label_chunks, pixels, past, labels = [], [], [], [], []
    device = next(model.parameters()).device
    def flush():
        if not pixels:
            return
        history = torch.stack(pixels).to(device)
        pa = torch.tensor(np.stack(past), dtype=torch.float32, device=device)
        a = torch.zeros((len(pixels), 0, 1), device=device)
        z = model.rollout_features(history, a, past_actions=pa)[:, 0]
        feature_chunks.append(z.cpu().numpy())
        label_chunks.append(np.asarray(labels))
        pixels.clear(); past.clear(); labels.clear()
    with h5py.File(path, "r") as f:
        for ep, t in pairs:
            off = int(f["ep_offset"][ep])
            pixels.append(pixels_tensor(f["pixels"][off+t-4:off+t+1], "cpu"))
            past.append(f["action"][off+t-4:off+t])
            labels.append(targets(f["state"][off+t:off+t+1])[0])
            if len(pixels) == batch_size:
                flush()
        flush()
    return np.concatenate(feature_chunks), np.concatenate(label_chunks), pairs


class Readout:
    def __init__(self, mode, mean, scale, ymean, yscale, weights=None, network=None):
        self.mode, self.mean, self.scale = mode, mean, scale
        self.ymean, self.yscale, self.weights, self.network = ymean, yscale, weights, network

    def predict(self, x):
        shape = x.shape[:-1]
        z = (x.reshape(-1, x.shape[-1]) - self.mean) / self.scale
        if self.mode == "ridge":
            y = np.c_[z, np.ones(len(z))] @ self.weights
        else:
            with torch.no_grad():
                y = self.network(torch.as_tensor(z, dtype=torch.float32)).numpy()
        y = y * self.yscale + self.ymean
        return y.reshape(*shape, 3)

    def serialize(self):
        return dict(mode=self.mode, mean=torch.tensor(self.mean), scale=torch.tensor(self.scale),
                    ymean=torch.tensor(self.ymean), yscale=torch.tensor(self.yscale),
                    weights=None if self.weights is None else torch.tensor(self.weights),
                    network=None if self.network is None else self.network.state_dict())


def fit_readout(x, y, xv, yv, mode="ridge", steps=500, seed=0):
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-5)
    ym, ys = y.mean(0), np.maximum(y.std(0), 1e-5)
    xn, yn, vn, vyn = (x-mu)/sd, (y-ym)/ys, (xv-mu)/sd, (yv-ym)/ys
    if mode == "ridge":
        a, va = np.c_[xn, np.ones(len(xn))], np.c_[vn, np.ones(len(vn))]
        best, result = float("inf"), None
        for alpha in (1e-4, .01, .1, 1., 10.):
            penalty = np.eye(a.shape[1])*alpha
            penalty[-1, -1] = 0
            w = np.linalg.solve(a.T@a + penalty, a.T@yn)
            score = float(np.mean((va@w-vyn)**2))
            if score < best:
                best, result = score, (w, alpha)
        probe = Readout(mode, mu, sd, ym, ys, weights=result[0])
        info = dict(validation_standardized_mse=best, alpha=result[1])
    else:
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.Tanh(),
                                  torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 3))
        opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.0001)
        xx, yy = torch.tensor(xn, dtype=torch.float32), torch.tensor(yn, dtype=torch.float32)
        vx, vy = torch.tensor(vn, dtype=torch.float32), torch.tensor(vyn, dtype=torch.float32)
        rng = torch.Generator().manual_seed(seed)
        best, best_state, best_step = float("inf"), None, 0
        for step in range(1, steps+1):
            idx = torch.randint(len(xx), (min(256, len(xx)),), generator=rng)
            loss = (net(xx[idx])-yy[idx]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0 or step == steps:
                with torch.no_grad():
                    score = float((net(vx)-vy).square().mean())
                if score < best:
                    best = score
                    best_step = step
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        net.eval()
        probe = Readout(mode, mu, sd, ym, ys, network=net)
        info = dict(validation_standardized_mse=best, best_step=best_step, optimization_steps=steps,
                    hidden_layers=[128, 128], label_budget=len(x))
    return probe, info


def decode_state(pred):
    return np.stack((np.arctan2(pred[..., 0], pred[..., 1]), pred[..., 2]), axis=-1)


def metrics(pred, truth, reference_work, reference_diss, meta, passive):
    angle = np.arctan2(np.sin(pred[:, 0]-truth[:, 0]), np.cos(pred[:, 0]-truth[:, 0]))
    pe = energy(pred, meta["mass"], meta["length"], meta["gravity"])
    te = energy(truth, meta["mass"], meta["length"], meta["gravity"])
    interval = np.diff(pe) - reference_work[:-1] + reference_diss[:-1]
    return dict(angle_rmse_rad=float(np.sqrt(np.mean(angle[1:]**2))),
                omega_rmse_rad_s=float(np.sqrt(np.mean((pred[1:, 1]-truth[1:, 1])**2))),
                energy_rmse_J=float(np.sqrt(np.mean((pe[1:]-te[1:])**2))),
                reference_budget_rmse_J=float(np.sqrt(np.mean(interval**2))),
                anchor_angle_error_rad=float(abs(angle[0])),
                anchor_omega_error_rad_s=float(abs(pred[0, 1]-truth[0, 1])),
                passive_energy_increase_fraction=float(np.mean(np.diff(pe)>1e-5)) if passive else None)


def evaluate(args):
    torch.set_num_threads(args.threads)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty output folder: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    if "dataset_sha256" in checkpoint:
        digest = hashlib.sha256()
        with args.data.open("rb") as handle:
            for block in iter(lambda: handle.read(1024*1024), b""):
                digest.update(block)
        if digest.hexdigest() != checkpoint["dataset_sha256"]:
            raise ValueError("Checkpoint and evaluation dataset SHA256 differ")
    elif Path(checkpoint["data_path"]).resolve() != args.data.resolve():
        raise ValueError("Legacy checkpoint lacks dataset hash and data path differs")
    with h5py.File(args.data, "r") as f:
        dt = float(f.attrs["dt"])
        for requested in args.horizons:
            steps = round(requested/dt)
            if steps < 1 or not np.isclose(steps*dt, requested, rtol=0, atol=1e-8):
                raise ValueError(f"Horizon {requested} must be a positive integer multiple of dt={dt}")
    actual_cap = checkpoint["train_frames"]
    x, y, train_pairs = calibration_features(model, args.data, "train", args.labels, args.seed, actual_cap)
    xv, yv, val_pairs = calibration_features(model, args.data, "val", args.val_labels, args.seed, actual_cap)
    probes, probe_info = {}, {}
    for mode in args.probes:
        probes[mode], probe_info[mode] = fit_readout(x, y, xv, yv, mode, args.probe_steps, args.seed)
    torch.save({k: v.serialize() for k, v in probes.items()}, args.output/"readouts.pt")
    (args.output/"calibration.json").write_text(json.dumps(dict(train=train_pairs, val=val_pairs, probes=probe_info), indent=2))
    with h5py.File(args.data, "r") as f:
        test_ids = f["splits/test"][:].tolist()
    if args.max_episodes:
        test_ids = test_ids[:args.max_episodes]
    rows, predictions, truths, features_all, used_ids = [], [], [], [], []
    device = next(model.parameters()).device
    rollout_seconds = 0.
    for ep in test_ids:
        data = load_episode(args.data, ep)
        meta, anchor = data["metadata"], 4
        max_steps = int(round(max(args.horizons)/meta["dt"]))
        if max_steps > len(data["state"])-1-anchor:
            available = (len(data["state"])-1-anchor)*meta["dt"]
            raise ValueError(f"Requested horizon does not fit episode {ep}; maximum is {available:.3f}s. Set --horizons explicitly.")
        history = pixels_tensor(data["pixels"][:5], device)[None]
        actions = torch.tensor(data["action"][anchor:anchor+max_steps], dtype=torch.float32, device=device)[None]
        past = torch.tensor(data["action"][:4], dtype=torch.float32, device=device)[None]
        synchronize(device); t = time.perf_counter()
        z = model.rollout_features(history, actions, past_actions=past)[0].cpu().numpy()
        synchronize(device); rollout_seconds += time.perf_counter()-t
        if not np.isfinite(z).all():
            raise FloatingPointError(f"Non-finite latent rollout in episode {ep}")
        truth = data["state"][anchor:anchor+max_steps+1]
        predictions_ep = []
        for mode, probe in probes.items():
            pred = decode_state(probe.predict(z))
            if not np.isfinite(pred).all():
                raise FloatingPointError(f"Non-finite readout in episode {ep}")
            predictions_ep.append(pred)
            for requested in args.horizons:
                steps = int(round(requested/meta["dt"]))
                if steps > max_steps:
                    continue
                end = anchor+steps+1
                m = metrics(pred[:steps+1], truth[:steps+1], data["work"][anchor:end],
                            data["dissipation"][anchor:end], meta, data["mode"] == 0)
                rows.append(dict(episode=ep, mode="passive" if data["mode"] == 0 else "driven",
                                 probe=mode, horizon_s=requested, **m))
        predictions.append(np.stack(predictions_ep)); truths.append(truth); features_all.append(z); used_ids.append(ep)
    if not rows:
        raise ValueError("No requested horizon fits test episodes")
    aggregate = []
    for probe in args.probes:
        for mode in ("passive", "driven"):
            for h in args.horizons:
                selected = [r for r in rows if (r["probe"], r["mode"], r["horizon_s"]) == (probe, mode, h)]
                if not selected:
                    continue
                means, spreads = {}, {}
                for key in selected[0]:
                    if key in ("episode", "probe", "mode", "horizon_s") or selected[0][key] is None:
                        continue
                    values = [r[key] for r in selected]
                    means[key] = float(np.mean(values))
                    spreads[key] = float(np.std(values))
                aggregate.append(dict(probe=probe, mode=mode, horizon_s=h, episodes=len(selected), mean=means, episode_std=spreads))
    result = dict(model=model.kind, checkpoint_step=checkpoint["step"], seed=checkpoint["seed"],
                  checkpoint=str(args.checkpoint), data=str(args.data), probe_info=probe_info,
                  calibration_train_labels=len(x), calibration_val_labels=len(xv),
                  aggregate=aggregate, episodes=rows, rollout_seconds=rollout_seconds,
                  total_seconds=time.perf_counter()-started, independent_training_seeds=1,
                  interpretation="Physical state inferred by separately calibrated frozen probes; report anchor error. Reference-budget residual uses true trajectory W,D, not intrinsic model energy balance. Short runs are pipeline checks only.")
    (args.output/"metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(args.output/"trajectories.npz", predictions=np.stack(predictions), truth=np.stack(truths),
                        features=np.stack(features_all), episode_ids=np.array(used_ids), probe_names=np.array(args.probes),
                        dt=float(meta["dt"]))
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--probes", nargs="+", choices=["ridge", "mlp"], default=["ridge", "mlp"])
    p.add_argument("--labels", type=int, default=4096)
    p.add_argument("--val-labels", type=int, default=1024)
    p.add_argument("--probe-steps", type=int, default=500)
    p.add_argument("--horizons", type=float, nargs="+", default=[4., 8., 16., 32.])
    p.add_argument("--max-episodes", type=int, default=0, help="0 means all test episodes; positive is debug only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if min(a.labels, a.val_labels, a.probe_steps) < 1 or min(a.horizons) <= 0:
        p.error("budgets and horizons must be positive")
    evaluate(a)


if __name__ == "__main__":
    main()
