#!/usr/bin/env python3
"""Frozen final-step dimension comparison using train-only calibrated uniform kNN.

Default: all 64 validation episodes, anchor20, K32; test is never opened unless
--split test/both is explicitly selected. Training labels fit a diagnostic
readout only. The model receives initial21 images once, then actions/time only.
This is the existing K0 causal-state probe, not the author's K1 probe protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[2]
K_VALUES = (1, 5, 25, 50)
EXPECTED_COUNTS = {"train": 384, "val": 64, "test": 128}
EVALUATION_FILES = ("dpbench/__init__.py", "dpbench/evaluate_passive.py", "dpbench/evaluate.py",
                    "dpbench/models.py", "dpbench/data.py", "dpbench/generate.py", "dpbench/passive_family.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_helpers(code_root):
    code_root = Path(code_root).resolve()
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if (name == "dpbench" or name.startswith("dpbench.")) and filename:
            if not Path(filename).resolve().is_relative_to(code_root):
                raise RuntimeError("dpbench is already imported from another code root")
    sys.path.insert(0, str(code_root))
    from dpbench.dissipative_model import DissipativeWorldModel
    from dpbench.evaluate_passive import calibration_features, autonomous_rollout, EVALUATION_DIAGNOSTICS
    from dpbench.evaluate import pixels_tensor
    from dpbench.train_dissipative import source_hashes
    return DissipativeWorldModel, calibration_features, autonomous_rollout, pixels_tensor, source_hashes, EVALUATION_DIAGNOSTICS


def evaluation_sources(code_root, manifest=None):
    actual = {name: sha256(Path(code_root) / name) for name in EVALUATION_FILES}
    if manifest is not None:
        expected = json.loads(Path(manifest).read_text())["files"]
        if any(expected.get(name) != value for name, value in actual.items()):
            raise ValueError("Evaluation helpers differ from the supplied frozen code manifest")
    return actual


def verify_checkpoint(saved, sources, dataset_hash, expected_dim=None):
    config = saved.get("config", {})
    core = config.get("core", {})
    if (saved.get("kind") != "dissipative" or saved.get("step") != 10000
            or saved.get("validation_status") != "completed"):
        raise ValueError("Requires the final completed step10000 checkpoint, not latent-best selection")
    if (config.get("objective_mode") != "lehami" or config.get("q_dim") not in (2, 8, 32)
            or core.get("damping_mode") != "none" or core.get("potential_mode") != "raw"
            or core.get("substeps") != 3 or core.get("integrator") != "frozen_dg"
            or config.get("target_tail_frames") != 2 or config.get("momentum_reference_time") != .1
            or config.get("environment_context_dim") != 0
            or config.get("history") != 21 or config.get("dt") != 0.1):
        raise ValueError("Checkpoint does not match the undamped LeHami dimension protocol")
    if expected_dim is not None and config["q_dim"] != expected_dim:
        raise ValueError("Checkpoint dimension differs from --expected-q-dim")
    if saved.get("source_hashes") != sources or saved.get("dataset_sha256") != dataset_hash:
        raise ValueError("Frozen source/data hashes differ from the checkpoint")


def split_training_records(records, fit_count=288, seed=5705):
    ordered = sorted(records, key=lambda row: row["group_id"])
    if len({row["group_id"] for row in ordered}) != len(ordered) or not 0 < fit_count < len(ordered):
        raise ValueError("Readout fit/selection must use disjoint nonempty training groups")
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    return ([ordered[i] for i in permutation[:fit_count]], [ordered[i] for i in permutation[fit_count:]])


def read_records(handle, split, minimum_length):
    records = sorted([dict(episode=int(ep), group_id=int(handle["group_id"][ep]),
                           energy_band=int(handle["energy_band"][ep]))
                      for ep in handle["splits/" + split][:]], key=lambda row: row["group_id"])
    if len(records) != EXPECTED_COUNTS[split] or len({r["group_id"] for r in records}) != len(records):
        raise ValueError(f"Requires all {EXPECTED_COUNTS[split]} unique {split} groups")
    if any(int(handle["ep_len"][r["episode"]]) < minimum_length for r in records):
        raise ValueError(f"{split} cannot cover the requested history/horizon")
    return records


def targets_to_physics(y):
    y = np.asarray(y)
    return np.stack((np.arctan2(y[..., 0], y[..., 1]), y[..., 2]), axis=-1)


class KNNReadout:
    """Exactly the existing uniform kNN, fit-only scaling, train96 selection."""
    def __init__(self, x, y, selection_x, selection_y):
        x, y, sx, sy = [np.asarray(a, dtype=np.float64) for a in (x, y, selection_x, selection_y)]
        if (x.ndim != 2 or y.shape != (len(x), 3) or sx.ndim != 2
                or sx.shape[1] != x.shape[1] or sy.shape != (len(sx), 3)
                or len(x) < max(K_VALUES) or not len(sx)
                or not all(np.isfinite(a).all() for a in (x, y, sx, sy))):
            raise ValueError("Invalid finite training readout features/labels")
        self.x_mean, self.x_scale = x.mean(0), np.maximum(x.std(0), 1e-6)
        self.y_scale = np.maximum(y.std(0), 1e-6)
        self.x, self.y = x, y
        self.tree = cKDTree((x - self.x_mean) / self.x_scale)
        _, ids = self.tree.query((sx - self.x_mean) / self.x_scale, k=max(K_VALUES), workers=1)
        self.candidates = [dict(k=k, selection_normalized_mse=float(np.mean(
            ((y[ids[:, :k]].mean(1) - sy) / self.y_scale) ** 2))) for k in K_VALUES]
        self.k = min(self.candidates, key=lambda row: row["selection_normalized_mse"])["k"]

    def predict(self, features):
        features = np.asarray(features)
        flat = features.reshape(-1, features.shape[-1])
        finite = np.isfinite(flat).all(1)
        targets = np.full((len(flat), 3), np.nan)
        nearest = np.full(len(flat), np.nan)
        if finite.any():
            distance, ids = self.tree.query((flat[finite] - self.x_mean) / self.x_scale, k=self.k, workers=1)
            if self.k == 1:
                distance, ids = distance[:, None], ids[:, None]
            targets[finite] = self.y[ids].mean(1)
            nearest[finite] = distance[:, 0]
        return (targets_to_physics(targets).reshape(*features.shape[:-1], 2),
                nearest.reshape(features.shape[:-1]))

    def save(self, path, fit, selection):
        np.savez_compressed(path, fit_features=self.x, fit_targets=self.y,
                            x_mean=self.x_mean, x_scale=self.x_scale, y_scale=self.y_scale,
                            selected_k=self.k, fit_group_ids=[r["group_id"] for r in fit],
                            selection_group_ids=[r["group_id"] for r in selection])


def physical_metrics(pred, truth, omega_scale):
    if not len(pred):
        return dict(angle_mae_rad=None, angle_rmse_rad=None, omega_rmse_rad_s=None, omega_nrmse=None)
    angle = np.arctan2(np.sin(pred[..., 0] - truth[..., 0]), np.cos(pred[..., 0] - truth[..., 0]))
    omega_rmse = float(np.sqrt(np.mean((pred[..., 1] - truth[..., 1]) ** 2)))
    return dict(angle_mae_rad=float(np.mean(abs(angle))), angle_rmse_rad=float(np.sqrt(np.mean(angle ** 2))),
                omega_rmse_rad_s=omega_rmse, omega_nrmse=omega_rmse / max(float(omega_scale), 1e-6))


def motion_metrics(pred, truth, omega_scale, dt=.1):
    """Truth-defined activity and 1s windows; no encoder-dependent mask."""
    if not len(pred):
        return dict(velocity_rms_ratio=None, velocity_cosine=None, direction_agreement=None,
                    stopped_dynamic_window_fraction=None, trajectory_with_stop_fraction=None)
    pw, tw = pred[..., 1], truth[..., 1]
    pt, tt = float(np.mean(pw ** 2)), float(np.mean(tw ** 2))
    active = abs(tw) > .1 * omega_scale
    result = dict(velocity_rms_ratio=float(np.sqrt(pt / tt)) if tt > 0 else None,
                  velocity_cosine=float(np.mean(pw * tw) / np.sqrt(pt * tt)) if pt * tt > 0 else None,
                  direction_agreement=float(np.mean(np.sign(pw[active]) == np.sign(tw[active]))) if active.any() else None)
    width = round(1 / dt)
    if pw.shape[1] < width:
        return dict(result, stopped_dynamic_window_fraction=None, trajectory_with_stop_fraction=None,
                    stop_status="NA: horizon shorter than1second", dynamic_window_count=0)
    tp = np.mean(np.lib.stride_tricks.sliding_window_view(tw ** 2, width, axis=1), axis=-1)
    pp = np.mean(np.lib.stride_tricks.sliding_window_view(pw ** 2, width, axis=1), axis=-1)
    dynamic = tp > (.1 * omega_scale) ** 2
    stopped = dynamic & (pp < .01 * tp)
    supported = dynamic.any(1)
    return dict(result, stopped_dynamic_window_fraction=float(stopped.sum() / dynamic.sum()) if dynamic.any() else None,
                trajectory_with_stop_fraction=float(stopped.any(1)[supported].mean()) if supported.any() else None,
                dynamic_window_count=int(dynamic.sum()), stopped_dynamic_window_count=int(stopped.sum()),
                dynamic_trajectory_count=int(supported.sum()), stop_status="truth-defined one-second windows")


def physical_energy(state, physical):
    return (.5 * physical["mass"] * physical["length"] ** 2 * state[..., 1] ** 2
            + physical["mass"] * physical["gravity"] * physical["length"] * (1 - np.cos(state[..., 0])))


def energy_metrics(pred, truth, physical):
    if not len(pred):
        return dict(initial_energy_relative_error_median=None, max_relative_drift_median=None, max_relative_drift_p90=None)
    pe, te = physical_energy(pred, physical), physical_energy(truth, physical)
    floor = 1e-6 * physical["mass"] * physical["gravity"] * physical["length"]
    scale = np.maximum(abs(te[:, 0]), floor)
    drift = np.max(abs(pe[:, 1:] - pe[:, :1]), axis=1) / scale
    return dict(initial_energy_relative_error_median=float(np.median(abs(pe[:, 0] - te[:, 0]) / scale)),
                max_relative_drift_median=float(np.median(drift)), max_relative_drift_p90=float(np.quantile(drift, .9)),
                normalization="True energy at anchor, floor1e-6*m*g*L; drift from decoded anchor energy")


def baseline_predictions(anchor, steps, dt=.1):
    hold = np.repeat(anchor[:, None], steps + 1, axis=1)
    hold[:, 1:, 1] = 0
    constant = np.repeat(anchor[:, None], steps + 1, axis=1)
    angle = anchor[:, None, 0] + np.arange(steps + 1)[None, :] * dt * anchor[:, None, 1]
    constant[..., 0] = np.arctan2(np.sin(angle), np.cos(angle))
    return dict(hold=hold, constant_velocity=constant)


def summarize_horizon(pred, teacher, truth, valid, records, horizon, omega_scale, physical):
    complete = (valid >= horizon) & np.isfinite(pred[:, :horizon + 1]).all((1, 2))
    masks = {"all": np.ones(len(records), dtype=bool)}
    masks.update({f"energy_band{band}": np.array([r["energy_band"] == band for r in records])
                  for band in sorted({r["energy_band"] for r in records})})
    baselines = {"decoded_" + key: value for key, value in baseline_predictions(pred[:, 0], horizon).items()}
    baselines.update({"oracle_" + key: value for key, value in baseline_predictions(truth[:, 0], horizon).items()})
    groups = {}
    for name, mask in masks.items():
        eligible = mask & complete
        pp, tt = pred[eligible, :horizon + 1], truth[eligible, :horizon + 1]
        metrics = physical_metrics(pp[:, 1:], tt[:, 1:], omega_scale)
        metrics.update(motion_metrics(pp[:, 1:], tt[:, 1:], omega_scale))
        groups[name] = dict(n_total=int(mask.sum()), n_complete=int(eligible.sum()),
                            coverage=float(eligible.sum() / mask.sum()), metrics=metrics,
                            physical_energy=energy_metrics(pp, tt, physical),
                            baselines_matched_support={key: physical_metrics(value[eligible, 1:], tt[:, 1:], omega_scale)
                                                       for key, value in baselines.items()})
    observed = physical_metrics(teacher[:, 1:horizon + 1], truth[:, 1:horizon + 1], omega_scale)
    teacher_gate = observed["angle_mae_rad"] <= .1 and observed["omega_nrmse"] <= .15
    overall = groups["all"]["metrics"]
    score = (.5 * (overall["angle_mae_rad"] / np.pi + overall["omega_nrmse"])
             if overall["angle_mae_rad"] is not None else None)
    return dict(horizon_steps=horizon, horizon_seconds=.1 * horizon, groups=groups,
                teacher_metrics=observed, teacher_quality_gate_passed=bool(teacher_gate),
                conditional_physical_score=score, ranking_eligible=bool(teacher_gate and complete.all()),
                score_definition="0.5*(future-time angleMAE/pi+omegaNRMSE), conditional on complete trajectories; compare with coverage and teacher quality")


def collect_rollouts(model, path, records, horizon, batch_size, device, autonomous, pixels_tensor, diagnostic_names):
    truth, feature, energy, valid, failures = [], [], [], [], []
    diagnostics = {key: [] for key in diagnostic_names}
    with h5py.File(path, "r") as handle:
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            images, actions, intervals = [], [], []
            for row in rows:
                offset = int(handle["ep_offset"][row["episode"]])
                images.append(pixels_tensor(handle["pixels"][offset:offset + 21], "cpu"))
                actions.append(handle["action"][offset:offset + 20 + horizon])
                intervals.append(handle["dt_frame"][offset:offset + 20 + horizon])
                truth.append(handle["state"][offset + 20:offset + 21 + horizon])
            action = torch.as_tensor(np.stack(actions), device=device)
            dt = torch.as_tensor(np.stack(intervals), device=device)
            if not torch.all(action == 0) or not torch.allclose(dt.double(), torch.full_like(dt.double(), .1)):
                raise ValueError("Evaluation requires passive data at dt=.1")
            # This is the ONLY autonomous call: exactly initial21 images, no
            # labels, future RGB, energy-band metadata or teacher states.
            out = autonomous(model, torch.stack(images).to(device), action[:, :20], dt[:, :20],
                             action[:, 20:], dt[:, 20:], diagnostic_names=diagnostic_names)
            feature.append(out["features"]); energy.append(out["latent_energy"])
            valid.extend(out["valid_steps"].tolist())
            for key in diagnostic_names:
                diagnostics[key].append(out["diagnostics"][key])
            for failure in out["failures"]:
                failures.append({**failure, "evaluation_index": start + failure["batch_index"],
                                 "episode": rows[failure["batch_index"]]["episode"]})
    features, energies, valid = np.concatenate(feature), np.concatenate(energy), np.asarray(valid)
    if (features.shape != (len(records), horizon + 1, 2 * model.q_dim)
            or energies.shape != (len(records), horizon + 1) or valid.shape != (len(records),)
            or not np.issubdtype(valid.dtype, np.integer)):
        raise ValueError("Unexpected rollout shape")
    for index, count in enumerate(valid):
        if (not 0 <= count <= horizon or not np.isfinite(features[index, :count + 1]).all()
                or not np.isfinite(energies[index, :count + 1]).all()):
            raise ValueError("Invalid declared rollout prefix")
        if count < horizon and not np.isnan(features[index, count + 1:]).all():
            raise ValueError("Failed suffix must remain NaN, never hold the last state")
    return dict(features=features, truth=np.asarray(truth), latent_energy=energies, valid_steps=valid,
                failures=failures, diagnostics={key: np.concatenate(value) for key, value in diagnostics.items()})


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Use a new evaluation output directory")
    if args.batch_size != 32 or not 32 <= args.horizon <= 320:
        raise ValueError("Protocol fixes batch32 and supports horizons32..320")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(args.threads)
    evaluation_hashes = evaluation_sources(args.code_root, args.code_manifest)
    Model, calibration, autonomous, pixels_tensor, source_hashes, diagnostic_names = load_frozen_helpers(args.code_root)
    sources, data_hash = source_hashes(), sha256(args.data)
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    verify_checkpoint(saved, sources, data_hash, args.expected_q_dim)
    splits = ("val", "test") if args.split == "both" else (args.split,)
    with h5py.File(args.data, "r") as handle:
        if handle.attrs.get("training_scope") != "undamped_stage_a" or handle.attrs.get("renderer") != "smooth_v1":
            raise ValueError("Requires the unchanged undamped smooth_v1 data")
        records = {"train": read_records(handle, "train", 81)}
        for split in splits:  # default val NEVER opens splits/test
            records[split] = read_records(handle, split, 21 + args.horizon)
        physical = {key: float(handle.attrs[key]) for key in ("mass", "length", "gravity")}
    partitions = [{r["group_id"] for r in rows} for rows in records.values()]
    if any(partitions[i] & partitions[j] for i in range(len(partitions)) for j in range(i)):
        raise ValueError("Training and evaluation groups overlap")
    fit, selection = split_training_records(records["train"])
    model = Model(saved["config"]).to(args.device).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = state_hash(model)
    args.output.mkdir(parents=True)
    def phase(name):
        print(json.dumps(dict(phase=name, elapsed_seconds=time.perf_counter() - started)), flush=True)
    phase("fit_train288_readout_features")
    fit_pairs = [(r["episode"], t) for r in fit for t in range(20, 81)]
    selection_pairs = [(r["episode"], t) for r in selection for t in range(20, 81)]
    x, y = calibration(model, args.data, fit_pairs, 32)
    phase("select_k_on_train96_only")
    sx, sy = calibration(model, args.data, selection_pairs, 32)
    readout = KNNReadout(x, y, sx, sy)
    readout.save(args.output / "readout.npz", fit, selection)
    omega_scale = float(readout.y_scale[2])
    output = {}
    for split in splits:
        rows = records[split]
        directory = args.output / split
        directory.mkdir()
        phase(f"{split}_observed_history_readout")
        pairs = [(r["episode"], t) for r in rows for t in range(20, 21 + args.horizon)]
        teacher_features, _ = calibration(model, args.data, pairs, 32)
        teacher_features = teacher_features.reshape(len(rows), args.horizon + 1, 2 * model.q_dim)
        if not np.isfinite(teacher_features).all():
            raise FloatingPointError("Observed-history encoder produced nonfinite features")
        teacher, teacher_distance = readout.predict(teacher_features)
        phase(f"{split}_autonomous_K{args.horizon}")
        rollout = collect_rollouts(model, args.data, rows, args.horizon, 32, args.device,
                                   autonomous, pixels_tensor, diagnostic_names)
        pred, distance = readout.predict(rollout["features"])
        metadata = {key: np.array([r[key] for r in rows], dtype=np.int64) for key in ("episode", "group_id", "energy_band")}
        selected = np.array([min((i for i, r in enumerate(rows) if r["energy_band"] == band),
                                 key=lambda i: rows[i]["group_id"]) for band in sorted(set(metadata["energy_band"]))])
        # Display-only future RGB is read after autonomous prediction completed.
        with h5py.File(args.data, "r") as handle:
            images = np.stack([handle["pixels"][int(handle["ep_offset"][rows[i]["episode"]]) + 20:
                                                int(handle["ep_offset"][rows[i]["episode"]]) + 21 + args.horizon] for i in selected])
        np.savez_compressed(directory / "trajectories.npz", pred=pred, teacher=teacher,
            truth=rollout["truth"], features=rollout["features"], teacher_features=teacher_features,
            valid_steps=rollout["valid_steps"], latent_energy=rollout["latent_energy"],
            nearest_fit_distance=distance, teacher_nearest_fit_distance=teacher_distance,
            time=np.arange(args.horizon + 1) * .1, selected_indices=selected, **metadata,
            **{"diagnostic_" + key: value for key, value in rollout["diagnostics"].items()})
        np.savez_compressed(directory / "animation_inputs.npz", true_rgb=images,
            truth=rollout["truth"][selected], teacher_prediction=teacher[selected],
            autonomous_prediction=pred[selected], valid_steps=rollout["valid_steps"][selected],
            time=np.arange(args.horizon + 1) * .1, **{key: value[selected] for key, value in metadata.items()})
        horizons = sorted(set((8, 32, args.horizon)))
        curves = []
        for step in range(1, args.horizon + 1):
            eligible = rollout["valid_steps"] >= step
            curves.append(dict(step=step, seconds=.1 * step, coverage=float(eligible.mean()),
                               **physical_metrics(pred[eligible, step], rollout["truth"][eligible, step], omega_scale)))
        result = dict(split=split, episode_count=len(rows), anchor_frame=20,
                      horizon_steps=args.horizon, horizon_seconds=.1 * args.horizon,
                      rollout={str(h): summarize_horizon(pred, teacher, rollout["truth"], rollout["valid_steps"], rows, h,
                                                        omega_scale, physical) for h in horizons},
                      curves=curves, failures=rollout["failures"],
                      solver_failure_trajectory_fraction=float(np.mean(rollout["valid_steps"] < args.horizon)),
                      selected_records=[dict(rows[i], evaluation_index=int(i)) for i in selected],
                      selection="Minimum group_id in each energy band, independent of prediction quality")
        write_json(directory / "summary.json", result)
        output[split] = result
    after = state_hash(model)
    if after != before:
        raise RuntimeError("Frozen world-model state changed")
    summary = dict(status="completed", checkpoint_step=10000, q_dim=model.q_dim, phase_dim=2 * model.q_dim,
                   checkpoint_sha256=checkpoint_hash, dataset_sha256=data_hash, source_hashes=sources,
                   evaluation_source_hashes=evaluation_hashes, script_sha256=sha256(__file__),
                   code_manifest_sha256=sha256(args.code_manifest) if args.code_manifest else None,
                   code_root=str(args.code_root.resolve()), device=args.device, torch=str(torch.__version__),
                   model_hash_before=before, model_hash_after=after, parameters_unchanged=True,
                   world_model_optimizer_created=False, world_model_updates=0,
                   test_split_accessed="test" in splits, future_observations_passed_to_predictor=False,
                   calibration=dict(fit_groups=288, selection_groups=96, frames_per_group=61, seed=5705,
                                    fit_pairs_sha256=hashlib.sha256(np.asarray(fit_pairs, dtype=np.int64).tobytes()).hexdigest(),
                                    selection_pairs_sha256=hashlib.sha256(np.asarray(selection_pairs, dtype=np.int64).tobytes()).hexdigest(),
                                    kind="K0 causal observed phase state; same frozen readout for teacher and autonomous states",
                                    standardization="Fit288 only; per-feature and target std floor1e-6",
                                    selected_k=readout.k, k_candidates=readout.candidates, weights="uniform",
                                    distance="Euclidean in fit-standardized phase features"),
                   omega_nrmse_denominator=omega_scale, evaluation=output,
                   metrics_scope="Future frames excluding anchor; equal episode/time weights on complete support. Failures retain NaN suffix and reduce coverage. Baselines use identical support.",
                   protocol_update="Replaces obsolete design-only fixed25/distance-weighted proposal with the existing nonlinear visualization protocol: train-only selected uniform KNN.",
                   limitations=["KNN always maps finite queries to training labels, even off-manifold; decoded quality alone is not latent dynamics proof.",
                                "Final10000 model only; this single-seed comparison does not establish statistical significance.",
                                "At10000 the curriculum trained only throughK16; K32 is extrapolation.",
                                "Test is opt-in and must not be used to choose dimension, k, architecture or thresholds."],
                   elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / "summary.json", summary)
    phase("completed")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=ROOT)
    parser.add_argument("--code-manifest", type=Path)
    for name in ("checkpoint", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--split", choices=("val", "test", "both"), default="val")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--expected-q-dim", type=int, choices=(2, 8, 32))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
