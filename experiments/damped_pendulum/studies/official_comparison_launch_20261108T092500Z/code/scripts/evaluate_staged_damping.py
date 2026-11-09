#!/usr/bin/env python3
"""Frozen A20k and staged damping evaluation, validation only.

A calibrates uniform kNN on train288, chooses k on the other train96, and
freezes the artifact. B loads that exact A readout without fitting anything.
Only initial21 RGB/action/dt enters each autonomous rollout. Physical labels
are diagnostic readout/evaluation targets, never world-model inputs. This is
the K0 causal-state probe, not the author's K1 probe protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[1]
K_VALUES = (1, 5, 25, 50)
EXPECTED_COUNTS = {"train": 384, "val": 64, "test": 128}
SEEN_DAMPING = (0., .05, .1, .2, .5, 1.)
SHARED_PREFIXES = {"encoder": "encoder.", "initializer": "velocity_correction.",
                   "potential": "core.potential_net.", "input_port": "core.input_net."}
FEATURE_KEYS = ("history", "velocity_history", "dt", "q_dim", "encoder_width",
                "velocity_correction_scale", "velocity_initializer")
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


def shared_hashes(state):
    """Exact frozen tensor identities; D/environment may differ between stages."""
    result = {}
    for module, prefix in SHARED_PREFIXES.items():
        selected = [(name, tensor) for name, tensor in sorted(state.items()) if name.startswith(prefix)]
        if not selected:
            raise ValueError(f"Missing frozen {module} tensors")
        digest = hashlib.sha256()
        for name, tensor in selected:
            value = tensor.detach().cpu().contiguous()
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite frozen tensor {name}")
            digest.update(name.encode()); digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
        result[module] = digest.hexdigest()
    return result


def feature_semantics(config):
    if any(key not in config for key in FEATURE_KEYS):
        raise ValueError("Checkpoint lacks resolved encoder/initializer configuration")
    return {key: config[key] for key in FEATURE_KEYS}


def verify_checkpoint(saved, sources, dataset_hash, stage, expected_step):
    config, training = saved.get("config", {}), saved.get("train_config", {}).get("training", {})
    core = config.get("core", {})
    if (stage not in ("a", "b") or isinstance(expected_step, bool) or expected_step < 1
            or (stage == "a" and expected_step != 20000)):
        raise ValueError("A requires exact step20000; B requires a positive expected step")
    if (saved.get("kind") != "dissipative" or saved.get("step") != expected_step
            or saved.get("validation_status") != "completed"):
        raise ValueError("Requires the exact completed, validated checkpoint step")
    if (config.get("objective_mode") != "lehami" or config.get("q_dim") != 2
            or core.get("potential_mode") != "raw" or core.get("substeps") != 3
            or core.get("integrator") != "frozen_dg" or config.get("target_tail_frames") != 2
            or config.get("momentum_reference_time") != .1 or config.get("history") != 21
            or config.get("velocity_history") != 5 or config.get("dt") != .1):
        raise ValueError("Checkpoint does not match the fixed 2D LeHami/time protocol")
    context = config.get("environment_context_dim")
    if stage == "a":
        expected_modules, scope = {"encoder", "initializer", "potential"}, "undamped_stage_a"
        if core.get("damping_mode") != "none" or context != 0:
            raise ValueError("A requires D=none and no environment context")
    else:
        if core.get("damping_mode") != "position_velocity" or context not in (0, 4):
            raise ValueError("B requires position_velocity D and context dimension0(single) or4(multi)")
        expected_modules = {"damping", "environment"} if context else {"damping"}
        scope = "damping_stage_b_multi" if context else "damping_stage_b_single"
    if (set(training.get("trainable_modules", [])) != expected_modules
            or set(saved.get("trainability", {}).get("trainable_modules", [])) != expected_modules
            or training.get("required_data_scope") != scope
            or training.get("required_renderer") != "smooth_v1"
            or training.get("train_frames") != 81 or saved.get("train_frames") != 81):
        raise ValueError("Checkpoint trainability/data scope does not match the stage")
    if saved.get("source_hashes") != sources or saved.get("dataset_sha256") != dataset_hash:
        raise ValueError("Frozen source/data hashes differ from the checkpoint")
    if not saved.get("state_dict") or any(not torch.isfinite(t).all() for t in saved["state_dict"].values()):
        raise ValueError("Checkpoint contains missing/nonfinite world-model tensors")
    feature_semantics(config)
    return scope


def read_records(handle, split, minimum_length, damping_values):
    """Read train/val metadata only; never open a held-out test partition."""
    if split not in ("train", "val"):
        raise ValueError("Staged evaluation never reads test")
    records = sorted([dict(episode=int(ep), group_id=int(handle["group_id"][ep]),
                           energy_band=int(handle["energy_band"][ep]), damping=float(handle["damping"][ep]))
                      for ep in handle["splits/" + split][:]], key=lambda r: (r["damping"], r["group_id"]))
    unique = {r["group_id"] for r in records}
    pairs = {(r["group_id"], r["damping"]) for r in records}
    expected = {(g, b) for g in unique for b in damping_values}
    if (len(unique) != EXPECTED_COUNTS[split] or pairs != expected or len(pairs) != len(records)
            or len({r["episode"] for r in records}) != len(records)):
        raise ValueError(f"Requires {EXPECTED_COUNTS[split]} unique {split} groups, each at every selected damping exactly once")
    if any(int(handle["ep_len"][r["episode"]]) < minimum_length for r in records):
        raise ValueError(f"{split} cannot cover the fixed history/horizon")
    for group in unique:
        if len({r["energy_band"] for r in records if r["group_id"] == group}) != 1:
            raise ValueError("Paired conditions disagree on initial energy strata")
    return records


def inspect_data(path, scope):
    with h5py.File(path, "r") as handle:
        values = tuple(json.loads(handle.attrs.get("selected_damping_json", "[]")))
        if (scope == "undamped_stage_a" and values != (0.,)
                or scope == "damping_stage_b_multi" and set(values) != set(SEEN_DAMPING)
                or scope == "damping_stage_b_single" and
                (len(values) != 1 or values[0] <= 0 or values[0] not in SEEN_DAMPING)
                or len(set(values)) != len(values)):
            raise ValueError("Selected damping metadata does not match the stage scope")
        values = tuple(sorted(values))
        if (handle.attrs.get("training_scope") != scope or handle.attrs.get("renderer") != "smooth_v1"
                or "requires_environment_context" not in handle.attrs
                or bool(handle.attrs["requires_environment_context"]) != (len(values) > 1)
                or handle.attrs.get("history_required") != 21 or handle.attrs.get("dt") != .1
                or handle.attrs.get("train_frames") != 81):
            raise ValueError("Data scope/renderer/context/history/dt does not match the checkpoint")
        # b is audited as metadata; it never enters model calls or readout queries.
        records = {split: read_records(handle, split, 81 if split == "train" else 53, values)
                   for split in ("train", "val")}
        physical = {key: float(handle.attrs[key]) for key in ("mass", "length", "gravity")}
        if not all(np.isfinite(value) and value > 0 for value in physical.values()):
            raise ValueError("Invalid physical evaluation constants")
        parent_hash = str(handle.attrs.get("parent_config_sha256", ""))
        if len(parent_hash) != 64 or any(c not in "0123456789abcdef" for c in parent_hash):
            raise ValueError("Data lacks a valid parent configuration hash")
    groups = {split: sorted({r["group_id"] for r in rows}) for split, rows in records.items()}
    if set(groups["train"]) & set(groups["val"]):
        raise ValueError("Training and validation groups overlap")
    return records, physical, groups, values, parent_hash


def split_training_records(records, fit_count=288, seed=5705):
    ordered = sorted(records, key=lambda row: row["group_id"])
    if len({row["group_id"] for row in ordered}) != len(ordered) or not 0 < fit_count < len(ordered):
        raise ValueError("Readout fit/selection must use disjoint nonempty training groups")
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    return ([ordered[i] for i in permutation[:fit_count]], [ordered[i] for i in permutation[fit_count:]])


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

    @classmethod
    def load(cls, path):
        """Rebuild an index from frozen arrays, without fitting/scaling/selection."""
        obj = cls.__new__(cls)
        with np.load(path, allow_pickle=False) as arrays:
            obj.x, obj.y = arrays["fit_features"].copy(), arrays["fit_targets"].copy()
            obj.x_mean, obj.x_scale = arrays["x_mean"].copy(), arrays["x_scale"].copy()
            obj.y_scale, obj.k = arrays["y_scale"].copy(), int(arrays["selected_k"])
            obj.fit_groups, obj.selection_groups = arrays["fit_group_ids"].tolist(), arrays["selection_group_ids"].tolist()
        if (obj.x.ndim != 2 or obj.x.shape[1] != 4 or obj.y.shape != (len(obj.x), 3)
                or len(obj.x) != 288 * 61 or obj.x_mean.shape != (4,) or obj.x_scale.shape != (4,)
                or obj.y_scale.shape != (3,) or obj.k not in K_VALUES
                or not all(np.isfinite(a).all() for a in (obj.x, obj.y, obj.x_mean, obj.x_scale, obj.y_scale))
                or not (obj.x_scale > 0).all() or not (obj.y_scale > 0).all()
                or len(obj.fit_groups) != 288 or len(set(obj.fit_groups)) != 288
                or len(obj.selection_groups) != 96 or len(set(obj.selection_groups)) != 96
                or set(obj.fit_groups) & set(obj.selection_groups)):
            raise ValueError("Malformed frozen A readout artifact")
        obj.tree = cKDTree((obj.x - obj.x_mean) / obj.x_scale)
        obj.candidates = []  # No B selection scores are computed.
        return obj


def verify_readout_provenance(path, saved, groups, physical, parent_hash):
    provenance_path = Path(path).with_name("readout_provenance.json")
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("format_version") != 1 or provenance.get("stage") != "a"
            or provenance.get("checkpoint_step") != 20000
            or provenance.get("readout_sha256") != sha256(path)
            or provenance.get("source_hashes") != saved.get("source_hashes")
            or provenance.get("shared_module_sha256") != shared_hashes(saved["state_dict"])
            or provenance.get("feature_semantics") != feature_semantics(saved["config"])
            or provenance.get("group_ids") != groups or provenance.get("physical_parameters") != physical
            or provenance.get("parent_config_sha256") != parent_hash
            or provenance.get("renderer") != "smooth_v1"):
        raise ValueError("A readout provenance/frozen mechanics/paired data mismatch")
    readout = KNNReadout.load(path)
    expected_fit, expected_selection = split_training_records([dict(group_id=g) for g in groups["train"]])
    if (readout.fit_groups != [r["group_id"] for r in expected_fit]
            or readout.selection_groups != [r["group_id"] for r in expected_selection]
            or provenance.get("calibration", {}).get("selected_k") != readout.k):
        raise ValueError("Frozen A readout group partition or selected k mismatch")
    return readout, provenance, provenance_path


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
        return dict(initial_energy_relative_error_median=None, max_relative_drift_median=None, max_relative_drift_p90=None,
                    future_energy_relative_mae_median=None, energy_change_relative_mae_median=None)
    pe, te = physical_energy(pred, physical), physical_energy(truth, physical)
    floor = 1e-6 * physical["mass"] * physical["gravity"] * physical["length"]
    scale = np.maximum(abs(te[:, 0]), floor)
    drift = np.max(abs(pe[:, 1:] - pe[:, :1]), axis=1) / scale
    return dict(initial_energy_relative_error_median=float(np.median(abs(pe[:, 0] - te[:, 0]) / scale)),
                max_relative_drift_median=float(np.median(drift)), max_relative_drift_p90=float(np.quantile(drift, .9)),
                future_energy_relative_mae_median=float(np.median(np.mean(abs(pe[:, 1:] - te[:, 1:]), axis=1) / scale)),
                energy_change_relative_mae_median=float(np.median(np.mean(abs(
                    (pe[:, 1:] - pe[:, :1]) - (te[:, 1:] - te[:, :1])), axis=1) / scale)),
                normalization="True energy at anchor, floor1e-6*m*g*L; drift from decoded anchor energy",
                interpretation="Physical energy drift is expected with damping; compare true energy/change error, not a conservation gate")


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
    if all("damping" in r for r in records):
        masks.update({f"damping_{b:g}": np.array([r["damping"] == b for r in records])
                      for b in sorted({r["damping"] for r in records})})
    baselines = {"decoded_" + key: value for key, value in baseline_predictions(pred[:, 0], horizon).items()}
    baselines.update({"oracle_" + key: value for key, value in baseline_predictions(truth[:, 0], horizon).items()})
    groups = {}
    for name, mask in masks.items():
        eligible = mask & complete
        pp, tt = pred[eligible, :horizon + 1], truth[eligible, :horizon + 1]
        metrics = physical_metrics(pp[:, 1:], tt[:, 1:], omega_scale)
        metrics.update(motion_metrics(pp[:, 1:], tt[:, 1:], omega_scale))
        observed_group = physical_metrics(teacher[mask, 1:horizon + 1], truth[mask, 1:horizon + 1], omega_scale)
        groups[name] = dict(n_total=int(mask.sum()), n_records=int(mask.sum()),
                            n_unique_groups=len({r.get("group_id", i) for i, r in enumerate(records) if mask[i]}),
                            n_complete=int(eligible.sum()), teacher_metrics=observed_group,
                            teacher_quality_gate_passed=bool(observed_group["angle_mae_rad"] <= .1 and observed_group["omega_nrmse"] <= .15),
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



def animation_artifact(directory, selected, rows, images, rollout, pred, teacher, checkpoint_hash,
                       step, readout, summary, damping=None):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {key: np.array([rows[i][key] for i in selected], dtype=np.int64)
                for key in ('episode', 'group_id', 'energy_band')}
    metadata['damping'] = np.array([rows[i]['damping'] for i in selected])
    np.savez_compressed(directory / 'animation_inputs.npz', true_rgb=images,
        truth=rollout['truth'][selected], teacher_prediction=teacher[selected],
        autonomous_prediction=pred[selected], valid_steps=rollout['valid_steps'][selected],
        time=np.arange(33) * .1, **metadata)
    gate = summary['teacher_quality_gate_passed']
    write_json(directory / 'animation_meta.json', dict(checkpoint_step=step, checkpoint_sha256=checkpoint_hash,
        selected_readout='uniform_knn', selected_k=readout.k, teacher_readout_quality=summary['teacher_metrics'],
        readout_quality_passed=gate,
        interpretation=('Frozen physical readout diagnostic; not a proof of latent dynamics correctness.' if gate else
                        'Readout quality failed: diagnostic only, physical prediction interpretation inconclusive.'),
        damping=damping, selected_records=[dict(rows[i], evaluation_index=int(i),
                                               valid_steps=int(rollout['valid_steps'][i])) for i in selected]))


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError('Use a new evaluation output directory')
    if args.batch_size != 32 or args.threads < 1:
        raise ValueError('Protocol fixes batch32 and requires positive CPU threads')
    if (args.stage == 'b') != (args.readout is not None):
        raise ValueError('--readout is required only for B; A always fits a new exact20k readout')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.threads)
    evaluation_hashes = evaluation_sources(args.code_root, args.code_manifest)
    Model, calibration, autonomous, pixels_tensor, source_hashes, diagnostic_names = load_frozen_helpers(args.code_root)
    sources, data_hash = source_hashes(), sha256(args.data)
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=True)
    scope = verify_checkpoint(saved, sources, data_hash, args.stage, args.expected_step)
    records, physical, groups, damping_values, parent_hash = inspect_data(args.data, scope)
    shared = shared_hashes(saved['state_dict'])
    if args.stage == 'b':
        readout, provenance, source_provenance = verify_readout_provenance(
            args.readout, saved, groups, physical, parent_hash)
    model = Model(saved['config']).to(args.device).eval()
    model.load_state_dict(saved['state_dict'], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = state_hash(model)
    args.output.mkdir(parents=True)

    def phase(name):
        print(json.dumps(dict(phase=name, elapsed_seconds=time.perf_counter() - started)), flush=True)

    if args.stage == 'a':
        fit, selection = split_training_records(records['train'])
        fit_pairs = [(r['episode'], t) for r in fit for t in range(20, 81)]
        selection_pairs = [(r['episode'], t) for r in selection for t in range(20, 81)]
        phase('fit_train288_readout_features')
        x, y = calibration(model, args.data, fit_pairs, 32)
        phase('select_k_on_train96_only')
        sx, sy = calibration(model, args.data, selection_pairs, 32)
        readout = KNNReadout(x, y, sx, sy)
        readout.save(args.output / 'readout.npz', fit, selection)
        calibration_report = dict(fit_groups=288, selection_groups=96, frames_per_group=61, seed=5705,
            fit_pairs_sha256=hashlib.sha256(np.asarray(fit_pairs, dtype=np.int64).tobytes()).hexdigest(),
            selection_pairs_sha256=hashlib.sha256(np.asarray(selection_pairs, dtype=np.int64).tobytes()).hexdigest(),
            kind='K0 causal observed phase state; same frozen readout for teacher and autonomous states',
            standardization='Fit288 only; per-feature and target std floor1e-6',
            selected_k=readout.k, k_candidates=readout.candidates, weights='uniform',
            distance='Euclidean in fit-standardized phase features')
        provenance = dict(format_version=1, stage='a', checkpoint_step=20000,
            checkpoint_sha256=checkpoint_hash, dataset_sha256=data_hash, source_hashes=sources,
            evaluation_source_hashes=evaluation_hashes, evaluator_sha256=sha256(__file__),
            readout_sha256=sha256(args.output / 'readout.npz'), shared_module_sha256=shared,
            feature_semantics=feature_semantics(saved['config']), group_ids=groups,
            physical_parameters=physical, parent_config_sha256=parent_hash,
            renderer='smooth_v1', calibration=calibration_report)
        write_json(args.output / 'readout_provenance.json', provenance)
    else:
        phase('load_frozen_A20k_readout_no_refit')
        shutil.copyfile(args.readout, args.output / 'readout.npz')
        shutil.copyfile(source_provenance, args.output / 'readout_provenance.json')
        if (sha256(args.output / 'readout.npz') != provenance['readout_sha256']
                or sha256(args.output / 'readout_provenance.json') != sha256(source_provenance)):
            raise RuntimeError('Copied A readout/provenance bytes changed')
        calibration_report = provenance['calibration']
    readout_hash = sha256(args.output / 'readout.npz')
    omega_scale = float(readout.y_scale[2])
    rows = records['val']
    directory = args.output / 'val'
    directory.mkdir()
    phase('val_observed_history_readout')
    pairs = [(r['episode'], t) for r in rows for t in range(20, 53)]
    # Labels returned here are discarded; neither B fitting nor model training exists.
    teacher_features, _ = calibration(model, args.data, pairs, 32)
    teacher_features = teacher_features.reshape(len(rows), 33, 4)
    if not np.isfinite(teacher_features).all():
        raise FloatingPointError('Observed-history encoder produced nonfinite features')
    teacher, teacher_distance = readout.predict(teacher_features)
    phase('val_autonomous_K32')
    rollout = collect_rollouts(model, args.data, rows, 32, 32, args.device,
                               autonomous, pixels_tensor, diagnostic_names)
    pred, distance = readout.predict(rollout['features'])
    metadata = {key: np.array([r[key] for r in rows], dtype=np.int64) for key in ('episode', 'group_id', 'energy_band')}
    metadata['damping'] = np.array([r['damping'] for r in rows])
    summaries = {str(h): summarize_horizon(pred, teacher, rollout['truth'], rollout['valid_steps'], rows,
                                           h, omega_scale, physical) for h in (8, 32)}
    selected_by_b = {}
    for b in damping_values:
        selected_by_b[b] = np.array([min((i for i, r in enumerate(rows) if r['energy_band'] == band and r['damping'] == b),
                                         key=lambda i: rows[i]['group_id'])
                                    for band in sorted(set(metadata['energy_band']))])
    default_b = .2 if .2 in selected_by_b else damping_values[0]
    selected = selected_by_b[default_b]
    np.savez_compressed(directory / 'trajectories.npz', pred=pred, teacher=teacher,
        truth=rollout['truth'], features=rollout['features'], teacher_features=teacher_features,
        valid_steps=rollout['valid_steps'], latent_energy=rollout['latent_energy'],
        nearest_fit_distance=distance, teacher_nearest_fit_distance=teacher_distance,
        time=np.arange(33) * .1, selected_indices=selected, **metadata,
        **{'diagnostic_' + key: value for key, value in rollout['diagnostics'].items()})
    # Stable top-level artifact names consumed by the staged orchestrator.
    shutil.copyfile(directory / 'trajectories.npz', args.output / 'trajectories.npz')
    # Display-only future RGB is read only after autonomous prediction completed.
    for b, indices in selected_by_b.items():
        with h5py.File(args.data, 'r') as handle:
            images = np.stack([handle['pixels'][int(handle['ep_offset'][rows[i]['episode']]) + 20:
                                                int(handle['ep_offset'][rows[i]['episode']]) + 53] for i in indices])
        grouped = summaries['32']['groups'][f'damping_{b:g}']
        if len(damping_values) > 1:
            animation_artifact(directory / f'damping_{b:g}', indices, rows, images, rollout, pred, teacher,
                               checkpoint_hash, args.expected_step, readout, grouped, b)
        if b == default_b:
            animation_artifact(directory, indices, rows, images, rollout, pred, teacher,
                               checkpoint_hash, args.expected_step, readout, grouped, b)
    curves = []
    for step in range(1, 33):
        eligible = rollout['valid_steps'] >= step
        curves.append(dict(step=step, seconds=.1 * step, coverage=float(eligible.mean()),
                           **physical_metrics(pred[eligible, step], rollout['truth'][eligible, step], omega_scale)))
    result = dict(split='val', episode_count=len(rows), record_count=len(rows), unique_group_count=len(groups['val']),
        condition_count=len(damping_values), damping_values=list(damping_values), anchor_frame=20,
        horizon_steps=32, horizon_seconds=3.2, rollout=summaries, curves=curves, failures=rollout['failures'],
        solver_failure_trajectory_fraction=float(np.mean(rollout['valid_steps'] < 32)),
        selected_records=[dict(rows[i], evaluation_index=int(i)) for i in selected],
        selection=f'Minimum group_id in each energy band within each damping; default animation uses b={default_b:g}',
        aggregation='Equal record/time weights; each validation group contributes one record per selected damping. Conditions share initial groups, not independent samples.')
    write_json(directory / 'summary.json', result)
    after = state_hash(model)
    if after != before or shared_hashes(model.state_dict()) != shared:
        raise RuntimeError('Frozen world-model state changed')
    if sha256(args.output / 'readout.npz') != readout_hash:
        raise RuntimeError('Frozen readout artifact changed during evaluation')
    summary = dict(status='completed', stage=args.stage, training_scope=scope, checkpoint_step=args.expected_step,
        q_dim=2, phase_dim=4, environment_context_dim=saved['config']['environment_context_dim'],
        checkpoint_sha256=checkpoint_hash, dataset_sha256=data_hash, source_hashes=sources,
        evaluation_source_hashes=evaluation_hashes, script_sha256=sha256(__file__),
        code_manifest_sha256=sha256(args.code_manifest) if args.code_manifest else None,
        code_root=str(args.code_root.resolve()), device=args.device, torch=str(torch.__version__),
        model_hash_before=before, model_hash_after=after, parameters_unchanged=True,
        shared_module_sha256=shared, shared_frozen_matches_A=(True if args.stage == 'b' else None),
        world_model_optimizer_created=False, world_model_updates=0, test_split_accessed=False,
        future_observations_passed_to_predictor=False, physical_labels_passed_to_predictor=False,
        readout_sha256=readout_hash, readout_provenance_sha256=sha256(args.output / 'readout_provenance.json'),
        readout_source=str(args.readout.resolve()) if args.readout else None,
        readout_A_checkpoint_sha256=provenance['checkpoint_sha256'], readout_A_checkpoint_step=20000,
        readout_refit=(args.stage == 'a'), readout_artifact_unchanged=True,
        calibration=calibration_report, omega_nrmse_denominator=omega_scale,
        evaluation={'val': result}, physical_parameters=physical,
        metrics_scope='Future frames excluding anchor; equal record/time weights on complete support. Failures retain NaN suffix and reduce coverage. Baselines use identical support.',
        quality_policy='Report is saved regardless of physical/readout quality. Quality thresholds are diagnostic and never authorize or block the next stage.',
        limitations=['KNN always maps finite queries to training labels, even off-manifold; inspect nearest-fit distances and teacher quality.',
                     'The A readout is fixed for B: its physical accuracy is not guaranteed on damped/OOD states.',
                     'Teacher reconstruction re-encodes observed histories and is not autonomous prediction.',
                     'A single seed and paired conditions do not establish statistical significance.',
                     'No test/unseen partition is read or used for fitting, selection, or evaluation.'],
        elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / 'summary.json', summary)
    phase('completed')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, default=ROOT)
    parser.add_argument('--code-manifest', type=Path)
    for name in ('checkpoint', 'data', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=('a', 'b'), required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--readout', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
