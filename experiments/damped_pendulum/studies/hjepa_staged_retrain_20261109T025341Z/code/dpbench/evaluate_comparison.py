"""Shared K0 physical evaluation for the comparison-model adapters.

Every checkpoint fits its own diagnostic readout on training groups only.
The world model sees RGB and passive actions, never physical labels or damping.
This is the project's matched K0 probe, not an author's native K1 evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from dpbench.evaluate import pixels_tensor, targets
from scripts.evaluate_staged_damping import (
    KNNReadout, physical_metrics, read_records, sha256, state_hash,
    summarize_horizon, write_json,
)

HISTORY = 21
TRAIN_FRAMES = 81
DT = .1
FIT_GROUPS = 288
SELECTION_SEED = 5705
SEEN_DAMPING = (0., .05, .1, .2, .5, 1.)
ROOT = Path(__file__).resolve().parents[1]


class DissipativeEvaluationAdapter(torch.nn.Module):
    """Read-only adapter for our existing staged checkpoints and the same probe."""
    def __init__(self, config):
        super().__init__()
        from dpbench.dissipative_model import DissipativeWorldModel
        self.model = DissipativeWorldModel(config)
        self.provenance = {
            "kind": "ours_dissipative", "adapter": "legacy_initialize_and_autonomous_rollout",
            "feature_semantics": "K0 concatenated q,p from the model's causal initializer",
            "core": config.get("core", {}),
            "readout_policy": "New train-only K0 readout on this checkpoint's own training conditions; not the historical frozen A readout",
        }

    def observed_features(self, history, past_actions=None):
        if past_actions is None:
            past_actions = history.new_zeros((len(history), HISTORY - 1, 1))
        intervals = torch.full_like(past_actions, DT)
        q, p, _, _ = self.model.initialize(history, past_actions, intervals)
        return torch.cat((q, p), -1)

    def rollout_features(self, history, actions, past_actions):
        from dpbench.evaluate_passive import autonomous_rollout
        result = autonomous_rollout(self.model, history, past_actions, torch.full_like(past_actions, DT),
            actions, torch.full_like(actions, DT), diagnostic_names=())
        return torch.as_tensor(result["features"], device=history.device)


def build_comparison_model(config, device):
    """Lazy import keeps metric/data tests independent of optional baseline code."""
    from dpbench.comparison_models import build_comparison_model as factory
    return factory(config, device=device)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _hash_string(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def normalize_horizons(horizons):
    values = tuple(sorted({_positive_integer(h, "horizon") for h in horizons}))
    if not values:
        raise ValueError("At least one evaluation horizon is required")
    if max(values) > TRAIN_FRAMES - HISTORY:
        raise ValueError("81-frame data and a 21-frame history support at most 60 future steps (6 seconds); longer evaluation requires a separately audited dataset/protocol")
    return values


def inspect_data(path, horizon, split="val"):
    """Audit only train/val metadata; do not open a test split or its data."""
    if split != "val":
        raise ValueError("Comparison evaluation currently supports val only; test is sealed")
    with h5py.File(path, "r") as f:
        if (f.attrs.get("renderer") != "smooth_v1" or f.attrs.get("history_required") != HISTORY
                or f.attrs.get("train_frames") != TRAIN_FRAMES or not np.isclose(f.attrs.get("dt", np.nan), DT)):
            raise ValueError("Evaluation requires smooth_v1, history21, train_frames81 and dt0.1")
        values = tuple(sorted(float(v) for v in json.loads(f.attrs.get("selected_damping_json", "[]"))))
        if not values or len(set(values)) != len(values) or not set(values).issubset(SEEN_DAMPING):
            raise ValueError("Data must declare unique selected seen damping conditions")
        rows = {"train": read_records(f, "train", TRAIN_FRAMES, values),
                "val": read_records(f, "val", HISTORY + horizon, values)}
        physical = {key: float(f.attrs[key]) for key in ("mass", "length", "gravity")}
        if not all(np.isfinite(v) and v > 0 for v in physical.values()):
            raise ValueError("Physical evaluation constants must be positive and finite")
        parent_hash = _hash_string(str(f.attrs.get("parent_config_sha256", "")), "parent_config_sha256")
    groups = {name: sorted({r["group_id"] for r in records}) for name, records in rows.items()}
    if set(groups["train"]) & set(groups["val"]):
        raise ValueError("Train and validation initial-condition groups overlap")
    return rows, groups, physical, values, parent_hash


def partition_training_groups(records, seed=SELECTION_SEED):
    """Split initial conditions, keeping every condition of a group together."""
    by_group = {}
    for row in records:
        by_group.setdefault(row["group_id"], []).append(row)
    group_ids = sorted(by_group)
    if len(group_ids) != 384:
        raise ValueError("Readout protocol requires 384 unique training groups")
    permutation = np.random.default_rng(seed).permutation(len(group_ids))
    fit_ids = [group_ids[i] for i in permutation[:FIT_GROUPS]]
    selection_ids = [group_ids[i] for i in permutation[FIT_GROUPS:]]
    def expand(ids):
        return [row for gid in ids for row in sorted(by_group[gid], key=lambda r: r["damping"])]
    return expand(fit_ids), expand(selection_ids), fit_ids, selection_ids


def _pairs(records, last_anchor):
    return [(r["episode"], t) for r in records for t in range(HISTORY - 1, last_anchor + 1)]


def _check_actions(actions, intervals):
    if (not np.isfinite(actions).all() or not np.all(actions == 0)
            or not np.isfinite(intervals).all() or not np.allclose(intervals, DT, rtol=0, atol=1e-7)):
        raise ValueError("Comparison evaluation requires passive zero actions at dt0.1")


def _feature_array(value, expected_prefix):
    if not isinstance(value, torch.Tensor):
        raise TypeError("Comparison adapter must return a feature Tensor")
    if value.ndim != len(expected_prefix) + 1 or tuple(value.shape[:-1]) != tuple(expected_prefix) or value.shape[-1] < 1:
        raise ValueError(f"Unexpected comparison feature shape {tuple(value.shape)}; expected {expected_prefix} + (F,)")
    return value.detach().cpu().numpy().astype(np.float64, copy=False)


@torch.no_grad()
def collect_observed_features(model, path, pairs, batch_size=32, device="cpu"):
    """At anchor t read RGB[t-20:t+1], past actions[t-20:t], and target[t]."""
    if not pairs:
        raise ValueError("Observed feature collection needs at least one causal window")
    features, labels = [], []
    with h5py.File(path, "r") as f:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            images, past_actions, y = [], [], []
            for ep, anchor in batch:
                if anchor < HISTORY - 1 or anchor >= int(f["ep_len"][ep]):
                    raise ValueError("Observed feature anchor has insufficient causal history")
                offset = int(f["ep_offset"][ep])
                lo, hi = offset + anchor - HISTORY + 1, offset + anchor + 1
                images.append(pixels_tensor(f["pixels"][lo:hi], "cpu"))
                actions = f["action"][lo:hi - 1]
                _check_actions(actions, f["dt_frame"][lo:hi - 1])
                past_actions.append(actions)
                y.append(targets(f["state"][hi - 1:hi])[0])
            # Labels are collected outside the model call; no b or physical state enters it.
            out = model.observed_features(torch.stack(images).to(device),
                past_actions=torch.as_tensor(np.stack(past_actions), dtype=torch.float32, device=device))
            features.append(_feature_array(out, (len(batch),)))
            labels.extend(y)
    x, y = np.concatenate(features), np.asarray(labels, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError("Observed features or physical readout targets are nonfinite")
    return x, y


def finite_prefix(features):
    """Keep every finite prefix and leave failure suffixes NaN, never hold state."""
    features = np.asarray(features, dtype=np.float64).copy()
    if features.ndim != 3:
        raise ValueError("Rollout features must have shape records,time,features")
    finite = np.isfinite(features).all(-1)
    valid = np.full(len(features), features.shape[1] - 1, dtype=np.int64)
    failures = []
    for i in range(len(features)):
        invalid = np.flatnonzero(~finite[i])
        if len(invalid):
            first = int(invalid[0])
            valid[i] = first - 1
            features[i, first:] = np.nan
            failures.append({"evaluation_index": i, "failed_step": first, "error_type": "NonfiniteFeatures"})
    return features, valid, failures


@torch.no_grad()
def collect_autonomous_features(model, path, records, horizon, batch_size=32, device="cpu", feature_dim=None):
    """One initial history per trajectory; future RGB is never read here."""
    feature_chunks, truth, numeric_exceptions = [], [], {}
    def predict(images, action, indices):
        try:
            out = model.rollout_features(images, actions=action[:, HISTORY - 1:],
                                         past_actions=action[:, :HISTORY - 1])
            return _feature_array(out, (len(indices), horizon + 1))
        except FloatingPointError as exc:
            # Isolate a numerical failure without changing the algorithm, time step,
            # or input support. Structural/RuntimeError failures are never swallowed.
            if len(indices) > 1:
                cut = len(indices) // 2
                return np.concatenate((predict(images[:cut], action[:cut], indices[:cut]),
                                       predict(images[cut:], action[cut:], indices[cut:])))
            if feature_dim is None:
                raise
            numeric_exceptions[indices[0]] = {"error_type": type(exc).__name__, "error": str(exc)}
            return np.full((1, horizon + 1, feature_dim), np.nan)
    with h5py.File(path, "r") as f:
        for start in range(0, len(records), batch_size):
            rows = records[start:start + batch_size]
            images, actions = [], []
            for row in rows:
                offset = int(f["ep_offset"][row["episode"]])
                images.append(pixels_tensor(f["pixels"][offset:offset + HISTORY], "cpu"))
                action = f["action"][offset:offset + HISTORY - 1 + horizon]
                _check_actions(action, f["dt_frame"][offset:offset + HISTORY - 1 + horizon])
                actions.append(action)
                truth.append(f["state"][offset + HISTORY - 1:offset + HISTORY + horizon])
            action = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device)
            feature_chunks.append(predict(torch.stack(images).to(device), action,
                                          list(range(start, start + len(rows)))))
    features, valid, failures = finite_prefix(np.concatenate(feature_chunks))
    for failure in failures:
        failure["episode"] = records[failure["evaluation_index"]]["episode"]
        failure.update(numeric_exceptions.get(failure["evaluation_index"], {}))
    truth = np.asarray(truth, dtype=np.float64)
    if truth.shape != (len(records), horizon + 1, 2) or not np.isfinite(truth).all():
        raise ValueError("Physical evaluation truth has invalid shape or values")
    return {"features": features, "truth": truth, "valid_steps": valid, "failures": failures}


def _provenance(model):
    value = model.provenance
    value = value() if callable(value) else value
    # Require serializable explicit model provenance instead of silently stringifying it.
    json.dumps(value, allow_nan=False)
    return value


def _source_hashes():
    files = ["dpbench/evaluate_comparison.py", "scripts/evaluate_comparison.py",
             "scripts/evaluate_staged_damping.py", "dpbench/evaluate.py", "dpbench/comparison_models.py",
             "dpbench/evaluate_passive.py", "dpbench/dissipative_model.py", "dpbench/dissipative_core.py",
             "dpbench/efficient_dissipative_core.py", "dpbench/automatic_discrete_gradient.py"]
    return {name: sha256(ROOT / name) for name in files if (ROOT / name).is_file()}


def verify_checkpoint_sources(saved):
    """Reject changed comparison training code; label legacy compatibility explicitly."""
    if saved["kind"] == "dissipative":
        return {
            "status": "legacy_compatibility_mode", "strict_source_identity_verified": False,
            "reason": "Historical checkpoints use the older source schema; current adapter/evaluator hashes are recorded separately",
            "recorded_sources": saved.get("source_hashes", saved.get("source_sha256")),
        }
    recorded = saved.get("source_sha256")
    if recorded is None:
        return {"status": "checkpoint_manifest_not_recorded", "strict_source_identity_verified": False}
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("Comparison checkpoint source_sha256 manifest is malformed")
    from dpbench.comparison_protocol import source_manifest
    current = source_manifest()
    if current != recorded:
        changed = sorted(k for k in set(current) | set(recorded) if current.get(k) != recorded.get(k))
        raise ValueError(f"Comparison checkpoint source identity differs from current code: {changed}")
    return {"status": "matched_current_manifest", "strict_source_identity_verified": True,
            "source_sha256": current}


def evaluate_checkpoint(checkpoint_path, data_path, output, device="cpu", batch_size=32,
                        split="val", horizons=(8, 16, 32)):
    """Evaluate our comparison checkpoint without a model optimizer or test access.

    `weights_only=False` is intentionally limited to locally produced comparison
    or legacy dissipative checkpoints. Do not pass third-party pickle files.
    """
    started = time.perf_counter()
    checkpoint_path, data_path, output = map(Path, (checkpoint_path, data_path, output))
    batch_size = _positive_integer(batch_size, "batch_size")
    horizons = normalize_horizons(horizons)
    if output.exists():
        raise FileExistsError("Use a new evaluation output directory")
    if split != "val":
        raise ValueError("Comparison evaluation currently supports val only; test is sealed")
    rows, groups, physical, damping_values, parent_hash = inspect_data(data_path, max(horizons), split)
    data_hash = sha256(data_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    if (not isinstance(saved, dict) or saved.get("kind") not in ("comparison", "dissipative")
            or not saved.get("state_dict")):
        raise ValueError("Requires a locally produced comparison or dissipative checkpoint")
    step = _positive_integer(saved.get("step"), "checkpoint step")
    if saved.get("dataset_sha256") != data_hash:
        raise ValueError("Evaluation data SHA-256 differs from checkpoint training data")
    legacy = saved["kind"] == "dissipative"
    protocol_hash = (None if legacy and saved.get("protocol_sha256") is None
                     else _hash_string(saved.get("protocol_sha256"), "protocol_sha256"))
    if any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in saved["state_dict"].values()):
        raise ValueError("Checkpoint contains nonfinite or non-tensor model state")
    config = saved.get("config") if legacy else saved.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint lacks resolved model configuration")
    if config.get("history") != HISTORY or not np.isclose(config.get("dt", np.nan), DT):
        raise ValueError("Model history/dt differs from the shared evaluation protocol")
    source_verification = verify_checkpoint_sources(saved)
    model = DissipativeEvaluationAdapter(config) if legacy else build_comparison_model(config, device=device)
    (model.model if legacy else model).load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = state_hash(model)
    model_provenance = _provenance(model)
    output.mkdir(parents=True)
    evaluation_hashes = _source_hashes()
    evaluation_protocol = {
        "schema_version": 1, "history": HISTORY, "dt": DT, "train_frames": TRAIN_FRAMES,
        "horizons": list(horizons), "split": split, "fit_groups": FIT_GROUPS,
        "selection_groups": 96, "seed": SELECTION_SEED, "k_candidates": [1, 5, 25, 50],
        "readout": "own-model train-only K0 uniform kNN; fit-only standardized features/targets",
        "training_conditions": list(damping_values), "calibration_anchors": [20, 80],
        "future_metrics_exclude_anchor": True, "failure_policy": "finite prefix, NaN suffix, explicit coverage",
    }
    evaluation_protocol_hash = hashlib.sha256(json.dumps(evaluation_protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def phase(name):
        print(json.dumps({"phase": name, "elapsed_seconds": time.perf_counter() - started}), flush=True)

    fit, selection, fit_ids, selection_ids = partition_training_groups(rows["train"])
    fit_pairs, selection_pairs = _pairs(fit, TRAIN_FRAMES - 1), _pairs(selection, TRAIN_FRAMES - 1)
    phase("fit_own_readout_on_train288_groups")
    x, y = collect_observed_features(model, data_path, fit_pairs, batch_size, device)
    phase("select_k_on_train96_groups_only")
    sx, sy = collect_observed_features(model, data_path, selection_pairs, batch_size, device)
    readout = KNNReadout(x, y, sx, sy)
    readout.save(output / "readout.npz", [{"group_id": g} for g in fit_ids], [{"group_id": g} for g in selection_ids])
    readout_hash = sha256(output / "readout.npz")
    omega_scale = float(readout.y_scale[2])
    calibration_report = {
        "kind": "K0 causal observed features; each model owns its readout; same readout for teacher/autonomous",
        "feature_dimension": int(x.shape[1]), "fit_groups": len(fit_ids), "selection_groups": len(selection_ids),
        "fit_records": len(fit), "selection_records": len(selection), "frames_per_record": 61,
        "conditions_per_group": len(damping_values), "selected_damping": list(damping_values), "seed": SELECTION_SEED,
        "fit_pairs_sha256": hashlib.sha256(np.asarray(fit_pairs, dtype=np.int64).tobytes()).hexdigest(),
        "selection_pairs_sha256": hashlib.sha256(np.asarray(selection_pairs, dtype=np.int64).tobytes()).hexdigest(),
        "standardization": "Fit288 groups only; feature/target std floor1e-6", "selected_k": readout.k,
        "k_candidates": readout.candidates, "weights": "uniform", "distance": "Euclidean in standardized own-model features",
        "fit_group_ids": fit_ids, "selection_group_ids": selection_ids,
    }
    write_json(output / "readout_provenance.json", {
        "format_version": 1, "checkpoint_sha256": checkpoint_hash, "checkpoint_step": step,
        "dataset_sha256": data_hash, "protocol_sha256": protocol_hash, "parent_config_sha256": parent_hash,
        "evaluation_protocol": evaluation_protocol, "evaluation_protocol_sha256": evaluation_protocol_hash,
        "readout_sha256": readout_hash, "model_provenance": model_provenance,
        "checkpoint_source_verification": source_verification,
        "evaluation_source_hashes": evaluation_hashes, "calibration": calibration_report,
        "group_ids": groups, "physical_parameters": physical,
        "validation_used_for_fitting_or_selection": False, "readout_reused_from_other_model": False,
    })
    records, horizon = rows[split], max(horizons)
    # Autonomous call precedes all future observed-history diagnostics. Its function
    # reads only initial RGB; teacher state is never available to the predictor.
    phase(f"{split}_autonomous_K{horizon}")
    rollout = collect_autonomous_features(model, data_path, records, horizon, batch_size, device, feature_dim=x.shape[1])
    if rollout["features"].shape[-1] != x.shape[1]:
        raise ValueError("Autonomous and observed adapter feature dimensions differ")
    pred, distance = readout.predict(rollout["features"])
    phase(f"{split}_observed_history_diagnostic")
    teacher_features, _ = collect_observed_features(model, data_path,
        _pairs(records, HISTORY - 1 + horizon), batch_size, device)
    teacher_features = teacher_features.reshape(len(records), horizon + 1, x.shape[1])
    teacher, teacher_distance = readout.predict(teacher_features)
    summaries = {str(h): summarize_horizon(pred, teacher, rollout["truth"], rollout["valid_steps"],
                    records, h, omega_scale, physical) for h in horizons}
    metadata = {key: np.asarray([r[key] for r in records], dtype=np.int64)
                for key in ("episode", "group_id", "energy_band")}
    metadata["damping"] = np.asarray([r["damping"] for r in records], dtype=np.float64)
    directory = output / split
    directory.mkdir()
    np.savez_compressed(directory / "trajectories.npz", pred=pred, teacher=teacher,
        truth=rollout["truth"], features=rollout["features"], teacher_features=teacher_features,
        valid_steps=rollout["valid_steps"], time=np.arange(horizon + 1) * DT,
        nearest_fit_distance=distance, teacher_nearest_fit_distance=teacher_distance, **metadata)
    curves = []
    for t in range(1, horizon + 1):
        eligible = rollout["valid_steps"] >= t
        curves.append(dict(step=t, seconds=DT * t, coverage=float(eligible.mean()),
            **physical_metrics(pred[eligible, t], rollout["truth"][eligible, t], omega_scale)))
    result = dict(split=split, record_count=len(records), unique_group_count=len(groups[split]),
        condition_count=len(damping_values), damping_values=list(damping_values), anchor_frame=HISTORY - 1,
        horizon_steps=horizon, horizon_seconds=DT * horizon, rollout=summaries, curves=curves,
        failures=rollout["failures"], failed_trajectory_fraction=float(np.mean(rollout["valid_steps"] < horizon)),
        aggregation="Equal record/time weights on complete support; paired conditions share initial groups and are not independent samples")
    write_json(directory / "summary.json", result)
    after = state_hash(model)
    if before != after or sha256(output / "readout.npz") != readout_hash:
        raise RuntimeError("Frozen model or diagnostic readout changed during evaluation")
    summary = dict(status="completed", kind="comparison_evaluation", checkpoint_kind=saved["kind"], checkpoint_step=step,
        checkpoint_sha256=checkpoint_hash, dataset_sha256=data_hash, protocol_sha256=protocol_hash,
        evaluation_protocol=evaluation_protocol, evaluation_protocol_sha256=evaluation_protocol_hash,
        model_config=config, model_provenance=model_provenance,
        checkpoint_source_hashes=saved.get("source_sha256", saved.get("source_hashes")),
        checkpoint_source_verification=source_verification,
        model_hash_before=before, model_hash_after=after,
        parameters_unchanged=True, world_model_optimizer_created=False, world_model_updates=0,
        future_observations_passed_to_predictor=False, physical_labels_passed_to_predictor=False,
        damping_values_passed_to_predictor=False, test_split_accessed=False, readout_reused_from_other_model=False,
        readout_sha256=readout_hash, readout_provenance_sha256=sha256(output / "readout_provenance.json"),
        readout_artifact_unchanged=True, calibration=calibration_report, omega_nrmse_denominator=omega_scale,
        physical_parameters=physical, evaluation_source_hashes=evaluation_hashes, evaluation={split: result},
        device=str(device), torch=str(torch.__version__), elapsed_seconds=time.perf_counter() - started,
        metrics_scope="Future frames exclude anchor; incomplete trajectories reduce coverage and retain NaN suffix; matched-support trivial baselines",
        limitations=["Matched K0 readout differs from author-native evaluation protocols.",
            "Observed-history reconstruction is diagnostic, not autonomous prediction or a strict error lower bound.",
            "kNN maps finite off-manifold queries to training labels; inspect teacher quality and nearest-fit distances.",
            "Each model owns a train-only readout; no shared latent-space readout weights.",
            "No test/unseen data is used for fitting, selection or evaluation."])
    write_json(output / "summary.json", summary)
    files = ["readout.npz", "readout_provenance.json", "summary.json", f"{split}/summary.json", f"{split}/trajectories.npz"]
    write_json(output / "complete.json", dict(status="completed", checkpoint_sha256=checkpoint_hash,
        dataset_sha256=data_hash, protocol_sha256=protocol_hash,
        evaluation_protocol_sha256=evaluation_protocol_hash,
        artifacts={name: sha256(output / name) for name in files}))
    phase("completed")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Locally produced comparison or legacy dissipative checkpoint only")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--horizons", default=[8, 16, 32], nargs="+", type=int)
    args = parser.parse_args(argv)
    torch.set_num_threads(_positive_integer(args.threads, "threads"))
    return evaluate_checkpoint(args.checkpoint, args.data, args.output, device=args.device,
        batch_size=args.batch_size, split=args.split, horizons=args.horizons)


if __name__ == "__main__":
    main()
