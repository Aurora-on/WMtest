#!/usr/bin/env python3
"""Compare frozen step10000 teacher-observed linear and nonlinear readouts.

World-model parameters are never updated. Readouts fit on 288 train groups,
hyperparameters are chosen on 96 other train groups, and val is evaluation only.
This is not autonomous prediction and does not access the test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree
import torch

from evaluate_control import (
    choose_ridge, load_frozen_helpers, physical_metrics, sha256,
    split_training_records, state_hash, verify_evaluation_sources, write_json,
)

K_VALUES = (1, 5, 25, 50)


def check_checkpoint(saved, arm, hashes, data_hash):
    objective = {"control": "current", "lehami": "lehami"}[arm]
    c = saved.get("config", {})
    if (saved.get("kind") != "dissipative" or saved.get("step") != 10000
            or saved.get("validation_status") != "completed"):
        raise ValueError(f"{arm}: requires completed step10000 checkpoint")
    if (c.get("objective_mode") != objective or c.get("core", {}).get("damping_mode") != "none"
            or c.get("environment_context_dim") != 0):
        raise ValueError(f"{arm}: objective or undamped-model identity differs")
    if saved.get("source_hashes") != hashes or saved.get("dataset_sha256") != data_hash:
        raise ValueError(f"{arm}: frozen sources/data do not match checkpoint")


def targets_to_physics(y):
    y = np.asarray(y)
    return np.stack((np.arctan2(y[..., 0], y[..., 1]), y[..., 2]), axis=-1)


def ridge_targets(readout, x):
    standardized = (x - readout["x_mean"]) / readout["x_scale"]
    return (np.column_stack((standardized, np.ones(len(x)))) @ readout["weights"]
            * readout["y_scale"] + readout["y_mean"])


def knn_predictions(x, y, selection_x, selection_y, evaluation_x):
    """Uniform Euclidean kNN; feature scaling and target scales use fit only."""
    arrays = [np.asarray(a, dtype=np.float64) for a in (x, y, selection_x, selection_y, evaluation_x)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite readout input")
    x, y, selection_x, selection_y, evaluation_x = arrays
    xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
    ys = np.maximum(y.std(0), 1e-6)
    tree = cKDTree((x - xm) / xs)
    max_k = max(K_VALUES)
    if len(x) < max_k:
        raise ValueError("Insufficient fitting points for fixed kNN candidates")
    _, selection_ids = tree.query((selection_x - xm) / xs, k=max_k, workers=1)
    candidates = []
    for k in K_VALUES:
        predicted = y[selection_ids[:, :k]].mean(axis=1)
        score = float(np.mean(((predicted - selection_y) / ys) ** 2))
        candidates.append(dict(k=k, selection_normalized_mse=score))
    chosen = min(candidates, key=lambda item: item["selection_normalized_mse"])
    _, evaluation_ids = tree.query((evaluation_x - xm) / xs, k=chosen["k"], workers=1)
    if chosen["k"] == 1:
        evaluation_ids = evaluation_ids[:, None]
    return y[evaluation_ids].mean(axis=1), chosen["k"], candidates


def representation_summary(x, y):
    centered = x - x.mean(0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    std = x.std(0)
    normalized = centered / np.maximum(std, 1e-6)
    standardized_targets = (y - y.mean(0)) / np.maximum(y.std(0), 1e-6)
    return dict(feature_std=std.tolist(), centered_singular_values=singular_values.tolist(),
                numeric_rank=int(np.linalg.matrix_rank(centered)),
                standardized_feature_target_cross_moment=(normalized.T @ standardized_targets / len(x)).tolist(),
                interpretation="Small variance alone does not establish lost information; readout evaluation tests recoverability.")


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    if args.batch_size != 32:
        raise ValueError("Use batch32, matching the existing calibration protocol")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(args.threads)
    evaluation_hashes = verify_evaluation_sources(args.code_root, args.code_manifest)
    Model, calibration, _, _, source_hashes, _ = load_frozen_helpers(args.code_root)
    hashes, data_hash = source_hashes(), sha256(args.data)
    with h5py.File(args.data, "r") as f:
        if f.attrs.get("training_scope") != "undamped_stage_a":
            raise ValueError("Requires undamped stage-A data")
        # Do not inspect splits/test or any of its observations/labels.
        records = {split: sorted([dict(episode=int(ep), group_id=int(f["group_id"][ep]))
                                  for ep in f["splits/" + split][:]], key=lambda r: r["group_id"])
                   for split in ("train", "val")}
    if len(records["train"]) != 384 or len(records["val"]) != 64:
        raise ValueError("Requires 384 train and 64 val groups")
    fit, select = split_training_records(records["train"])
    groups = [set(r["group_id"] for r in rows) for rows in (fit, select, records["val"])]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("Fit, selection and validation groups overlap")
    fit_pairs = [(r["episode"], t) for r in fit for t in range(20, 81)]
    select_pairs = [(r["episode"], t) for r in select for t in range(20, 81)]
    val_pairs = [(r["episode"], t) for r in records["val"] for t in range(20, 53)]
    args.output.mkdir(parents=True)
    results = {}
    for arm, checkpoint in (("control", args.control_checkpoint), ("lehami", args.lehami_checkpoint)):
        def phase(name):
            print(json.dumps(dict(arm=arm, phase=name, elapsed_seconds=time.perf_counter() - started)), flush=True)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        check_checkpoint(saved, arm, hashes, data_hash)
        model = Model(saved["config"]).to(args.device).eval()
        model.load_state_dict(saved["state_dict"], strict=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if model.history != 21 or model.dt != .1:
            raise ValueError("Requires history21 and dt=.1")
        before = state_hash(model)
        phase("fit_feature_extraction")
        x, y = calibration(model, args.data, fit_pairs, args.batch_size)
        phase("selection_feature_extraction")
        sx, sy = calibration(model, args.data, select_pairs, args.batch_size)
        phase("val_teacher_feature_extraction")
        vx, vy = calibration(model, args.data, val_pairs, args.batch_size)
        ridge, alpha, ridge_candidates = choose_ridge(x, y, sx, sy)
        ridge_y = ridge_targets(ridge, vx)
        knn_y, k, knn_candidates = knn_predictions(x, y, sx, sy, vx)
        constant_y = np.broadcast_to(y.mean(0), vy.shape).copy()
        truth = targets_to_physics(vy).reshape(64, 33, 2)
        omega_scale = float(ridge["y_scale"][2])
        metrics = {}
        for method, prediction in (("ridge", ridge_y), ("knn", knn_y), ("constant_train_mean", constant_y)):
            physical = targets_to_physics(prediction).reshape(64, 33, 2)
            metrics[method] = physical_metrics(physical[:, 1:], truth[:, 1:], omega_scale)
            normalized_error = ((prediction - vy) / ridge["y_scale"]) ** 2
            metrics[method]["normalized_target_mse"] = float(normalized_error.reshape(64, 33, 3)[:, 1:].mean())
            metrics[method]["quality_gate_passed"] = (metrics[method]["angle_mae_rad"] <= .1
                                                       and metrics[method]["omega_nrmse"] <= .15)
        after = state_hash(model)
        if before != after:
            raise RuntimeError("Frozen model parameters changed")
        np.savez_compressed(args.output / f"{arm}_features.npz", fit_features=x, fit_targets=y,
                            selection_features=sx, selection_targets=sy, val_features=vx, val_targets=vy,
                            ridge_prediction_targets=ridge_y, knn_prediction_targets=knn_y,
                            constant_prediction_targets=constant_y, fit_group_ids=[r["group_id"] for r in fit],
                            selection_group_ids=[r["group_id"] for r in select], val_group_ids=[r["group_id"] for r in records["val"]],
                            val_episodes=[r["episode"] for r in records["val"]],
                            fit_pairs=fit_pairs, selection_pairs=select_pairs, val_pairs=val_pairs,
                            **{"ridge_" + key: value for key, value in ridge.items()})
        result = dict(checkpoint_step=10000, checkpoint_sha256=sha256(checkpoint),
                      objective_mode=saved["config"]["objective_mode"], metrics=metrics,
                      ridge=dict(selected_alpha=alpha, candidates=ridge_candidates),
                      knn=dict(selected_k=k, candidates=knn_candidates, weights="uniform", distance="Euclidean in fit-standardized phase features"),
                      representation=representation_summary(x, y), model_hash_before=before,
                      model_hash_after=after, parameters_unchanged=True, world_model_updates=0,
                      omega_nrmse_denominator=omega_scale)
        results[arm] = result
        write_json(args.output / f"{arm}_summary.json", result)
        phase("completed")
        del model, saved
        if args.device == "cuda":
            torch.cuda.empty_cache()
    summary = dict(status="completed", arms=results, dataset_sha256=data_hash, source_hashes=hashes,
                   evaluation_source_hashes=evaluation_hashes, script_sha256=sha256(__file__),
                   helper_script_sha256=sha256(Path(__file__).with_name("evaluate_control.py")),
                   code_manifest_sha256=sha256(args.code_manifest), code_root=str(args.code_root.resolve()),
                   device=args.device, torch=str(torch.__version__),
                   calibration=dict(fit_groups=288, selection_groups=96, fit_frames_per_group=61,
                                    selection_frames_per_group=61, seed=5705, scaler_fit="fit groups only, floor1e-6"),
                   evaluation=dict(val_groups=64, frames_per_group=33, scored_future_frames=32,
                                   teacher_observed=True, autonomous_rollout=False, test_split_accessed=False),
                   world_model_optimizer_created=False, world_model_updates=0,
                   protocol="Hyperparameters selected only on held-out train groups; val scores exclude anchor. kNN is a nonlinear teacher-feature probe, not the author's knn_k1 one-step-output protocol.",
                   elapsed_seconds=time.perf_counter() - started)
    write_json(args.output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "code-manifest", "data", "control-checkpoint", "lehami-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    return run(parser.parse_args())


if __name__ == "__main__":
    main()
