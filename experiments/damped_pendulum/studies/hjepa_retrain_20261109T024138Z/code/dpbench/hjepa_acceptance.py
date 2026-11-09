"""Fail-closed, read-only engineering acceptance of H-JEPA validation artifacts.

These thresholds are predeclared project screens, not H-JEPA paper criteria or
evidence of physical validity. No optimizer, world model, or test data is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


THRESHOLDS = {
    "teacher_angle_mae_rad_max": .1,
    "teacher_omega_nrmse_max": .15,
    "complete_trajectory_coverage_required": 1.,
    "dynamic_angle_freeze_trajectory_fraction_max": .05,
    "freeze_transition_count": 10,
    "freeze_angle_tolerance_rad": 1e-12,
    "truth_active_omega_rms_fraction_of_fit_std": .1,
    "state_mse_must_be_strictly_below_constant_anchor": True,
}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _verify(directory):
    from scripts.run_comparison import verify_evaluation
    summary = _json(directory / "summary.json")
    verify_evaluation(directory, summary["checkpoint_sha256"], summary["dataset_sha256"],
                      summary["protocol_sha256"], summary["checkpoint_step"])
    provenance = _json(directory / "readout_provenance.json")
    protocol, calibration = provenance["evaluation_protocol"], provenance["calibration"]
    config = summary["model_config"]
    if (config.get("kind") != "hjepa" or config.get("rank") != 32
            or config.get("history") != 21 or config.get("dt") != .1
            or protocol.get("history") != 21 or protocol.get("dt") != .1
            or protocol.get("train_frames") != 81
            or protocol.get("fit_groups") != 288 or protocol.get("selection_groups") != 96
            or protocol.get("future_metrics_exclude_anchor") is not True):
        raise ValueError("Requires rank32 H-JEPA and the unchanged train-only K0/32-step protocol")
    if (calibration.get("fit_groups") != 288 or calibration.get("selection_groups") != 96
            or calibration.get("feature_dimension") != 64 or calibration.get("weights") != "uniform"
            or provenance.get("validation_used_for_fitting_or_selection") is not False
            or provenance.get("readout_reused_from_other_model") is not False
            or summary.get("parameters_unchanged") is not True
            or summary.get("world_model_updates") != 0
            or summary.get("test_split_accessed") is not False):
        raise ValueError("Readout calibration/frozen evaluation provenance is not qualified")
    fit_ids, selection_ids = calibration["fit_group_ids"], calibration["selection_group_ids"]
    if (len(set(fit_ids)) != 288 or len(fit_ids) != 288
            or len(set(selection_ids)) != 96 or len(selection_ids) != 96
            or set(fit_ids) & set(selection_ids)):
        raise ValueError("Readout fitting and selection groups must be disjoint")
    val_ids = provenance["group_ids"]["val"]
    if len(val_ids) != 64 or len(set(val_ids)) != 64 or set(val_ids) & (set(fit_ids) | set(selection_ids)):
        raise ValueError("Validation groups must be 64 distinct, held-out groups")
    conditions = protocol["training_conditions"]
    if (not conditions or sorted(set(conditions)) != conditions
            or not set(conditions) <= {0., .05, .1, .2, .5, 1.}
            or calibration.get("selected_damping") != conditions):
        raise ValueError("Damping conditions disagree with calibration")
    readout_hash = _sha(directory / "readout.npz")
    if (provenance.get("readout_sha256") != readout_hash
            or summary.get("readout_sha256") != readout_hash
            or summary.get("readout_provenance_sha256") != _sha(directory / "readout_provenance.json")):
        raise ValueError("Readout provenance hash disagrees with artifact identity")
    with np.load(directory / "readout.npz", allow_pickle=False) as readout:
        scale = readout["y_scale"]
        if scale.shape != (3,) or not np.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Readout must carry finite positive fit-only target scales")
        omega_scale = float(scale[2])
    if summary.get("omega_nrmse_denominator") != omega_scale:
        raise ValueError("Omega denominator is inconsistent with the fitted readout")
    return summary, provenance, omega_scale


def _variance(x):
    trajectory_mean = x.mean(1, keepdims=True)
    within = float(np.mean((x - trajectory_mean) ** 2))
    between = float(np.mean((trajectory_mean - x.mean((0, 1), keepdims=True)) ** 2))
    total = within + between
    return {"within_trajectory": within, "between_trajectory_means": between,
            "total": total, "within_fraction": within / total if total else None}


def _physical(pred, truth, omega_scale):
    angle = pred[..., 0] - truth[..., 0]
    omega_rmse = float(np.sqrt(np.mean((pred[..., 1] - truth[..., 1]) ** 2)))
    return {"angle_mae_rad": float(np.mean(abs(np.arctan2(np.sin(angle), np.cos(angle))))),
            "omega_rmse_rad_s": omega_rmse, "omega_nrmse": omega_rmse / omega_scale}


def condition_metrics(arrays, mask, omega_scale):
    """All teacher scores use future frames; only complete rollout metrics are conditional."""
    pred, teacher, truth = (arrays[key][mask] for key in ("pred", "teacher", "truth"))
    state = arrays["features"][mask, :, :32]
    observed = arrays["teacher_features"][mask, :, :32]
    complete = ((arrays["valid_steps"][mask] == 32) & np.isfinite(pred).all((1, 2))
                & np.isfinite(arrays["features"][mask]).all((1, 2)))
    teacher_metrics = _physical(teacher[:, 1:], truth[:, 1:], omega_scale)
    teacher_pass = (teacher_metrics["angle_mae_rad"] <= THRESHOLDS["teacher_angle_mae_rad_max"]
                    and teacher_metrics["omega_nrmse"] <= THRESHOLDS["teacher_omega_nrmse_max"])
    count = int(mask.sum())
    metrics = {"trajectories": count, "complete_trajectories": int(complete.sum()),
               "coverage": float(complete.mean()), "teacher": teacher_metrics,
               "observed_state_variance": _variance(observed),
               "metric_support": "Teacher: all records. Prediction: complete records only; any missing record fails coverage."}
    gates = {"teacher_readout": bool(teacher_pass), "complete_coverage": bool(complete.all()),
             "dynamic_angle_freeze": False, "better_than_constant_state": False}
    if complete.any():
        pp, tt = pred[complete], truth[complete]
        ss, oo = state[complete], observed[complete]
        mse = float(np.mean((ss[:, 1:] - oo[:, 1:]) ** 2))
        hold_mse = float(np.mean((oo[:, :1] - oo[:, 1:]) ** 2))
        step_norm = np.linalg.norm(np.diff(ss, axis=1), axis=-1)
        observed_step_norm = np.linalg.norm(np.diff(oo, axis=1), axis=-1)
        pred_median, observed_median = float(np.median(step_norm)), float(np.median(observed_step_norm))
        dtheta = np.diff(pp[..., 0], axis=1)
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
        static = abs(dtheta) <= THRESHOLDS["freeze_angle_tolerance_rad"]
        width = THRESHOLDS["freeze_transition_count"]
        frozen_windows = np.lib.stride_tricks.sliding_window_view(static, width, axis=1).all(-1)
        truth_omega_rms = np.sqrt(np.mean(np.lib.stride_tricks.sliding_window_view(
            tt[:, 1:, 1] ** 2, width, axis=1), axis=-1))
        active_windows = truth_omega_rms > THRESHOLDS["truth_active_omega_rms_fraction_of_fit_std"] * omega_scale
        frozen = (frozen_windows & active_windows).any(1)
        freeze_fraction = float(frozen.mean())
        omega_mid = .5 * (pp[:, 1:, 1] + pp[:, :-1, 1])
        metrics.update(autonomous=_physical(pp[:, 1:], tt[:, 1:], omega_scale),
            state={"future_mse": mse, "constant_anchor_future_mse": hold_mse,
                   "mse_over_constant_anchor": mse / hold_mse if hold_mse else None,
                   "predicted_step_norm_median": pred_median, "observed_step_norm_median": observed_median,
                   "predicted_over_observed_step_norm_median": pred_median / observed_median if observed_median else None,
                   "predicted_variance": _variance(ss), "observed_variance_on_same_support": _variance(oo)},
            physical_freeze={"dynamic_angle_freeze_trajectory_count": int(frozen.sum()),
                             "dynamic_angle_freeze_trajectory_fraction": freeze_fraction,
                             "exact_angle_repeat_transition_fraction": float(static.mean()),
                             "static_angle_but_nonzero_speed_transition_fraction": float(
                                 (static & (abs(omega_mid) > .1 * omega_scale)).mean()),
                             "decoded_omega_vs_angle_finite_difference_nrmse": float(
                                 np.sqrt(np.mean((dtheta / .1 - omega_mid) ** 2)) / omega_scale)})
        gates["dynamic_angle_freeze"] = freeze_fraction <= THRESHOLDS["dynamic_angle_freeze_trajectory_fraction_max"]
        gates["better_than_constant_state"] = mse < hold_mse
    else:
        metrics.update(autonomous=None, state=None, physical_freeze=None)
    return dict(metrics=metrics, gates=gates, qualified=bool(all(gates.values())))


def qualify_evaluation(directory):
    directory = Path(directory).resolve()
    summary, provenance, omega_scale = _verify(directory)
    with np.load(directory / "val/trajectories.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in
                  ("pred", "teacher", "truth", "features", "teacher_features", "valid_steps", "time", "damping", "group_id")}
    conditions = provenance["evaluation_protocol"]["training_conditions"]
    n = 64 * len(conditions)
    for key in ("pred", "teacher", "truth"):
        if arrays[key].shape != (n, 33, 2):
            raise ValueError(f"Invalid {key} shape")
    for key in ("features", "teacher_features"):
        if arrays[key].shape != (n, 33, 64):
            raise ValueError(f"Invalid {key} shape; expected [s32, delta32]")
    for key in ("valid_steps", "damping", "group_id"):
        if arrays[key].shape != (n,):
            raise ValueError(f"Invalid {key} metadata shape")
    if (not np.issubdtype(arrays["valid_steps"].dtype, np.integer)
            or np.any((arrays["valid_steps"] < -1) | (arrays["valid_steps"] > 32))
            or arrays["time"].shape != (33,)
            or not np.allclose(arrays["time"], np.arange(33) * .1, rtol=0, atol=1e-12)
            or not np.array_equal(np.unique(arrays["damping"]), conditions)):
        raise ValueError("Invalid coverage, times, or damping conditions")
    if not all(np.isfinite(arrays[key]).all() for key in ("teacher", "truth", "teacher_features")):
        raise ValueError("Observed-history diagnostics and truth must be finite")
    groups = set(provenance["group_ids"]["val"])
    by_damping = {}
    for damping in conditions:
        mask = arrays["damping"] == damping
        if int(mask.sum()) != 64 or set(arrays["group_id"][mask]) != groups:
            raise ValueError("Every damping condition must have each validation group once")
        by_damping[f"{damping:g}"] = condition_metrics(arrays, mask, omega_scale)
    qualified = all(row["qualified"] for row in by_damping.values())
    return {
        "schema_version": 1, "status": "qualified" if qualified else "not_qualified", "qualified": qualified,
        "threshold_scope": "Predeclared engineering screens; not paper criteria, a cross-method ranking, or proof of physical validity",
        "thresholds": dict(THRESHOLDS), "evaluation_directory": str(directory),
        "checkpoint_step": summary["checkpoint_step"], "checkpoint_sha256": summary["checkpoint_sha256"],
        "dataset_sha256": summary["dataset_sha256"], "protocol_sha256": summary["protocol_sha256"],
        "evaluation_complete_sha256": _sha(directory / "complete.json"),
        "acceptance_module_sha256": _sha(Path(__file__)),
        "omega_scale_source": "Hash-verified readout.npz y_scale[2], fitted on train288 groups only",
        "omega_nrmse_denominator": omega_scale, "state_feature_slice": [0, 32],
        "future_frames": [1, 32], "per_damping": by_damping,
        "failed_checks": [f"b={b}: {gate}" for b, row in by_damping.items()
                          for gate, passed in row["gates"].items() if not passed],
        "limitations": ["kNN angle plateaus do not by themselves prove the latent state is stationary.",
                        "Last32 delta features are excluded from state-motion diagnostics.",
                        "Passing is a continuation screen, not final scientific acceptance; inspect plots and full physical errors.",
                        "Validation informs this screen; held-out test remains sealed."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output and (args.output.exists() or args.output.resolve().is_relative_to(args.evaluation.resolve())):
        parser.error("Use a new report path outside the immutable evaluation directory")
    try:
        report = qualify_evaluation(args.evaluation)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"schema_version": 1, "status": "invalid_evaluation", "qualified": False,
                  "error_type": type(exc).__name__, "error": str(exc),
                  "evaluation_directory": str(args.evaluation.resolve())}
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(payload, end="")
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
