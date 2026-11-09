"""Read-only startup health/ETA for the fresh undamped A run; never trains.

Only training artifacts are opened. --output writes an optional audit JSON,
never a trainer artifact. --checkpoint optionally imports the frozen code to
compare the input port with an independent seed-0 CPU initialization; no
optimizer is constructed. Run with the environment used by the frozen code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


HISTORICAL = {
    "source": "47727 L40/batch32/3-substeps, first 4000 history rows; old damped model",
    "source_file": "studies/stall_diagnosis_20261105/frozen_training/history.jsonl",
    "k4_median_seconds": 0.5224075242877007,
    "k8_median_seconds": 1.0793442912399769,
    "validation_median_seconds": 45.53769016265869,
    "checkpoint_median_seconds": 0.14940977096557617,
    "limits": "Old damped dynamics and 18 validation strata; A changes losses, D and batch organization. Prior only.",
}


def finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def read_history(path):
    """Ignore only an unfinished final line; never hide malformed closed rows."""
    path = Path(path)
    if not path.exists():
        return [], {"exists": False, "partial_tail_ignored": False, "parse_errors": []}
    raw = path.read_bytes()
    partial = bool(raw) and not raw.endswith(b"\n")
    lines = raw.splitlines()
    if partial:
        lines = lines[:-1]
    rows, errors = [], []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
            rows.append(row)
        except (ValueError, UnicodeDecodeError) as exc:
            errors.append({"line": number, "error": str(exc)})
    return rows, {"exists": True, "partial_tail_ignored": partial, "parse_errors": errors,
                  "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}


def summarize_history(rows, parse):
    steps = [row.get("step") for row in rows]
    contiguous = steps == list(range(1, len(rows) + 1)) and not parse["parse_errors"]
    bad_finite, bad_zero, missing = [], [], []
    ratios, splits = [], []
    validations = []
    for row in rows:
        step = row.get("step")
        losses = row.get("losses")
        if not isinstance(losses, dict) or not losses:
            missing.append({"step": step, "field": "losses"})
            losses = {}
        for key, value in list(losses.items()) + [("gradient_norm", row.get("gradient_norm"))]:
            if not finite_number(value):
                bad_finite.append({"step": step, "field": key, "value": value})
        for label, metrics in [("train", losses)] + [
                ("validation/" + key, value) for key, value in
                row.get("validation", {}).get("groups", {}).items()]:
            for key in ("dissipation", "work", "solver_residual_ratio", "solver_split_depth"):
                value = metrics.get(key)
                if not finite_number(value):
                    missing.append({"step": step, "field": label + "/" + key})
                elif key in ("dissipation", "work") and value != 0.0:
                    bad_zero.append({"step": step, "field": label + "/" + key, "value": value})
                elif key == "solver_residual_ratio":
                    ratios.append(value)
                elif key == "solver_split_depth":
                    splits.append(value)
        val = row.get("validation")
        if isinstance(val, dict):
            if not finite_number(val.get("score")):
                bad_finite.append({"step": step, "field": "validation/score", "value": val.get("score")})
            for group_name, group in val.get("groups", {}).items():
                for key, value in group.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                        bad_finite.append({"step": step, "field": "validation/" + group_name + "/" + key, "value": value})
            validations.append({"step": step, "score": val.get("score"),
                                "gate_passed": val.get("gate_passed"),
                                "position_score": val.get("position_score"),
                                "motion_score": val.get("motion_score"),
                                "validation_seconds": row.get("validation_seconds"),
                                "failed_prediction_groups": [name for name, group in val.get("groups", {}).items()
                                                             if group.get("prediction_gate_passed") is False]})
    residual_ok = bool(ratios) and all(0 <= x <= 1.0 for x in ratios)
    healthy = bool(rows) and contiguous and not bad_finite and not bad_zero and not missing and residual_ok
    return {
        "rows": len(rows), "latest_step": steps[-1] if steps else 0,
        "contiguous_from_1": contiguous, "finite_loss_and_gradient": not bad_finite,
        "nonfinite_count": len(bad_finite), "nonfinite_examples": bad_finite[:10],
        "logged_work_and_dissipation_exactly_zero": not bad_zero and not missing and bool(rows),
        "nonzero_work_or_dissipation_count": len(bad_zero), "nonzero_examples": bad_zero[:10],
        "missing_or_invalid_diagnostics_count": len(missing), "missing_examples": missing[:10],
        "maximum_solver_residual_ratio": max(ratios) if ratios else None,
        "solver_residual_ratios_within_tolerance": residual_ok,
        "maximum_split_depth": max(splits) if splits else None,
        "nonzero_split_diagnostic_count": sum(x > 0 for x in splits),
        "latest_losses": rows[-1].get("losses") if rows else None,
        "latest_gradient_norm": rows[-1].get("gradient_norm") if rows else None,
        "validation_rounds": len(validations), "latest_validation": validations[-1] if validations else None,
        "recent_validations": validations[-4:],
        "numerical_log_health_passed": healthy,
        "scope": "Logged aggregated work/dissipation only, not a fresh per-sample physical audit. "
                 "Prediction gate failures are reported separately from numerical/log health.",
    }


def timing_estimate(rows, run, now=None, target=4000):
    now = now or datetime.now(timezone.utc)
    m = max((row["step"] for row in rows if isinstance(row.get("step"), int)), default=0)
    recent = {}
    for horizon in (4, 8):
        candidates = [row["step_seconds"] for row in rows
                      if row.get("horizon_steps") == horizon and row.get("step", 0) > (20 if horizon == 4 else 1020)
                      and finite_number(row.get("step_seconds")) and row["step_seconds"] > 0][-200:]
        recent[horizon] = {"samples": len(candidates),
                           "median_seconds": statistics.median(candidates) if candidates else None,
                           "minimum_seconds": min(candidates) if candidates else None,
                           "maximum_seconds": max(candidates) if candidates else None}
    s4 = recent[4]["median_seconds"] or HISTORICAL["k4_median_seconds"]
    source4 = "current_K4_recent_median" if recent[4]["samples"] else "historical_K4_prior"
    if recent[8]["samples"]:
        s8, source8 = recent[8]["median_seconds"], "current_K8_recent_median"
    elif recent[4]["samples"]:
        s8 = s4 * HISTORICAL["k8_median_seconds"] / HISTORICAL["k4_median_seconds"]
        source8 = "EXTRAPOLATED_current_K4_times_historical_K8_over_K4"
    else:
        s8, source8 = HISTORICAL["k8_median_seconds"], "historical_K8_prior"
    completed_validations = [row for row in rows if isinstance(row.get("validation"), dict) and row.get("step", 0) <= target]
    val_times = [row["validation_seconds"] for row in completed_validations
                 if finite_number(row.get("validation_seconds")) and row["validation_seconds"] > 0][-4:]
    ckpt_times = [row["checkpoint_seconds"] for row in completed_validations
                  if finite_number(row.get("checkpoint_seconds")) and row["checkpoint_seconds"] > 0][-4:]
    v = statistics.median(val_times) if val_times else HISTORICAL["validation_median_seconds"]
    c = statistics.median(ckpt_times) if ckpt_times else HISTORICAL["checkpoint_median_seconds"]
    counts = {"K4": max(0, min(target, 1000) - m),
              "K8": max(0, target - max(m, 1000)),
              "validation": max(0, target // 250 - len(completed_validations))}
    remaining = counts["K4"] * s4 + counts["K8"] * s8 + counts["validation"] * (v + c)
    cfg = run.get("train_config", {}).get("training", {})
    schedule_ok = (target == 4000 and cfg.get("val_every") == 250
                   and cfg.get("curriculum", [])[:2] == [
                       {"until_step": 1000, "horizon": 4}, {"until_step": 4000, "horizon": 8}])
    return {"target_step": target, "latest_completed_step": m, "schedule_matches_estimator": schedule_ok,
            "recent_step_timings": recent, "K4_seconds_used": s4, "K4_timing_source": source4,
            "K8_seconds_used": s8, "K8_timing_source": source8,
            "validation_seconds_used": v, "validation_timing_source": "current_recent_median" if val_times else "historical_prior",
            "checkpoint_seconds_used": c, "remaining_counts": counts,
            "remaining_seconds": remaining if schedule_ok else None,
            "estimated_completion_utc": (now + timedelta(seconds=remaining)).isoformat() if schedule_ok else None,
            "current_validation_measured": bool(val_times),
            "confidence": "current_both_stages_and_validation" if recent[8]["samples"] and val_times else "provisional_extrapolation_or_prior",
            "historical_prior": HISTORICAL,
            "limits": "Excludes queue, data generation and startup outside trainer. Assumes uninterrupted execution, "
                      "unchanged throughput and successful solves. No uncertainty confidence interval. "
                      "step_seconds excludes validation; do not add validation again to iteration_seconds."}


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_audit(path, run, code_root):
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    step = checkpoint.get("step")
    state = checkpoint.get("state_dict", {})
    parameter_finite = bool(state) and all(torch.isfinite(value).all().item() for value in state.values())
    adam = checkpoint.get("optimizer", {})
    optimizer_steps = []
    optimizer_finite = True
    for values in adam.get("state", {}).values():
        for value in values.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all().item():
                optimizer_finite = False
        value = values.get("step")
        optimizer_steps.append(float(value.item()) if torch.is_tensor(value) else value)
    step_match = bool(optimizer_steps) and all(value == step for value in optimizer_steps)
    sources = checkpoint.get("source_hashes", {})
    source_rows = {name: (Path(code_root, name).is_file() and _file_sha(Path(code_root, name)) == digest)
                   for name, digest in sources.items()}
    source_match = bool(source_rows) and all(source_rows.values())
    result = {"path": str(Path(path).resolve()), "step": step,
              "validation_status": checkpoint.get("validation_status"),
              "sampler_next_step": checkpoint.get("sampler_next_step"),
              "sampler_matches_step": checkpoint.get("sampler_next_step") == step,
              "parameter_tensors": len(state), "parameters_finite": parameter_finite,
              "optimizer_state_tensors_with_steps": len(optimizer_steps),
              "optimizer_step_values": sorted(set(optimizer_steps), key=str),
              "optimizer_steps_match_checkpoint": step_match, "optimizer_tensors_finite": optimizer_finite,
              "config_matches_run": checkpoint.get("config") == run.get("config"),
              "dataset_hash_matches_run": checkpoint.get("dataset_sha256") == run.get("dataset_sha256"),
              "frozen_code_hashes": source_rows, "frozen_code_hashes_match": source_match,
              "input_port_seed0_comparison": {"performed": False},
              "optimizer_created": False, "model_updates": 0}
    if source_match and checkpoint.get("seed") == 0:
        sys.path.insert(0, str(Path(code_root).resolve()))
        from dpbench.dissipative_model import DissipativeWorldModel
        from dpbench.train_dissipative import configure_trainable_modules, build_optimizer_parameter_groups

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            initial = DissipativeWorldModel(checkpoint["config"])
        initial_state = initial.state_dict()
        names = [name for name in initial_state if name.startswith("core.input_net.")]
        matches = {name: name in state and torch.equal(initial_state[name], state[name]) for name in names}
        trainability = configure_trainable_modules(initial, checkpoint["train_config"]["training"])
        parameter_names = {id(value): name for name, value in initial.named_parameters()}
        expected_groups = build_optimizer_parameter_groups(initial, checkpoint["train_config"]["training"])
        expected_names = [parameter_names[id(value)] for group in expected_groups for value in group["params"]]
        optimizer_count = sum(len(group["params"]) for group in adam.get("param_groups", []))
        input_excluded = not any(name.startswith("core.input_net.") for name in expected_names)
        result["input_port_seed0_comparison"] = {
            "performed": True, "tensor_equal": matches, "all_equal": bool(matches) and all(matches.values()),
            "seed": 0, "device": "cpu", "input_port_frozen": not trainability["modules"]["input_port"]["trainable"],
            "input_port_excluded_from_expected_optimizer": input_excluded,
            "expected_optimizer_parameter_tensors": len(expected_names), "saved_optimizer_parameter_tensors": optimizer_count,
            "optimizer_parameter_count_matches": len(expected_names) == optimizer_count}
    comparison = result["input_port_seed0_comparison"]
    result["passed"] = all([parameter_finite, optimizer_finite, step_match, result["sampler_matches_step"],
                            result["config_matches_run"], result["dataset_hash_matches_run"], source_match,
                            comparison.get("all_equal", False), comparison.get("input_port_frozen", False),
                            comparison.get("input_port_excluded_from_expected_optimizer", False),
                            comparison.get("optimizer_parameter_count_matches", False)])
    return result


def inspect(run_root, code_root=None, checkpoint=None):
    run_root = Path(run_root)
    training = run_root / "training" if (run_root / "training").is_dir() else run_root
    read = lambda name: json.loads((training / name).read_text()) if (training / name).exists() else {}
    run, status = read("run.json"), read("status.json")
    rows, parse = read_history(training / "history.jsonl")
    summary = summarize_history(rows, parse)
    now = datetime.now(timezone.utc)
    config = run.get("config", {})
    checks = {"damping_disabled": config.get("core", {}).get("damping_mode") == "none",
              "environment_disabled": config.get("environment_context_dim") == 0,
              "data_scope_bound": run.get("preflight", {}).get("training_scope") == "undamped_stage_a",
              "trainable_modules_bound": run.get("trainability", {}).get("requested_modules") == ["encoder", "initializer", "potential"],
              "fresh_seed0": run.get("resume") is None and run.get("train_config", {}).get("training", {}).get("seed") == 0,
              "bound_4000": run.get("stop_after_step") == 4000}
    result = {"checked_at_utc": now.isoformat(), "run_root": str(run_root.resolve()),
              "training_directory": str(training.resolve()), "operation": "read_only_training_artifact_audit",
              "test_data_read": False, "optimizer_created": False, "training_updates": 0,
              "history_parse": parse, "history": summary, "status_file": status,
              "protocol_checks": checks, "protocol_matches_stage_a": bool(run) and all(checks.values()),
              "eta": timing_estimate(rows, run, now),
              "overall_numerical_health_passed": bool(run) and all(checks.values())
                  and summary["numerical_log_health_passed"] and status.get("status") != "failed"}
    if checkpoint is not None:
        path = Path(checkpoint)
        path = path if path.is_absolute() else training / path
        if path.exists():
            if code_root is None:
                raise ValueError("--checkpoint requires --code-root for independent frozen-code initialization")
            result["checkpoint"] = checkpoint_audit(path, run, code_root)
        else:
            result["checkpoint"] = {"path": str(path), "exists": False, "passed": False}
        result["overall_numerical_health_passed"] &= result["checkpoint"]["passed"]
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NONFINITE:" + str(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--checkpoint", nargs="?", const="last.pt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = json_safe(inspect(args.run_root, args.code_root, args.checkpoint))
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        protected = {"history.jsonl", "run.json", "status.json", "last.pt", "best.pt"}
        if args.output.name in protected or args.output.suffix != ".json":
            raise ValueError("--output must be a separate audit .json, not a training artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")
    return result


if __name__ == "__main__":
    main()
