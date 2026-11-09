#!/usr/bin/env python3
"""Read-only paired-run startup audit; only --output is written.

Run in the server's training Python environment inside a short Slurm CPU job.
This script never constructs a model/optimizer, submits work, or updates a run.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def numeric_finite(value):
    if isinstance(value, dict):
        return all(numeric_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(numeric_finite(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    return True


def read_history(path):
    raw = Path(path).read_bytes()
    partial = bool(raw) and not raw.endswith(b"\n")
    lines = raw.splitlines()
    if partial:
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines if line.strip()]
    return [row for row in rows if "step" in row and "losses" in row], partial


def history_audit(rows, partial):
    steps = [row["step"] for row in rows]
    required = ("work", "dissipation", "solver_residual_ratio", "phase_prediction_mse")
    complete = bool(rows) and all(all(k in row.get("losses", {}) for k in required)
                                  and all(k in row for k in ("gradient_norm", "lr_factor", "step_seconds"))
                                  for row in rows)
    finite = bool(rows) and all(numeric_finite(row) for row in rows)
    zero = complete and all(row["losses"][k] == 0 for row in rows for k in ("work", "dissipation"))
    residuals = [row["losses"]["solver_residual_ratio"] for row in rows
                 if "solver_residual_ratio" in row.get("losses", {})]
    sequential = bool(steps) and steps == list(range(1, steps[-1] + 1))
    progressing = len(rows) >= 2
    positive_lr = complete and all(row["lr_factor"] > 0 for row in rows)
    validations = [row for row in rows if "validation" in row]
    latest = validations[-1] if validations else None
    return dict(first_step=steps[0] if rows else None, latest_logged_step=steps[-1] if rows else None,
                logged_steps=len(rows), complete_required_metrics=complete, all_numbers_finite=finite,
                consecutive_fresh_steps=sequential, multiple_steps_logged=progressing,
                all_learning_rate_factors_positive=positive_lr,
                work_and_dissipation_exactly_zero=zero,
                max_solver_residual_ratio=max(residuals) if residuals else None,
                residuals_within_tolerance=bool(residuals) and all(0 <= x <= 1 + 1e-10 for x in residuals),
                ignored_partial_final_line=partial,
                latest_validation_step=latest["step"] if latest else None,
                latest_validation_gate_passed=latest["validation"].get("gate_passed") if latest else None,
                latest_validation_score=latest["validation"].get("score") if latest else None,
                passed=(complete and finite and zero and sequential and progressing and positive_lr
                        and bool(residuals) and all(0 <= x <= 1 + 1e-10 for x in residuals)),
                scope="Logged aggregate numerical health, not independent physical validation; early quality gates do not determine startup success")


def estimate_remaining(rows, qualification, stop=10000, interval=250):
    step = rows[-1]["step"] if rows else 0
    rates = {}
    for horizon in (4, 8, 16):
        sample = [row["step_seconds"] for row in rows
                  if row.get("horizon_steps") == horizon and row["step"] > 1
                  and isinstance(row.get("step_seconds"), (float, int))
                  and math.isfinite(row["step_seconds"]) and row["step_seconds"] > 0][-100:]
        rates[horizon] = statistics.median(sample) if sample else None
    checks = qualification.get("checks", {}).get("rows", [])
    qualified = {}
    for row in checks:
        forward, backward = row.get("forward_seconds"), row.get("backward_seconds")
        if (row.get("horizon") in (4, 8, 16) and isinstance(forward, (float, int))
                and isinstance(backward, (float, int)) and forward + backward > 0):
            qualified[row["horizon"]] = forward + backward
    ratio = qualified[8] / qualified[4] if 4 in qualified and 8 in qualified else None
    k8_source = "measured_training_steps" if rates[8] is not None else "unavailable"
    if rates[8] is None and rates[4] is not None:
        # K4 qualification is the cold first CUDA call whereas K8 is warm.
        # Their ratio is contaminated by startup and must not drive ETA.
        rates[8] = rates[4] * 2
        k8_source = "unmeasured_rough_assumption_2_times_measured_K4"
    k16_source = "measured_training_steps" if rates[16] is not None else "unavailable"
    if rates[16] is None and rates[8] is not None:
        rates[16] = 2 * rates[8]
        k16_source = ("unmeasured_rough_assumption_2_times_measured_K8"
                      if k8_source == "measured_training_steps" else
                      "unmeasured_rough_assumption_4_times_measured_K4")
    left4 = max(0, min(stop, 1000) - step)
    left8 = max(0, min(stop, 4000) - max(step, 1000))
    left16 = max(0, min(stop, 10000) - max(step, 4000))
    training = None
    if all(count == 0 or rates[k] is not None for count, k in ((left4, 4), (left8, 8), (left16, 16))):
        training = left4 * (rates[4] or 0) + left8 * (rates[8] or 0) + left16 * (rates[16] or 0)
    validation = [row["validation_seconds"] for row in rows
                  if "validation" in row and row.get("validation_seconds", 0) > 0]
    median_validation = statistics.median(validation[-4:]) if validation else None
    last_validated = max((row["step"] for row in rows if "validation" in row), default=0)
    remaining_validations = max(0, stop // interval - last_validated // interval)
    total = (training + remaining_validations * median_validation
             if training is not None and median_validation is not None else None)
    return dict(latest_logged_step=step, cumulative_target_step=stop, recent_K4_step_seconds=rates[4],
                K8_step_seconds_estimate=rates[8], K8_rate_source=k8_source,
                K16_step_seconds_estimate=rates[16], K16_rate_source=k16_source,
                qualification_K8_K4_ratio=ratio, remaining_K4_steps=left4, remaining_K8_steps=left8,
                qualification_timings_used_for_eta=False,
                qualification_forward_backward_seconds=qualified,
                remaining_K16_steps=left16,
                training_seconds_remaining_estimate=training,
                validation_seconds_measured_median=median_validation,
                validation_timing_available=median_validation is not None,
                remaining_validation_events=remaining_validations,
                total_seconds_remaining_estimate=total,
                interpretation=("Rough projection, not a deadline. Qualification K4 is cold and K8 is warm; "
                                "their ratio is recorded but NOT used. Unmeasured K8 assumes2*K4 and K16 assumes4*K4 "
                                "(or2*measuredK8). Without measured validation, training-only duration is incomplete. "
                                "Continuation queue/startup, checkpoint/I/O overhead and contention can add time."))


def checkpoint_audit(path, run, qualification, history):
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", {})
    optimizer = checkpoint.get("optimizer", {})
    adam_state = optimizer.get("state", {})
    step = checkpoint.get("step")
    def finite_tensors(value):
        if torch.is_tensor(value):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite_tensors(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite_tensors(v) for v in value)
        return numeric_finite(value)
    adam_steps = [float(row["step"]) for row in adam_state.values() if "step" in row]
    hash_now = state_hash(state)
    initial_hash = qualification.get("checks", {}).get("initial_state_sha256")
    latest = history.get("latest_logged_step") or 0
    checks = dict(nonempty_state=bool(state), parameters_finite=bool(state) and finite_tensors(state),
                  nonempty_adam_state=bool(adam_state), optimizer_tensors_finite=finite_tensors(optimizer),
                  adam_steps_match_checkpoint=(bool(adam_steps) and len(adam_steps) == len(adam_state)
                                               and all(x == step for x in adam_steps)),
                  sampler_matches_checkpoint=checkpoint.get("sampler_next_step") == step,
                  config_matches_run=checkpoint.get("config") == run.get("config"),
                  train_config_matches_run=checkpoint.get("train_config") == run.get("train_config"),
                  dataset_matches_run=checkpoint.get("dataset_sha256") == run.get("dataset_sha256"),
                  sources_match_run=checkpoint.get("source_hashes") == run.get("source_hashes"),
                  validation_indices_match_run=checkpoint.get("validation_indices") == run.get("validation_indices"),
                  parameters_changed_since_initial=bool(initial_hash) and hash_now != initial_hash,
                  checkpoint_progress_consistent=type(step) is int and 1 <= step <= latest + 1)
    return dict(path=str(path), step=step, validation_status=checkpoint.get("validation_status"),
                state_sha256=hash_now, qualification_initial_sha256=initial_hash,
                adam_state_count=len(adam_state), adam_step_values=sorted(set(adam_steps)),
                checks=checks, passed=all(checks.values()), model_created=False, optimizer_created=False,
                comparison_scope="Hash change proves state changed, not independent verification of which parameters changed")


def inspect_arm(root, arm, target_step=10000):
    directory = root / "runs" / arm
    read = lambda path: json.loads(path.read_text())
    run, status = read(directory / "run.json"), read(directory / "status.json")
    qualification = read(root / f"{arm}_cuda_qualification.json")
    rows, partial = read_history(directory / "history.jsonl")
    history = history_audit(rows, partial)
    checkpoint = checkpoint_audit(directory / "last.pt", run, qualification, history)
    train = run.get("train_config", {}).get("training", {})
    model = run.get("config", {})
    allowed_status = status.get("status") in ("running", "budget_completed", "completed")
    if status.get("status") == "budget_completed":
        allowed_status = status.get("completed_steps") == 4000 and not status.get("full_budget_completed", True)
    if status.get("status") == "completed":
        allowed_status = False  # A 20k completed run is outside this 4k launch.
    run_checks = dict(fresh=run.get("resume") is None and train.get("seed") == 0,
                      configured_full_schedule=train.get("steps") == 20000,
                      stop_after_4000=run.get("stop_after_step") == 4000,
                      objective_correct=model.get("objective_mode") == ("current" if arm == "control" else "lehami"),
                      three_substeps=model.get("core", {}).get("substeps") == 3,
                      hard_zero_damping=model.get("core", {}).get("damping_mode") == "none",
                      target_only_tail=model.get("target_tail_frames") == 2,
                      training_cuda=run.get("device") == "cuda",
                      status_not_failed=allowed_status,
                      qualification_completed=qualification.get("status") == "completed",
                      qualification_no_updates=qualification.get("optimizer_updates") == 0,
                      qualification_sources_match=qualification.get("source_hashes") == run.get("source_hashes"),
                      qualification_data_match=qualification.get("dataset_sha256") == run.get("dataset_sha256"))
    return dict(run_checks=run_checks, passed=all(run_checks.values()) and history["passed"] and checkpoint["passed"],
                status=status, history=history, checkpoint=checkpoint,
                eta=estimate_remaining(rows, qualification, stop=target_step), allocation=qualification.get("allocation", {}),
                initial_state_sha256=qualification.get("checks", {}).get("initial_state_sha256"),
                dataset_sha256=run.get("dataset_sha256"), source_hashes=run.get("source_hashes"),
                validation_indices=run.get("validation_indices"),
                qualification_train_indices=qualification.get("fixed_train_window_indices"),
                last_loss=rows[-1].get("losses") if rows else None)


def audit(root, target_step=10000):
    root = Path(root).resolve()
    report = dict(root=str(root), checked_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  scope="Read-only checkpoint/log startup health, not convergence or physical accuracy",
                  optimizer_created=False, optimizer_updates=0, cumulative_target_step=target_step,
                  continuation_scope="This audit covers initial runs/{arm} to4000; jobs resuming to10000 must be checked separately", arms={})
    for arm in ("control", "lehami"):
        try:
            report["arms"][arm] = inspect_arm(root, arm, target_step)
        except Exception as error:
            report["arms"][arm] = dict(passed=False, inspection_error=type(error).__name__, detail=str(error))
    control, lehami = (report["arms"][arm] for arm in ("control", "lehami"))
    keys = ("initial_state_sha256", "dataset_sha256", "source_hashes", "validation_indices", "qualification_train_indices")
    pair = {f"same_{key}": bool(control.get(key)) and control.get(key) == lehami.get(key) for key in keys}
    uuids = [x.get("allocation", {}).get("device_uuid") for x in (control, lehami)]
    pair["distinct_physical_GPU_UUIDs"] = all(bool(x) for x in uuids) and uuids[0] != uuids[1]
    report.update(pair_checks=pair, passed=all(pair.values()) and all(x["passed"] for x in (control, lehami)),
                  GPU_UUIDs=uuids,
                  status_scope="Slurm liveness must be checked separately. status.json may lag history until the next validation/checkpoint.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-step", type=int, choices=(4000, 10000), default=10000)
    args = parser.parse_args()
    report = audit(args.root, args.target_step)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=report["passed"], output=str(args.output), pair_checks=report["pair_checks"])))


if __name__ == "__main__":
    main()
