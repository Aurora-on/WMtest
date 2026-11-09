"""Synthetic read-only checks; no optimizers, data generation or training."""
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stage_a_inspector", HERE / "inspect_startup.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(step, damping=0.0):
    value = {"step": step, "horizon_steps": 4 if step <= 1000 else 8,
             "step_seconds": .5 if step <= 1000 else 1., "gradient_norm": 2.0,
             "losses": {"loss": .5, "dissipation": damping, "work": 0.,
                        "solver_residual_ratio": .3, "solver_split_depth": 0.},
             "validation_seconds": 0., "checkpoint_seconds": 0.}
    if step % 250 == 0:
        value.update(validation={"score": .5, "gate_passed": False, "groups": {}},
                     validation_seconds=20., checkpoint_seconds=.1)
    return value


def run():
    return {"config": {"core": {"damping_mode": "none"}, "environment_context_dim": 0},
            "train_config": {"training": {"seed": 0, "val_every": 250, "curriculum": [
                {"until_step": 1000, "horizon": 4}, {"until_step": 4000, "horizon": 8}]}},
            "preflight": {"training_scope": "undamped_stage_a"}, "resume": None, "stop_after_step": 4000,
            "trainability": {"requested_modules": ["encoder", "initializer", "potential"]}}


def test_partial_tail_ignored_but_closed_bad_row_detected(tmp_path):
    p = tmp_path / "history.jsonl"
    p.write_text(json.dumps(row(1)) + '\n{"step":2')
    rows, parse = audit.read_history(p)
    assert len(rows) == 1 and parse["partial_tail_ignored"]
    assert audit.summarize_history(rows, parse)["numerical_log_health_passed"]
    p.write_text(json.dumps(row(1)) + '\nNOT_JSON\n' + json.dumps(row(3)) + '\n')
    rows, parse = audit.read_history(p)
    result = audit.summarize_history(rows, parse)
    assert parse["parse_errors"] and not result["contiguous_from_1"]
    assert not result["numerical_log_health_passed"]


def test_damped_nonfinite_and_excess_residual_never_pass():
    parse = {"parse_errors": []}
    assert not audit.summarize_history([row(1, damping=.2)], parse)["numerical_log_health_passed"]
    bad = row(1)
    bad["gradient_norm"] = float("nan")
    assert not audit.summarize_history([bad], parse)["finite_loss_and_gradient"]
    bad = row(1)
    bad["losses"]["solver_residual_ratio"] = 1.1
    assert not audit.summarize_history([bad], parse)["numerical_log_health_passed"]
    json.dumps(audit.json_safe(audit.summarize_history([bad], parse)), allow_nan=False)


def test_prediction_gate_failure_is_quality_not_numerical_failure():
    result = audit.summarize_history([row(i) for i in range(1, 251)], {"parse_errors": []})
    assert result["numerical_log_health_passed"]
    assert result["latest_validation"]["gate_passed"] is False
    assert result["validation_rounds"] == 1


def test_eta_counts_validation_once_and_marks_k8_extrapolation():
    now = datetime(2026, 11, 5, tzinfo=timezone.utc)
    result = audit.timing_estimate([row(i) for i in range(1, 251)], run(), now)
    assert result["remaining_counts"] == {"K4": 750, "K8": 3000, "validation": 15}
    assert result["K8_timing_source"].startswith("EXTRAPOLATED")
    expected = 750 * .5 + 3000 * result["K8_seconds_used"] + 15 * 20.1
    assert result["remaining_seconds"] == pytest.approx(expected)
    end = datetime.fromisoformat(result["estimated_completion_utc"])
    assert (end - now).total_seconds() == pytest.approx(expected, abs=1e-6)


def test_eta_replaces_prior_after_k8_and_stops_at_4000():
    result = audit.timing_estimate([row(i) for i in range(1, 1101)], run())
    assert result["K8_timing_source"] == "current_K8_recent_median"
    assert result["K8_seconds_used"] == 1.
    result = audit.timing_estimate([row(i) for i in range(1, 4001)], run())
    assert result["remaining_counts"] == {"K4": 0, "K8": 0, "validation": 0}
    assert result["remaining_seconds"] == 0.
    changed = run()
    changed["train_config"]["training"]["val_every"] = 100
    assert audit.timing_estimate([row(1)], changed)["remaining_seconds"] is None


def test_old_damped_protocol_not_labeled_stage_a(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    legacy = run()
    legacy["config"]["core"]["damping_mode"] = "position_velocity"
    (training / "run.json").write_text(json.dumps(legacy))
    (training / "status.json").write_text(json.dumps({"status": "running"}))
    (training / "history.jsonl").write_text(json.dumps(row(1, damping=.2)) + '\n')
    result = audit.inspect(tmp_path)
    assert not result["overall_numerical_health_passed"]
    assert not result["protocol_checks"]["damping_disabled"]
    assert result["history"]["nonzero_work_or_dissipation_count"] == 1


def test_checkpoint_seed0_frozen_port_and_synthetic_adam_metadata(tmp_path):
    import torch
    import platform
    from dpbench.dissipative_model import DissipativeWorldModel
    from dpbench.train_dissipative import configure_trainable_modules, build_optimizer_parameter_groups, source_hashes

    code_root = HERE.parents[1]
    cfg = json.loads((code_root / "configs/dissipative_undamped_stage_a_v1.json").read_text())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = DissipativeWorldModel(cfg["model"])
    configure_trainable_modules(model, cfg["training"])
    groups = build_optimizer_parameter_groups(model, cfg["training"])
    # Only checkpoint metadata is fabricated. No optimizer object or update.
    indices = []
    cursor = 0
    for group in groups:
        ids = list(range(cursor, cursor + len(group["params"])))
        cursor += len(ids)
        indices.append({"params": ids})
    checkpoint = {"step": 1, "sampler_next_step": 1, "validation_status": "not_run", "seed": 0,
                  "config": model.config, "train_config": cfg, "dataset_sha256": "synthetic",
                  "source_hashes": source_hashes(), "state_dict": model.state_dict(),
                  "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "cuda_build": torch.version.cuda, "system": platform.system(),
                              "platform": platform.platform(), "machine": platform.machine()},
                  "optimizer": {"param_groups": indices, "state": {i: {"step": torch.tensor(1.)} for i in range(cursor)}}}
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    meta = {"config": model.config, "dataset_sha256": "synthetic"}
    result = audit.checkpoint_audit(path, meta, code_root)
    assert result["passed"] and result["input_port_seed0_comparison"]["all_equal"]
    assert result["input_port_seed0_comparison"]["input_port_frozen"]
    assert not result["optimizer_created"]
    saved = torch.load(path, weights_only=True)
    key = next(k for k in saved["state_dict"] if k.startswith("core.input_net."))
    saved["state_dict"][key].flatten()[0] += 1.
    torch.save(saved, path)
    result = audit.checkpoint_audit(path, meta, code_root)
    assert not result["passed"] and not result["input_port_seed0_comparison"]["all_equal"]
    saved["runtime"]["torch"] = "different-build"
    torch.save(saved, path)
    result = audit.checkpoint_audit(path, meta, code_root)
    assert not result["passed"] and result["verification_status"] == "inconclusive"
    assert not result["input_port_seed0_comparison"]["performed"]
    assert result["input_port_seed0_comparison"]["all_equal"] is None
    assert "No conclusion" in result["input_port_seed0_comparison"]["interpretation"]


def test_missing_os_is_explicit_and_reference_os_must_match():
    import platform
    import torch

    recorded = {"python": platform.python_version(), "torch": str(torch.__version__),
                "cuda_build": torch.version.cuda}
    assert not audit.initialization_runtime_check(recorded)["compatible_for_exact_initialization"]
    same = audit.initialization_runtime_check(recorded, platform.system())
    assert same["compatible_for_exact_initialization"]
    assert same["reference_system_source"] == "explicit_reference_system_from_training_job_record"
    different = "Linux" if platform.system() != "Linux" else "Darwin"
    assert not audit.initialization_runtime_check(recorded, different)["compatible_for_exact_initialization"]
