import copy
import importlib.util
import json
from pathlib import Path
import torch

SCRIPT = Path(__file__).with_name("audit_pair_startup.py")
SPEC = importlib.util.spec_from_file_location("pair_startup_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def row(step=1, horizon=4):
    return dict(step=step, horizon_steps=horizon, step_seconds=.4, gradient_norm=2., lr_factor=.01,
                losses=dict(work=0., dissipation=0., solver_residual_ratio=.5, phase_prediction_mse=.2),
                validation_seconds=0.)


def test_history_ignores_only_incomplete_final_line_and_checks_numerics(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text(json.dumps(row(1))+"\n"+json.dumps(row(2))+"\n"+'{"step":3')
    rows, partial = audit.read_history(path)
    report = audit.history_audit(rows, partial)
    assert report["passed"] and report["latest_logged_step"] == 2 and partial
    rows[1]["losses"]["work"] = 1e-20
    assert not audit.history_audit(rows, False)["passed"]
    rows[1]["losses"]["work"] = 0.
    rows[1]["losses"]["solver_residual_ratio"] = 1.1
    assert not audit.history_audit(rows, False)["passed"]


def test_eta_does_not_invent_validation_and_marks_k16_assumption():
    qualification = dict(checks=dict(rows=[dict(horizon=4, forward_seconds=1., backward_seconds=.5),
                                           dict(horizon=8, forward_seconds=.4, backward_seconds=.3)]))
    result = audit.estimate_remaining([row(i) for i in range(1, 6)], qualification)
    assert result["K8_step_seconds_estimate"] == .8
    assert result["K16_step_seconds_estimate"] == 1.6
    assert result["K16_rate_source"] == "unmeasured_rough_assumption_4_times_measured_K4"
    assert result["K8_rate_source"] == "unmeasured_rough_assumption_2_times_measured_K4"
    assert result["qualification_timings_used_for_eta"] is False
    assert result["remaining_K4_steps"] == 995 and result["remaining_K8_steps"] == 3000
    assert result["remaining_K16_steps"] == 6000
    assert not result["validation_timing_available"]
    assert result["total_seconds_remaining_estimate"] is None


def test_eta_prefers_measured_k8_and_counts_pending_validation():
    rows = [row(3999, 8), row(4000, 8)]
    rows[0].update(validation={"score":.1}, validation_seconds=10.)
    result = audit.estimate_remaining(rows, {}, stop=10000)
    assert result["K8_rate_source"] == "measured_training_steps"
    assert result["K8_step_seconds_estimate"] == .4
    assert result["remaining_validation_events"] == 25
    assert result["total_seconds_remaining_estimate"] == 6000 * .8 + 25 * 10


def test_checkpoint_reads_adam_and_detects_inconsistent_steps_without_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim, "AdamW", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("optimizer")))
    initial = {"weight":torch.tensor([1., 2.])}
    run = dict(config={"objective_mode":"current"}, train_config={"training":{"steps":20000}},
               dataset_sha256="data", source_hashes={"a":"hash"}, validation_indices=[3,4])
    checkpoint = dict(run, step=2, sampler_next_step=2, state_dict={"weight":torch.tensor([1.1,2.])},
                      optimizer={"state":{0:{"step":torch.tensor(2.), "exp_avg":torch.zeros(2)}}})
    path = tmp_path / "last.pt"
    torch.save(checkpoint,path)
    qualification = dict(checks=dict(initial_state_sha256=audit.state_hash(initial)))
    result = audit.checkpoint_audit(path,run,qualification,{"latest_logged_step":1})
    assert result["passed"]  # A checkpoint can precede its completed log line.
    checkpoint["optimizer"]["state"][0]["step"] = torch.tensor(1.)
    torch.save(checkpoint,path)
    assert not audit.checkpoint_audit(path,run,qualification,{"latest_logged_step":2})["passed"]


def test_whole_pair_checks_different_gpus_and_common_input_contract(tmp_path,monkeypatch):
    base = dict(passed=True, initial_state_sha256="same", dataset_sha256="data", source_hashes={"a":"b"},
                validation_indices=[1], qualification_train_indices=[2])
    values = {arm:dict(copy.deepcopy(base), allocation=dict(device_uuid=arm)) for arm in ("control","lehami")}
    monkeypatch.setattr(audit,"inspect_arm",lambda _root,arm,_target:values[arm])
    assert audit.audit(tmp_path)["passed"]
    values["lehami"]["allocation"]["device_uuid"] = "control"
    assert not audit.audit(tmp_path)["passed"]


def test_missing_initial_files_yields_explicit_incomplete_report(tmp_path):
    report = audit.audit(tmp_path)
    assert not report["passed"]
    assert report["arms"]["control"]["inspection_error"] == "FileNotFoundError"
