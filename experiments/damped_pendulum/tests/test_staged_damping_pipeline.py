"""Milestone/parallelism/recovery tests without running a long training job."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("staged_pipeline", ROOT / "scripts/train_staged_damping.py")
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


def configs():
    return pipeline.build_configs(pipeline.read_json(ROOT / "configs/dissipative_loss_pair_lehami_v1.json"))


def test_plan_has_two_branches_but_keeps_a_exact_resume_config_and_physical_clock():
    args = pipeline.parser().parse_args([])
    plan = pipeline.experiment_plan(args)
    cfg = plan["configs"]
    assert cfg["a"] == pipeline.read_json(args.config_a)
    assert "q_dim" not in cfg["a"]["model"]  # adding the implicit default would break resume digest
    assert cfg["single"]["training"]["trainable_modules"] == ["damping"]
    assert cfg["multi"]["training"]["trainable_modules"] == ["damping", "environment"]
    assert cfg["multi"]["model"]["environment_context_dim"] == 4
    assert cfg["single"]["model"]["environment_context_dim"] == 0
    for name in pipeline.BRANCHES:
        original = copy.deepcopy(cfg[name])
        original["model"]["core"]["damping_mode"] = "none"
        original["model"]["environment_context_dim"] = 0
        original["training"]["trainable_modules"] = ["encoder", "initializer", "potential"]
        original["training"]["required_data_scope"] = "undamped_stage_a"
        assert original == cfg["a"]
    assert plan["seen_damping"] == [0, .05, .1, .2, .5, 1]


def test_default_cli_is_plan_only_and_never_creates_output(tmp_path):
    output = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/train_staged_damping.py"),
                             "--output", str(output)], check=True, text=True, capture_output=True)
    assert json.loads(result.stdout)["training_started"] is False
    assert not output.exists()


def fake_check(path, config, data_hash, device, target=None):
    saved = pipeline.load_checkpoint(path)
    if target is not None and (saved["step"] != target or saved.get("validation_status") != "completed"):
        raise ValueError("not exact validated target")
    return saved


def checkpoint(path, step, status="completed", value=1.):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "validation_status": status,
                "state_dict": {"encoder.weight": torch.tensor([value])}}, path)
    return path


def test_seal_uses_exact_last_not_earlier_best_and_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "check_checkpoint", fake_check)
    last = checkpoint(tmp_path / "run/last.pt", 20000)
    checkpoint(last.parent / "best.pt", 19750, value=2.)
    dest = tmp_path / "milestone"
    result = pipeline.seal_checkpoint(last, dest, configs()["a"], "data", "cpu")
    assert pipeline.load_checkpoint(result)["step"] == 20000
    assert pipeline.load_checkpoint(dest / "best.pt")["step"] == 19750
    assert pipeline.seal_checkpoint(last, dest, configs()["a"], "data", "cpu") == result
    checkpoint(last, 20000, value=4.)
    with pytest.raises(ValueError, match="does not match"):
        pipeline.seal_checkpoint(last, dest, configs()["a"], "data", "cpu")


@pytest.mark.parametrize("step,status", [(19750, "completed"), (20000, "not_run")])
def test_partial_or_unvalidated_checkpoint_cannot_be_sealed(tmp_path, monkeypatch, step, status):
    monkeypatch.setattr(pipeline, "check_checkpoint", fake_check)
    last = checkpoint(tmp_path / "run/last.pt", step, status)
    with pytest.raises(ValueError):
        pipeline.seal_checkpoint(last, tmp_path / "milestone", configs()["a"], "data", "cpu")
    assert not (tmp_path / "milestone").exists()


def test_resume_uses_latest_recovery_point_then_skips_completed_training(tmp_path, monkeypatch):
    from dpbench import train_dissipative
    monkeypatch.setattr(pipeline, "check_checkpoint", fake_check)
    data = tmp_path / "data.h5"
    data.write_bytes(b"fixture")
    stage = tmp_path / "stage"
    initial = checkpoint(tmp_path / "initial.pt", 10000)
    prior = checkpoint(stage / "segments/0001/last.pt", 15000)
    calls = []

    def train(data, output, config, device, resume, stop_after_step):
        calls.append((output, resume, stop_after_step))
        checkpoint(output / "last.pt", stop_after_step)

    monkeypatch.setattr(train_dissipative, "execute_run", train)
    final = pipeline.train_to_target(stage, data, configs()["a"], "cpu", initial)
    assert calls == [(stage / "segments/0002", prior, 20000)]
    assert pipeline.train_to_target(stage, data, configs()["a"], "cpu", initial) == final
    assert len(calls) == 1


def test_pending_final_validation_is_retried_without_additional_budget(tmp_path, monkeypatch):
    from dpbench import train_dissipative
    monkeypatch.setattr(pipeline, "check_checkpoint", fake_check)
    data = tmp_path / "data.h5"
    data.write_bytes(b"fixture")
    stage = tmp_path / "stage"
    prior = checkpoint(stage / "segments/0001/last.pt", 20000, "not_run")
    calls = []

    def train(data, output, config, device, resume, stop_after_step):
        calls.append((resume, stop_after_step))
        checkpoint(output / "last.pt", 20000)

    monkeypatch.setattr(train_dissipative, "execute_run", train)
    pipeline.train_to_target(stage, data, configs()["a"], "cpu")
    assert calls == [(prior, 20000)]


def test_corrupt_latest_checkpoint_is_not_silently_replaced_by_older(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "check_checkpoint", fake_check)
    stage = tmp_path / "stage"
    initial = checkpoint(tmp_path / "initial.pt", 10000)
    checkpoint(stage / "segments/0001/last.pt", 15000)
    corrupt = stage / "segments/0002/last.pt"
    corrupt.parent.mkdir()
    corrupt.write_bytes(b"incomplete")
    with pytest.raises(Exception):
        pipeline.select_resume(stage, initial, configs()["a"], "data", "cpu")


def test_both_workers_start_before_any_wait_and_peer_finishes_if_one_fails(tmp_path, monkeypatch):
    events = []

    class Process:
        def __init__(self, command, **kwargs):
            self.name = command[command.index("--worker") + 1]
            self.code = 1 if self.name == "single" else 0
            self.done = False
            events.append(("start", self.name, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))

        def wait(self, **kwargs):
            self.done = True
            events.append(("wait", self.name))
            return self.code

        def poll(self):
            return self.code if self.done else None

    monkeypatch.setattr(pipeline.subprocess, "Popen", Process)
    with pytest.raises(RuntimeError, match="single"):
        pipeline.launch_workers(tmp_path, ["single", "multi"],
                                [{"CUDA_VISIBLE_DEVICES": "GPU-a"}, {"CUDA_VISIBLE_DEVICES": "GPU-b"}])
    assert events == [("start", "single", "GPU-a"), ("start", "multi", "GPU-b"),
                      ("wait", "single"), ("wait", "multi")]


def test_failed_a_prevents_either_b_start_and_records_failure(tmp_path, monkeypatch):
    args = pipeline.parser().parse_args(["--data-a", str(tmp_path / "a.h5"),
                                        "--output", str(tmp_path / "run"), "--device", "cpu"])
    monkeypatch.setattr(pipeline, "source_snapshot", lambda: {})
    monkeypatch.setattr(pipeline, "prepare", lambda *a: {})
    calls = []

    def launch(root, names, envs):
        calls.append(names)
        if names == ["a"]:
            raise RuntimeError("A evaluation failed")

    monkeypatch.setattr(pipeline, "launch_workers", launch)
    with pytest.raises(RuntimeError):
        pipeline.execute(args, pipeline.experiment_plan(args))
    assert calls == [["qualify_a"], ["a"]]
    assert pipeline.read_json(args.output / "status.json")["status"] == "failed"


def test_frozen_weight_drift_is_detected_even_when_new_damping_tensors_exist(tmp_path):
    a = checkpoint(tmp_path / "a.pt", 20000)
    b = checkpoint(tmp_path / "b.pt", 20000)
    saved = pipeline.load_checkpoint(b)
    saved["state_dict"]["core.damping_net.weight"] = torch.tensor([3.])
    torch.save(saved, b)
    assert pipeline.verify_frozen_state(a, b)["all_parent_tensors_identical"]
    saved["state_dict"]["encoder.weight"] += .01
    torch.save(saved, b)
    with pytest.raises(ValueError, match="encoder.weight"):
        pipeline.verify_frozen_state(a, b)


def test_evaluation_is_atomic_reusable_and_detects_artifact_tampering(tmp_path, monkeypatch):
    cp = tmp_path / "last.pt"
    data = tmp_path / "data.h5"
    cp.write_bytes(b"checkpoint")
    data.write_bytes(b"data")
    output = tmp_path / "evaluation"
    calls = []

    def evaluate(command, **kwargs):
        calls.append(command)
        attempt = Path(command[command.index("--output") + 1])
        attempt.mkdir()
        for name in ("summary.json", "trajectories.npz", "readout.npz", "readout_provenance.json"):
            (attempt / name).write_bytes(b"artifact")

    monkeypatch.setattr(pipeline.subprocess, "run", evaluate)
    pipeline.evaluate(cp, data, output, "a", 20000, "cpu")
    pipeline.evaluate(cp, data, output, "a", 20000, "cpu")
    assert len(calls) == 1
    (output / "readout.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        pipeline.evaluate(cp, data, output, "a", 20000, "cpu")


def test_gpu_mapping_respects_scheduler_allocation_and_rejects_duplicate(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    envs = pipeline.gpu_environments("cuda", "0,1")
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["3", "7"]
    with pytest.raises(ValueError, match="distinct"):
        pipeline.gpu_environments("cuda", "0,0")


def test_failed_qualification_prevents_expensive_data_generation(tmp_path, monkeypatch):
    args = pipeline.parser().parse_args(["--data-a", str(tmp_path / "a.h5"),
                                        "--output", str(tmp_path / "run"), "--device", "cpu"])
    monkeypatch.setattr(pipeline, "source_snapshot", lambda: {})
    monkeypatch.setattr(pipeline, "prepare", lambda *a: pytest.fail("Data generation ran before A qualified"))

    def launch(root, names, envs):
        assert names == ["qualify_a"]
        raise RuntimeError("A runtime mismatch")

    monkeypatch.setattr(pipeline, "launch_workers", launch)
    with pytest.raises(RuntimeError, match="runtime mismatch"):
        pipeline.execute(args, pipeline.experiment_plan(args))


def test_qualification_checks_resume_runtime_without_optimizer(tmp_path, monkeypatch):
    from dpbench import train_dissipative as trainer
    from dpbench.dissipative_model import resolve_model_config
    from dpbench.passive_family import _parent_config_sha256
    data = tmp_path / "a.h5"
    data.write_bytes(b"only-preflight-is-mocked")
    resume = tmp_path / "input/last.pt"
    args = pipeline.parser().parse_args(["--data-a", str(data), "--resume-a", str(resume), "--device", "cpu"])
    plan = pipeline.experiment_plan(args)
    cfg = plan["configs"]["a"]
    manifest = {"plan": plan, "source_sha256": {}}
    pipeline.atomic_json(tmp_path / "manifest.json", {
        "config": plan["parent_data_config"],
        "parent_config_sha256": _parent_config_sha256(plan["parent_data_config"]),
        "datasets": [{"file": "a.h5", "sha256": pipeline.digest(data)}],
        "source_sha256": {n: pipeline.digest(ROOT / n) for n in ("dpbench/passive_family.py", "dpbench/generate.py")}})
    saved = dict(format_version=1, kind="dissipative", train_config=cfg,
        config=resolve_model_config(cfg["model"]), config_sha256=trainer.config_digest(cfg),
        dataset_sha256=pipeline.digest(data), source_hashes=trainer.source_hashes(), device="cpu",
        step=10000, sampler_next_step=10000, validation_status="completed", runtime={"version": "old"},
        state_dict={"encoder.weight": torch.tensor([1.])}, optimizer={}, rng={"torch_cuda": []},
        validation_indices=[0], best_score=None, best_step=None)
    resume.parent.mkdir()
    torch.save(saved, resume)
    monkeypatch.setattr(trainer, "preflight", lambda *a: {"dataset_sha256": pipeline.digest(data)})
    monkeypatch.setattr(trainer, "runtime_signature", lambda *a: {"version": "new"})
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("Qualification constructed optimizer"))
    with pytest.raises(ValueError, match="runtime"):
        pipeline.qualify_stage_a(tmp_path, manifest)
    assert not (tmp_path / "stage_a_qualification.json").exists()
    saved["runtime"] = {"version": "new"}
    torch.save(saved, resume)
    pipeline.qualify_stage_a(tmp_path, manifest)
    result = pipeline.read_json(tmp_path / "stage_a_qualification.json")
    assert result["resume_step"] == 10000 and result["optimizer_created"] is False
