"""Launch/evaluation transactions without submitting jobs or training models."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dpbench import evaluate_comparison, train_comparison
from dpbench.comparison_protocol import digest, json_digest, make_config
from scripts import run_comparison as pipeline


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def arguments(tmp_path, route="staged"):
    paths = {}
    for name in ("a", "single", "multi"):
        path = tmp_path / (name + ".h5")
        path.write_bytes(("fixture-data-" + name).encode())
        paths["data_" + name] = path
    return SimpleNamespace(methods=["lehami"], seeds=[0], route=route, output=tmp_path / "pipeline",
                           device="cpu", gpus="0,1", skip_evaluation=False, **paths)


def test_default_plan_is_read_only_and_does_not_launch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "execute", lambda *a, **kw: pytest.fail("No execution without --execute"))
    monkeypatch.setattr(pipeline.subprocess, "Popen", lambda *a, **kw: pytest.fail("No process launch in planning"))
    monkeypatch.setattr(pipeline, "preflight", lambda *a, **kw: pytest.fail("Planning does not read data"))
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("No optimizer in planning"))
    output = tmp_path / "must-not-exist"
    pipeline.main(["--output", str(output), "--data-a", str(tmp_path / "absent-a.h5")])
    spec = json.loads(capsys.readouterr().out)
    assert not output.exists() and list(tmp_path.iterdir()) == []
    assert spec["formal_training_started"] is False and spec["route"] == "staged"
    assert len(spec["jobs"]) == 3 and spec["single_path_budget"] == 40000
    assert spec["expected_optimizer_updates_per_method_seed"] == 60000
    assert all(set(j["configs"]) == {"a", "single", "multi"} for j in spec["jobs"])
    assert all(c["training"]["steps"] == 20000 for j in spec["jobs"] for c in j["configs"].values())


@pytest.mark.parametrize("route", ["staged", "scratch"])
def test_stage_dependencies_and_separate_parallel_branches(tmp_path, monkeypatch, route):
    args = arguments(tmp_path, route)
    spec = pipeline.plan(args)
    sources = {"fixture": "a" * 64}
    monkeypatch.setattr(pipeline, "source_manifest", lambda: sources)
    monkeypatch.setattr(pipeline, "validate_pair", lambda *a, **kw: {"paired": True})
    monkeypatch.setattr(pipeline, "preflight", lambda *a, **kw: {"fixture": True})
    monkeypatch.setattr(pipeline, "environments", lambda *a, **kw: [{"slot": "0"}, {"slot": "1"}])
    launches = []
    def launch(paths, envs):
        tasks = [json.loads(Path(p).read_text()) for p in paths]
        launches.append(tasks)
        for task in tasks:
            out = Path(task["output"])
            cp = out / "segments/0001/last.pt"
            cp.parent.mkdir(parents=True)
            torch.save({"step": task["config"]["training"]["steps"], "state_dict": {"x": torch.zeros(1)}}, cp)
            write_json(out / "checkpoint_manifest.json", {"path": str(cp.resolve()), "sha256": digest(cp)})
            write_json(out / "worker_status.json", {"status": "completed"})
    monkeypatch.setattr(pipeline, "launch", launch)
    pipeline.execute(args, spec)
    branches = launches[-1]
    assert len(branches) == 2 and {t["config"]["data"]["stage"] for t in branches} == {"single", "multi"}
    assert len({t["output"] for t in branches}) == len({t["data"] for t in branches}) == 2
    if route == "staged":
        assert len(launches) == 2 and len(launches[0]) == 1
        assert launches[0][0]["config"]["data"]["stage"] == "a"
        a_checkpoint = Path(launches[0][0]["output"]) / "segments/0001/last.pt"
        assert all(Path(t["parent"]) == a_checkpoint.resolve() for t in branches)
        assert all(t["parent_sha256"] == digest(a_checkpoint) for t in branches)
        assert all(t["config"]["training"]["steps"] == 20000 for t in branches)
    else:
        assert len(launches) == 1 and all(t["parent"] is None for t in branches)
        assert all(t["config"]["training"]["steps"] == 40000 for t in branches)
        assert spec["expected_optimizer_updates_per_method_seed"] == 80000
        assert set(spec["jobs"][0]["configs"]) == {"single", "multi"}
    assert json.loads((args.output / "status.json").read_text())["status"] == "completed"


def worker_fixture(tmp_path, monkeypatch, skip_evaluation=False):
    data = tmp_path / "data.h5"
    data.write_bytes(b"fixture dataset, preflight is replaced")
    config = make_config("lehami", "single")
    config["training_route"] = "scratch"
    sources = {"fixture.py": "a" * 64}
    out = tmp_path / "worker"
    cp = out / "segments/0001/last.pt"
    cp.parent.mkdir(parents=True)
    saved = dict(kind="comparison", checkpoint_version=1, train_config=config, model_config=config["model"],
        protocol_sha256=json_digest(config), dataset_sha256=digest(data), source_sha256=sources,
        state_dict={"fixture.weight": torch.zeros(1)}, step=20000, sampler_next_step=20000,
        validation_status="completed")
    torch.save(saved, cp)
    task = dict(config=config, data=str(data), data_sha256=digest(data), sources=sources,
                output=str(out), device="cpu", parent=None, skip_evaluation=skip_evaluation)
    task_path = tmp_path / "task.json"
    write_json(task_path, task)
    monkeypatch.setattr(pipeline, "source_manifest", lambda: sources)
    monkeypatch.setattr(pipeline, "preflight", lambda *a, **kw: {"dataset_sha256": digest(data)})
    monkeypatch.setattr(train_comparison, "execute_run", lambda *a, **kw: pytest.fail("Exact completed checkpoint must not retrain"))
    return task_path, task, cp


def publish_evaluation(task, checkpoint):
    """Real JSON/NPZ bytes with the evaluator's completion/identity contract."""
    root = Path(task["output"]) / "evaluation"
    root.mkdir()
    (root / "val").mkdir()
    protocol = {"horizons": [8, 16, 32], "split": "val"}
    identity = dict(checkpoint_sha256=digest(checkpoint), dataset_sha256=task["data_sha256"],
                    protocol_sha256=json_digest(task["config"]), evaluation_protocol_sha256=json_digest(protocol))
    result = {"split": "val", "horizon_steps": 32, "rollout": {str(h): {} for h in (8, 16, 32)}}
    np.savez(root / "readout.npz", selected_k=5)
    np.savez(root / "val/trajectories.npz", time=np.arange(33) * .1)
    write_json(root / "readout_provenance.json", dict(identity, evaluation_protocol=protocol, checkpoint_step=20000,
        readout_sha256=digest(root / "readout.npz")))
    write_json(root / "summary.json", dict(identity, status="completed", checkpoint_step=20000,
        evaluation_protocol=protocol, evaluation={"val": result}))
    write_json(root / "val/summary.json", result)
    artifacts = ["readout.npz", "readout_provenance.json", "summary.json", "val/summary.json", "val/trajectories.npz"]
    write_json(root / "complete.json", dict(identity, status="completed", artifacts={name: digest(root / name) for name in artifacts}))
    return root


def test_evaluation_failure_never_marks_worker_completed(tmp_path, monkeypatch):
    task_path, task, _ = worker_fixture(tmp_path, monkeypatch)
    def fail(*a, **kw):
        raise RuntimeError("fixture evaluation failure")
    monkeypatch.setattr(evaluate_comparison, "evaluate_checkpoint", fail)
    with pytest.raises(RuntimeError, match="evaluation failure"):
        pipeline.worker(task_path)
    out = Path(task["output"])
    assert json.loads((out / "worker_status.json").read_text())["status"] == "failed"
    assert not (out / "evaluation").exists()


def test_partial_evaluation_return_is_not_published(tmp_path, monkeypatch):
    task_path, task, _ = worker_fixture(tmp_path, monkeypatch)
    def partial(checkpoint, data, output, **kwargs):
        write_json(Path(output) / "summary.json", {"status": "completed"})
        return {"status": "completed"}
    monkeypatch.setattr(evaluate_comparison, "evaluate_checkpoint", partial)
    with pytest.raises((ValueError, FileNotFoundError)):
        pipeline.worker(task_path)
    out = Path(task["output"])
    assert json.loads((out / "worker_status.json").read_text())["status"] == "failed"
    assert not (out / "evaluation").exists()
    assert list((out / "evaluation_attempts").glob("*/summary.json"))


def test_skip_evaluation_is_explicitly_not_completed(tmp_path, monkeypatch):
    task_path, task, _ = worker_fixture(tmp_path, monkeypatch, skip_evaluation=True)
    monkeypatch.setattr(evaluate_comparison, "evaluate_checkpoint", lambda *a, **kw: pytest.fail("Evaluation explicitly skipped"))
    pipeline.worker(task_path)
    assert json.loads((Path(task["output"]) / "worker_status.json").read_text())["status"] == "trained_not_evaluated"


@pytest.mark.parametrize("changed", ["source", "data"])
def test_worker_rejects_source_or_data_identity_changes(tmp_path, monkeypatch, changed):
    task_path, task, _ = worker_fixture(tmp_path, monkeypatch)
    if changed == "source":
        monkeypatch.setattr(pipeline, "source_manifest", lambda: {"fixture.py": "b" * 64})
    else:
        Path(task["data"]).write_bytes(b"changed bytes")
    monkeypatch.setattr(pipeline, "preflight", lambda *a, **kw: pytest.fail("Identity audit must precede data/model work"))
    monkeypatch.setattr(evaluate_comparison, "evaluate_checkpoint", lambda *a, **kw: pytest.fail("Changed inputs must never evaluate"))
    with pytest.raises(ValueError, match="sources/data changed"):
        pipeline.worker(task_path)
    assert json.loads((Path(task["output"]) / "worker_status.json").read_text())["status"] == "failed"


def test_existing_valid_evaluation_reused_without_training_or_inference(tmp_path, monkeypatch):
    task_path, task, cp = worker_fixture(tmp_path, monkeypatch)
    root = publish_evaluation(task, cp)
    before = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(evaluate_comparison, "evaluate_checkpoint", lambda *a, **kw: pytest.fail("Verified publication should be reused"))
    pipeline.worker(task_path)
    after = {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert json.loads((Path(task["output"]) / "worker_status.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("corruption", ["empty_artifacts", "failed_status", "failed_summary", "different_data",
                                        "wrong_horizons", "missing_result_horizon", "wrong_summary_step",
                                        "different_val_summary", "modified_npz"])
def test_invalid_existing_evaluation_never_reused(tmp_path, monkeypatch, corruption):
    task_path, task, cp = worker_fixture(tmp_path, monkeypatch)
    root = publish_evaluation(task, cp)
    done = json.loads((root / "complete.json").read_text())
    if corruption == "empty_artifacts":
        done["artifacts"] = {}
    elif corruption == "failed_status":
        done["status"] = "failed"
    elif corruption == "different_data":
        done["dataset_sha256"] = "b" * 64
    elif corruption in ("wrong_horizons", "failed_summary", "missing_result_horizon", "wrong_summary_step"):
        summary = json.loads((root / "summary.json").read_text())
        if corruption == "failed_summary":
            summary["status"] = "failed"
        elif corruption == "wrong_summary_step":
            summary["checkpoint_step"] = 10000
        else:
            summary["evaluation"]["val"]["rollout"] = {"8": {}}
        if corruption == "wrong_horizons":
            summary["evaluation_protocol"]["horizons"] = [8]
        write_json(root / "summary.json", summary)
        done["artifacts"]["summary.json"] = digest(root / "summary.json")
    elif corruption == "different_val_summary":
        validation = json.loads((root / "val/summary.json").read_text())
        validation["horizon_steps"] = 8
        write_json(root / "val/summary.json", validation)
        done["artifacts"]["val/summary.json"] = digest(root / "val/summary.json")
    else:
        with (root / "val/trajectories.npz").open("ab") as f:
            f.write(b"changed archive bytes")
    write_json(root / "complete.json", done)
    monkeypatch.setattr(evaluate_comparison, "evaluate_checkpoint", lambda *a, **kw: pytest.fail("Corrupt publication requires explicit inspection"))
    with pytest.raises(ValueError):
        pipeline.worker(task_path)
    assert json.loads((Path(task["output"]) / "worker_status.json").read_text())["status"] == "failed"
