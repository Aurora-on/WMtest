"""Retraining orchestration transactions; no optimization or scheduler calls."""
import copy
import json
from pathlib import Path

import pytest
import torch

from dpbench.comparison_protocol import digest, json_digest, make_config
from dpbench.train_dissipative import runtime_signature
from scripts import run_hjepa_retrain as pipeline


def config():
    cfg = make_config("hjepa", "single")
    cfg["training_route"] = "scratch"
    cfg["training"].update(batch_size=128, lr=5e-5, weight_decay=1e-3,
                           curriculum=[{"until_step": 20000, "horizon": 5}])
    return cfg


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = config()
    data, data_a = tmp_path / "single.h5", tmp_path / "a.h5"
    data.write_bytes(b"fixture single"); data_a.write_bytes(b"fixture paired a")
    sources = {"official_core": "a" * 64}
    monkeypatch.setattr(pipeline, "source_manifest", lambda: sources)
    monkeypatch.setattr(pipeline, "extra_source_manifest", lambda: {"wrapper": "b" * 64})
    monkeypatch.setattr(pipeline, "preflight", lambda path, cfg: {"dataset_sha256": digest(path)})
    monkeypatch.setattr(pipeline, "validate_pair", lambda *a, **kw: {"status": "matched"})
    manifest, report = pipeline.prepare(cfg, data, data_a, "cpu")
    calls, evaluations, verifications = [], [], []

    def save(path, step):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "comparison", "checkpoint_version": 1, "train_config": cfg,
                    "model_config": cfg["model"], "protocol_sha256": json_digest(cfg),
                    "dataset_sha256": digest(data), "source_sha256": sources,
                    "runtime": runtime_signature("cpu"), "step": step, "sampler_next_step": step,
                    "parent_transfer": None, "state_dict": {"weight": torch.tensor([float(step)])},
                    "validation_status": "completed"}, path)

    def train(data_path, run, train_config, **kw):
        assert train_config == cfg and train_config["training"]["steps"] == 20000
        assert "init_from" not in kw
        calls.append(kw)
        save(Path(run) / "last.pt", kw["stop_after_step"])
        return Path(run) / "last.pt"

    def evaluate(cp, data_path, output, **kw):
        assert kw["batch_size"] == 32 and kw["horizons"] == (8, 16, 32)
        saved = torch.load(cp, weights_only=True)
        evaluations.append(saved["step"])
        output.mkdir(parents=True)
        (output / "complete.json").write_text(json.dumps([digest(cp), digest(data_path),
                                                         json_digest(cfg), saved["step"]]))

    def verify(path, *expected):
        verifications.append(Path(path))
        assert json.loads((Path(path) / "complete.json").read_text()) == list(expected)

    monkeypatch.setattr(pipeline, "execute_run", train)
    monkeypatch.setattr(pipeline, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(pipeline, "verify_evaluation", verify)
    monkeypatch.setattr(pipeline, "qualify_evaluation", lambda path: {"qualified": True})
    return dict(cfg=cfg, data=data, data_a=data_a, manifest=manifest, report=report,
                root=tmp_path / "output", calls=calls, evaluations=evaluations,
                verifications=verifications, save=save)


def run(s, until=20000):
    return pipeline.execute(s["root"], s["cfg"], s["manifest"], s["report"], until)


def test_preflight_never_constructs_optimizer_or_output(setup, monkeypatch, tmp_path, capsys):
    s = setup
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("Unexpected optimizer"))
    monkeypatch.setattr(pipeline, "execute", lambda *a, **kw: pytest.fail("Unexpected execution"))
    path = tmp_path / "config.json"; path.write_text(json.dumps(s["cfg"]))
    code = pipeline.main(["--config", str(path), "--data", str(s["data"]), "--data-a", str(s["data_a"]),
                          "--output", str(s["root"])])
    assert code == 0 and not s["root"].exists()
    out = json.loads(capsys.readouterr().out)
    assert not out["training_started"] and not out["optimizer_created"]


@pytest.mark.parametrize("change", ["scratch_a", "tiny", "long_horizon", "small_batch", "wrong_lr", "wrong_decay"])
def test_rejects_silent_protocol_changes(change):
    cfg = config()
    if change == "scratch_a": cfg["data"].update(stage="a", selected_damping=[0.])
    if change == "tiny": cfg["model"]["tiny_fixture"] = True
    if change == "long_horizon": cfg["training"]["curriculum"][0]["horizon"] = 32
    if change == "small_batch": cfg["training"]["batch_size"] = 32
    if change == "wrong_lr": cfg["training"]["lr"] = 1e-4
    if change == "wrong_decay": cfg["training"]["weight_decay"] = 1e-4
    with pytest.raises(ValueError): pipeline.validate_retrain_config(cfg)


def test_4k_then_resume_keeps_snapshot_and_reentry_does_not_train(setup):
    s = setup
    first = run(s, 4000)
    cp = s["root"] / "milestones/step_004000.pt"
    frozen_hash = digest(cp)
    assert first["status"] == "qualified_at_4000" and first["completed_steps"] == 4000
    final = run(s)
    assert final["status"] == "completed" and final["completed_steps"] == 20000
    assert [c["stop_after_step"] for c in s["calls"]] == [4000, 20000]
    assert s["calls"][0]["resume"] is None
    assert s["calls"][1]["resume"] == s["root"] / "run/last.pt"
    assert s["evaluations"] == [4000, 20000] and digest(cp) == frozen_hash
    assert run(s) == final and len(s["calls"]) == 2
    assert s["evaluations"] == [4000, 20000]
    requested_earlier = run(s, 4000)
    assert requested_earlier["status"] == "already_beyond_requested_milestone"
    assert requested_earlier["completed_steps"] == 20000 and len(s["calls"]) == 2


def test_failed_acceptance_stops_and_cli_returns_two(setup, monkeypatch, tmp_path):
    s = setup
    monkeypatch.setattr(pipeline, "qualify_evaluation", lambda path: {"qualified": False, "reason": "fixture"})
    path = tmp_path / "config.json"; path.write_text(json.dumps(s["cfg"]))
    args = ["--config", str(path), "--data", str(s["data"]), "--data-a", str(s["data_a"]),
            "--output", str(s["root"]), "--execute"]
    assert pipeline.main(args) == 2
    assert [c["stop_after_step"] for c in s["calls"]] == [4000]
    assert pipeline.main(args) == 2 and len(s["calls"]) == 1
    assert json.loads((s["root"] / "status.json").read_text())["status"] == "not_qualified"


def test_missing_4k_snapshot_after_progress_cannot_be_recreated(setup):
    s = setup
    run(s, 4000)
    (s["root"] / "milestones/step_004000.pt").unlink()
    s["save"](s["root"] / "run/last.pt", 12000)
    with pytest.raises(ValueError, match="unpreserved milestone"):
        run(s)
    assert len(s["calls"]) == 1


def test_resume_in_middle_does_not_roll_back_4k(setup):
    s = setup
    run(s, 4000)
    cp = s["root"] / "milestones/step_004000.pt"
    checksum = digest(cp)
    s["save"](s["root"] / "run/last.pt", 12000)
    assert run(s)["completed_steps"] == 20000
    assert digest(cp) == checksum and len(s["calls"]) == 2


def test_changed_sources_and_tampered_milestone_are_rejected(setup):
    s = setup
    run(s, 4000)
    changed = copy.deepcopy(s["manifest"])
    changed["extra_sources"]["wrapper"] = "c" * 64
    with pytest.raises(ValueError, match="identity differs"):
        pipeline.execute(s["root"], s["cfg"], changed, s["report"])
    cp = s["root"] / "milestones/step_004000.pt"
    s["save"](cp, 4000)
    saved = torch.load(cp, weights_only=True)
    saved["state_dict"]["weight"] += 1
    torch.save(saved, cp)
    with pytest.raises(ValueError, match="different milestone bytes"):
        run(s)
    assert len(s["calls"]) == 1


def test_existing_evaluation_is_verified_before_acceptance(setup):
    s = setup
    run(s, 4000)
    (s["root"] / "evaluations/step_004000/complete.json").write_text("[]")
    with pytest.raises(AssertionError): run(s)
    assert len(s["calls"]) == 1 and s["evaluations"] == [4000]


@pytest.fixture
def staged(tmp_path, monkeypatch):
    paths = {name: tmp_path / f"{name}.h5" for name in ("a", "single", "multi")}
    for name, path in paths.items(): path.write_bytes(name.encode())
    configs, roots, specs, calls = {}, {}, {}, []
    sources = {"core": "a" * 64}
    monkeypatch.setattr(pipeline, "source_manifest", lambda: sources)
    monkeypatch.setattr(pipeline, "extra_source_manifest", lambda: {"wrapper": "b" * 64})
    monkeypatch.setattr(pipeline, "preflight", lambda path, cfg: {
        "dataset_sha256": digest(path), "parent_config_sha256": "paired-family"})
    monkeypatch.setattr(pipeline, "validate_pair", lambda *a, **kw: {"status": "matched"})
    for stage, damping in (("a", [0.]), ("single", [.2]), ("multi", [0., .05, .1, .2, .5, 1.])):
        cfg = config()
        cfg["training_route"] = "staged"
        cfg["data"].update(stage=stage, selected_damping=damping)
        configs[stage] = cfg
        roots[stage] = tmp_path / f"run_{stage}"
        specs[stage] = pipeline.prepare(cfg, paths[stage], paths["a"], "cpu",
                                       roots["a"] if stage != "a" else None)

    def train(data, run, cfg, **kw):
        transfer = None
        initial_state = {"predictor.r.weight": torch.tensor([23.]), "encoder.weight": torch.tensor([42.])}
        if kw.get("init_from"):
            assert not kw.get("resume")
            parent = torch.load(kw["init_from"], weights_only=True)
            assert parent["step"] == 20000 and parent["train_config"]["data"]["stage"] == "a"
            transfer = dict(checkpoint=str(Path(kw["init_from"]).resolve()), sha256=digest(kw["init_from"]),
                            step=20000, optimizer_reset=True, all_parameters_trainable=True,
                            independently_verified_data_sha256=kw["expected_parent_data_sha256"])
            initial_state = parent["state_dict"]
        elif kw.get("resume"):
            old = torch.load(kw["resume"], weights_only=True)
            transfer, initial_state = old["parent_transfer"], old["state_dict"]
        calls.append(dict(stage=cfg["data"]["stage"], **kw))
        run.mkdir(parents=True, exist_ok=True)
        step = kw["stop_after_step"]
        torch.save(dict(kind="comparison", checkpoint_version=1, train_config=cfg,
                        model_config=cfg["model"], protocol_sha256=json_digest(cfg), dataset_sha256=digest(data),
                        source_sha256=sources, runtime=runtime_signature("cpu"), step=step, sampler_next_step=step,
                        parent_transfer=transfer, state_dict=initial_state, validation_status="completed",
                        validation={"score": .1}, parent_config_sha256="paired-family"), run / "last.pt")
        return run / "last.pt"

    def evaluate(checkpoint, data, output, **kw):
        saved = torch.load(checkpoint, weights_only=True)
        output.mkdir(parents=True)
        (output / "complete.json").write_text(json.dumps([digest(checkpoint), digest(data),
                                saved["protocol_sha256"], saved["step"]]))

    def verify(path, *expected):
        if json.loads((path / "complete.json").read_text()) != list(expected):
            raise ValueError("Evaluation identity mismatch")

    monkeypatch.setattr(pipeline, "execute_run", train)
    monkeypatch.setattr(pipeline, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(pipeline, "verify_evaluation", verify)
    monkeypatch.setattr(pipeline, "qualify_evaluation", lambda path: {"qualified": True})
    def execute(stage, until=20000):
        return pipeline.execute(roots[stage], configs[stage], *specs[stage], until_step=until)
    return dict(configs=configs, roots=roots, specs=specs, paths=paths, calls=calls, execute=execute)


def test_staged_preflight_binds_parent_without_requiring_training(staged):
    s = staged
    assert not s["roots"]["a"].exists()
    assert s["specs"]["single"][0]["parent_output"] == str(s["roots"]["a"])
    with pytest.raises(ValueError, match="requires --parent-output"):
        pipeline.prepare(s["configs"]["single"], s["paths"]["single"], s["paths"]["a"], "cpu")
    with pytest.raises(ValueError, match="requires --parent-output"):
        pipeline.prepare(s["configs"]["a"], s["paths"]["a"], s["paths"]["a"], "cpu", s["roots"]["a"])
    with pytest.raises(ValueError, match="Stage A must train"):
        pipeline.prepare(s["configs"]["a"], s["paths"]["single"], s["paths"]["a"], "cpu")


def test_staged_b_never_trains_without_qualified_a20k(staged):
    s = staged
    with pytest.raises(FileNotFoundError): s["execute"]("single")
    assert not s["calls"] and not s["roots"]["single"].exists()
    s["execute"]("a", 4000)
    with pytest.raises(ValueError, match="completed and qualified"):
        s["execute"]("single")
    assert [c["stage"] for c in s["calls"]] == ["a"]
    assert not s["roots"]["single"].exists()


def test_staged_b_transfers_once_then_resumes_same_parent_for_both_branches(staged):
    s = staged
    assert s["execute"]("a")["completed_steps"] == 20000
    assert s["execute"]("a", 4000)["status"] == "already_beyond_requested_milestone"
    parent = s["roots"]["a"] / "milestones/step_020000.pt"
    for branch in ("single", "multi"):
        assert s["execute"](branch, 4000)["qualified"]
        assert s["execute"](branch)["completed_steps"] == 20000
        calls = [c for c in s["calls"] if c["stage"] == branch]
        assert len(calls) == 2
        assert calls[0]["init_from"] == parent and calls[0]["resume"] is None
        assert calls[0]["expected_parent_data_sha256"] == digest(s["paths"]["a"])
        assert "init_from" not in calls[1] and calls[1]["resume"] is not None
        saved = torch.load(s["roots"][branch] / "milestones/step_020000.pt", weights_only=True)
        assert saved["parent_transfer"]["sha256"] == digest(parent)
        assert saved["parent_transfer"]["optimizer_reset"]
        torch.testing.assert_close(saved["state_dict"]["predictor.r.weight"], torch.tensor([23.]))
    assert all("init_from" not in c for c in s["calls"] if c["stage"] == "a")


@pytest.mark.parametrize("fault", ["acceptance", "source", "parent_route", "checkpoint", "evaluation"])
def test_staged_b_rejects_unqualified_or_changed_parent(staged, monkeypatch, fault):
    s = staged
    s["execute"]("a")
    parent = s["roots"]["a"]
    if fault == "acceptance":
        monkeypatch.setattr(pipeline, "qualify_evaluation", lambda path: {"qualified": False})
    elif fault in {"source", "parent_route"}:
        path = parent / "pipeline.json"; value = json.loads(path.read_text())
        if fault == "source": value["extra_sources"]["wrapper"] = "c" * 64
        else: value["config"]["training_route"] = "scratch"
        path.write_text(json.dumps(value))
    elif fault == "checkpoint":
        path = parent / "milestones/step_020000.pt"
        value = torch.load(path, weights_only=True)
        value["state_dict"]["predictor.r.weight"] += 1
        torch.save(value, path)
    else:
        (parent / "evaluations/step_020000/complete.json").write_text("[]")
    with pytest.raises(ValueError): s["execute"]("single")
    assert len(s["calls"]) == 2 and all(c["stage"] == "a" for c in s["calls"])
    assert not s["roots"]["single"].exists()


def test_staged_resume_binds_parent_transfer_provenance(staged):
    s = staged
    s["execute"]("a")
    s["execute"]("single", 4000)
    checkpoint = s["roots"]["single"] / "run/last.pt"
    saved = torch.load(checkpoint, weights_only=True)
    saved["parent_transfer"]["sha256"] = "c" * 64
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="initialization provenance"):
        s["execute"]("single")
    assert len(s["calls"]) == 3


def test_verify_parent_cli_is_read_only_and_rejects_incomplete_or_non_b(staged, monkeypatch, tmp_path, capsys):
    s = staged
    config_file = tmp_path / "single.json"
    config_file.write_text(json.dumps(s["configs"]["single"]))
    args = ["--config", str(config_file), "--data", str(s["paths"]["single"]),
            "--data-a", str(s["paths"]["a"]), "--parent-output", str(s["roots"]["a"]),
            "--output", str(s["roots"]["single"]), "--verify-parent"]
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **kw: pytest.fail("Unexpected optimizer"))
    with pytest.raises(FileNotFoundError): pipeline.main(args)
    assert not s["roots"]["single"].exists() and not s["calls"]
    s["execute"]("a")
    capsys.readouterr()
    assert pipeline.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["parent_verified"] and not report["optimizer_created"]
    assert not s["roots"]["single"].exists() and len(s["calls"]) == 2
    with pytest.raises(SystemExit): pipeline.main(args + ["--execute"])
    config_file.write_text(json.dumps(s["configs"]["a"]))
    with pytest.raises(SystemExit):
        pipeline.main(["--config", str(config_file), "--data", str(s["paths"]["a"]),
                       "--data-a", str(s["paths"]["a"]), "--output", str(tmp_path / "unused"),
                       "--verify-parent"])
