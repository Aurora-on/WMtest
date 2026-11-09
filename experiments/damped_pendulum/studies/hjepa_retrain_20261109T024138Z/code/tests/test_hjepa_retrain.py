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


@pytest.mark.parametrize("change", ["staged", "tiny", "long_horizon", "small_batch", "wrong_lr", "wrong_decay"])
def test_rejects_silent_protocol_changes(change):
    cfg = config()
    if change == "staged": cfg["training_route"] = "staged"
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
