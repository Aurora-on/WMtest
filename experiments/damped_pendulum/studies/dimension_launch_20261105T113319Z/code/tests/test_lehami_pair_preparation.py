"""Boundary checks for the paired launcher; never execute real training."""
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_lehami_loss_pair.py"
spec = importlib.util.spec_from_file_location("loss_pair_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def pair_configs():
    return {arm: json.loads(path.read_text()) for arm, path in launcher.CONFIGS.items()}


def test_pair_has_only_declared_objective_differences_and_preserves_full_schedule():
    configs = pair_configs()
    launcher.check_pair_contract(configs)
    control, lehami = configs["control"], configs["lehami"]
    assert control["training"] == lehami["training"]
    assert control["training"]["steps"] == 20000
    assert control["training"]["curriculum"][-1] == {"until_step": 20000, "horizon": 32}
    assert control["model"]["target_tail_frames"] == lehami["model"]["target_tail_frames"] == 2
    changed = copy.deepcopy(configs)
    changed["lehami"]["training"]["seed"] = 1
    with pytest.raises(ValueError):
        launcher.check_pair_contract(changed)
    changed = copy.deepcopy(configs)
    changed["lehami"]["model"]["velocity_history"] = 4
    with pytest.raises(ValueError, match="outside"):
        launcher.check_pair_contract(changed)


def test_default_cli_does_not_import_torch_or_create_outputs(tmp_path):
    code = (
        "import runpy,sys; sys.argv=[sys.argv[1], '--data', sys.argv[2], '--output-root', sys.argv[3]]; "
        "runpy.run_path(sys.argv[0],run_name='pair_readonly'); "
        "assert 'torch' not in sys.modules"
    )
    # Also exercise main in a normal process; absence of data must not block planning.
    output = tmp_path / "never-created"
    result = subprocess.run([sys.executable, str(SCRIPT), "--data", str(tmp_path / "absent.h5"),
                             "--output-root", str(output)], text=True, capture_output=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["mode"] == "plan_only" and not plan["training_started"]
    assert not plan["optimizer_created"] and not output.exists()
    subprocess.run([sys.executable, "-c", code, str(SCRIPT), str(tmp_path / "absent.h5"), str(output)],
                   check=True, text=True, capture_output=True)
    for row in plan["arms"]:
        command = row["training_command"]
        assert "--resume" not in command and "--steps" not in command
        assert command[command.index("--stop-after-step") + 1] == "4000"
    assert not output.exists()


@pytest.fixture
def execution_harness(tmp_path, monkeypatch):
    data = tmp_path / "fixture bytes.h5"
    data.write_bytes(b"not-hdf5: the subprocess is intercepted")
    calls = []
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs: calls.append(command))
    plan = launcher.build_plan(data=data, output_root=tmp_path / "fresh output",
                               python="/custom python", device="cpu",
                               expected_data_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    return plan, calls


def test_preflight_is_both_configs_without_training_or_output(execution_harness):
    plan, calls = execution_harness
    result = launcher.run_plan(plan, preflight=True)
    assert len(calls) == 2 and all("--execute" not in c for c in calls)
    assert result["dataset_locally_verified"] and result["mode"] == "data_preflight"
    assert not Path(plan["output_root"]).exists()


def test_execute_preflights_both_before_running_fresh_selected_arm(execution_harness):
    plan, calls = execution_harness
    plan["arms"] = [row for row in plan["arms"] if row["arm"] == "lehami"]
    result = launcher.run_plan(plan, execute=True)
    assert len(calls) == 3 and "--execute" not in calls[0] and "--execute" not in calls[1]
    assert "--execute" in calls[2] and "lehami" in calls[2][calls[2].index("--output") + 1]
    assert not any("--resume" in command or "--steps" in command for command in calls)
    assert result["mode"] == "completed_requested_segment"


def test_data_hash_mismatch_prevents_any_subprocess(execution_harness):
    plan, calls = execution_harness
    plan["expected_data_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        launcher.run_plan(plan, execute=True)
    assert calls == []


def test_existing_second_arm_blocks_first_arm_before_any_subprocess(execution_harness):
    plan, calls = execution_harness
    Path(plan["arms"][1]["output"]).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="existing"):
        launcher.run_plan(plan, execute=True)
    assert calls == []


def test_failed_preflight_never_reaches_optimizer_boundary(execution_harness, monkeypatch):
    plan, calls = execution_harness
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(launcher.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        launcher.run_plan(plan, execute=True)
    assert len(calls) == 1 and "--execute" not in calls[0]


def test_stop_and_mode_boundaries_preserve_fresh_run_contract():
    with pytest.raises(ValueError):
        launcher.build_plan(stop_after_step=4001)
    plan = launcher.build_plan(arm="control", stop_after_step=20000)
    assert len(plan["arms"]) == 1 and plan["stop_after_step"] == 20000
    with pytest.raises(ValueError):
        launcher.run_plan(plan, preflight=True, execute=True)


def test_protocol_config_hashes_and_selection_match_actual_configs():
    protocol = json.loads((ROOT / "studies/lehami_loss_pair_20261105/protocol.json").read_text())
    assert not protocol["training_started"]
    for arm, row in protocol["arms"].items():
        assert row["config_sha256"] == launcher.sha256_file(ROOT / row["config"])
        config = pair_configs()[arm]
        assert config["training"]["validation_score"] == "position_momentum"
        assert config["training"]["validation_motion_weight"] == 0
    assert protocol["data"]["sha256"] == launcher.DATA_SHA256


@pytest.mark.parametrize("changed_input", ["source", "control_config", "lehami_config"])
def test_inputs_changed_between_arms_prevent_second_training(tmp_path, monkeypatch, changed_input):
    # Use isolated files: the test must never edit the actual workspace inputs.
    configs = pair_configs()
    paths = {}
    for arm, config in configs.items():
        paths[arm] = tmp_path / f"{arm}.json"
        paths[arm].write_text(json.dumps(config))
    source = tmp_path / "mock_model.py"
    source.write_text("MODEL_VERSION = 1\n")
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "CONFIGS", paths)
    monkeypatch.setattr(launcher, "SOURCE_FILES", ("mock_model.py",))
    data = tmp_path / "data.h5"
    data.write_bytes(b"the subprocess is intercepted")
    plan = launcher.build_plan(data=data, output_root=tmp_path / "fresh",
                               expected_data_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
    assert set(plan["input_snapshot"]["configs"]) == {"control", "lehami"}
    assert plan["input_snapshot"]["source_sha256"]["mock_model.py"] == launcher.sha256_file(source)
    calls = []

    def intercepted(command, **kwargs):
        calls.append(command)
        if "--execute" in command:
            # Simulate a source/config edit while control was running.
            changed = source if changed_input == "source" else paths[changed_input.removesuffix("_config")]
            changed.write_text(changed.read_text() + "\n")

    monkeypatch.setattr(launcher.subprocess, "run", intercepted)
    with pytest.raises(ValueError, match="changed after planning"):
        launcher.run_plan(plan, execute=True)
    assert len(calls) == 3  # Two read-only preflights, then only control training.
    executed = [command for command in calls if "--execute" in command]
    assert len(executed) == 1
    assert executed[0][executed[0].index("--config") + 1] == str(paths["control"])


def test_config_change_between_plan_and_preflight_is_rejected(execution_harness, monkeypatch):
    plan, calls = execution_harness
    original = launcher.snapshot_inputs
    def changed_snapshot():
        snapshot = original()
        snapshot["configs"]["lehami"]["sha256"] = "0" * 64
        return snapshot
    monkeypatch.setattr(launcher, "snapshot_inputs", changed_snapshot)
    with pytest.raises(ValueError, match="changed after planning"):
        launcher.run_plan(plan, preflight=True)
    assert calls == []
