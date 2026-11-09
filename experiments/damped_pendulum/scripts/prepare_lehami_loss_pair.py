#!/usr/bin/env python3
"""Plan the paired objective experiment; --execute is required for optimization.

The default path uses only the Python standard library, reads configuration and
prints a plan. It does not import torch, inspect test samples, create directories,
submit jobs or run commands. --preflight checks both train/val data contracts.
--execute runs the selected arms sequentially, fresh, with the full 20k schedule
and a separately specified stopping point (4k by default).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies/lehami_loss_pair_20261105"
CONFIGS = {arm: ROOT / f"configs/dissipative_loss_pair_{arm}_v1.json"
           for arm in ("control", "lehami")}
DATA_SHA256 = "51cebf0876b1291abf1aa0f9ccee52a9dcf1cbed1840b51df173f55e35969b24"
DATA_PATH = "/home/weizhixin/ssd/undamped_stage_a_launch_20261105T053810Z/data/undamped_v1/undamped.h5"
STOP_POINTS = (250, 1000, 4000, 10000, 20000)
# The trainer's source_hashes() set, plus this orchestration entry point. Keep
# this standard-library-only so planning never imports a model or optimizer.
SOURCE_FILES = (
    "dpbench/train_dissipative.py", "dpbench/dissipative_model.py",
    "dpbench/dissipative_core.py", "dpbench/dissipative_data.py",
    "dpbench/efficient_dissipative_core.py", "dpbench/automatic_discrete_gradient.py",
    "third_party/le-wm/module.py", "scripts/prepare_lehami_loss_pair.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_inputs():
    return dict(source_sha256={name: sha256_file(ROOT / name) for name in SOURCE_FILES},
                configs={arm: dict(path=str(path), sha256=sha256_file(path))
                         for arm, path in CONFIGS.items()})


def verify_plan_inputs(plan):
    """Refuse source or either-config drift, including between sequential arms."""
    actual = snapshot_inputs()
    if actual != plan["input_snapshot"]:
        raise ValueError("Paired source/config inputs changed after planning; stop before the next arm")


def check_pair_contract(configs):
    """Fail before execution if anything except the declared objective differs."""
    if set(configs) != {"control", "lehami"}:
        raise ValueError("Both paired configurations are required")
    stripped = {}
    for arm, config in configs.items():
        model, training = config["model"], config["training"]
        expected = dict(objective_mode="current" if arm == "control" else "lehami",
                        motion_weight=1.0 if arm == "control" else 0.0,
                        momentum_weight=0.1 if arm == "control" else 1.0)
        if any(model.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{arm}: objective weights/mode differ from this protocol")
        if (model.get("target_tail_frames") != 2 or model.get("momentum_reference_time") != 0.1
                or model.get("sigreg_weight") != 0.25 or model.get("initializer_weight") != 0.001
                or model["core"].get("substeps") != 3
                or model["core"].get("damping_mode") != "none"
                or model["core"].get("potential_mode") != "raw"
                or model.get("environment_context_dim") != 0):
            raise ValueError(f"{arm}: model is not the declared undamped three-substep pair")
        if (training.get("validation_score") != "position_momentum"
                or training.get("validation_motion_weight") != 0
                or training.get("steps") != 20000 or training.get("seed") != 0
                or training.get("required_renderer") != "smooth_v1"
                or training.get("required_data_scope") != "undamped_stage_a"):
            raise ValueError(f"{arm}: training/selection contract differs")
        copy = json.loads(json.dumps(config))
        for key in expected:
            copy["model"].pop(key)
        stripped[arm] = copy
    if stripped["control"] != stripped["lehami"]:
        raise ValueError("Paired settings differ outside the declared objective changes")


def build_plan(data=DATA_PATH, output_root=None, python=sys.executable,
               device="cuda", arm="both", stop_after_step=4000,
               expected_data_sha256=DATA_SHA256):
    if arm not in ("both", "control", "lehami") or stop_after_step not in STOP_POINTS:
        raise ValueError("Unknown arm or stopping point")
    if device not in ("cpu", "cuda"):
        raise ValueError("Only CPU/CUDA are supported")
    if len(expected_data_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_data_sha256):
        raise ValueError("Expected data SHA256 must be 64 lowercase hexadecimal characters")
    configs = {name: json.loads(path.read_text()) for name, path in CONFIGS.items()}
    check_pair_contract(configs)
    output = Path(output_root or ROOT / "runs/lehami_loss_pair_20261105").resolve()
    data = str(Path(data).expanduser().resolve())
    selected = list(CONFIGS) if arm == "both" else [arm]
    rows = []
    for name in selected:
        command = [str(python), "-m", "dpbench.train_dissipative", "--config", str(CONFIGS[name]),
                   "--data", data]
        rows.append(dict(arm=name, config=str(CONFIGS[name]),
                         config_file_sha256=sha256_file(CONFIGS[name]),
                         output=str(output / name), preflight_command=command,
                         training_command=command + ["--output", str(output / name),
                             "--device", device, "--stop-after-step", str(stop_after_step), "--execute"]))
    return dict(mode="plan_only", training_started=False, optimizer_created=False,
                remote_submitted=False, initial_step=0, resume=None,
                configured_total_steps=20000, stop_after_step=stop_after_step,
                selected_arms=selected, code_directory=str(ROOT), data=data,
                path_scope="Commands are local to code_directory; regenerate the plan after deploying or relocating the frozen bundle",
                expected_data_sha256=expected_data_sha256,
                reference_data_sha256=DATA_SHA256,
                data_override=expected_data_sha256 != DATA_SHA256,
                dataset_locally_verified=False, historical_data_record_not_live_status=True,
                execution="Sequential arms; --arm supports separate GPU allocation by the caller",
                output_root=str(output), arms=rows, input_snapshot=snapshot_inputs(),
                scientific_scope="Single seed, bundled loss migration on undamped data; not an ablation of each individual term or proof of dissipative learning",
                comparison_rule="Both arms start fresh. Historical stage-A training is not the paired control.")


def run_plan(plan, preflight=False, execute=False):
    if preflight and execute:
        raise ValueError("Choose either preflight or execute")
    if not (preflight or execute):
        return plan
    verify_plan_inputs(plan)
    data = Path(plan["data"])
    if not data.is_file():
        raise FileNotFoundError(f"Dataset is unavailable on this host: {data}")
    actual = sha256_file(data)
    if actual != plan["expected_data_sha256"]:
        raise ValueError("Dataset SHA256 differs from the expected paired input")
    if execute:
        # Check ALL destinations before either arm can update parameters.
        for row in plan["arms"]:
            if Path(row["output"]).exists():
                raise FileExistsError(f"Fresh pair refuses existing arm output: {row['output']}")
    # Validate BOTH configs even when a caller schedules only one arm.
    for name, config in CONFIGS.items():
        verify_plan_inputs(plan)
        command = [plan["arms"][0]["preflight_command"][0], "-m", "dpbench.train_dissipative",
                   "--config", str(config), "--data", plan["data"]]
        subprocess.run(command, cwd=ROOT, check=True)
    result = dict(plan, mode="data_preflight" if not execute else "executing",
                  dataset_locally_verified=True, actual_data_sha256=actual)
    if not execute:
        return result
    for row in plan["arms"]:
        # The trainer creates its fresh output and stores config/data/source/RNG
        # provenance; a failed arm prevents the sequential second arm starting.
        verify_plan_inputs(plan)
        subprocess.run(row["training_command"], cwd=ROOT, check=True)
    return dict(result, mode="completed_requested_segment", training_started=True,
                optimizer_created=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--expected-data-sha256", default=DATA_SHA256)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--arm", choices=("both", "control", "lehami"), default="both")
    parser.add_argument("--stop-after-step", type=int, choices=STOP_POINTS, default=4000)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(data=args.data, output_root=args.output_root, python=args.python,
                      device=args.device, arm=args.arm, stop_after_step=args.stop_after_step,
                      expected_data_sha256=args.expected_data_sha256)
    result = run_plan(plan, preflight=args.preflight, execute=args.execute)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
