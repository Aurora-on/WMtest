#!/usr/bin/env python3
"""Dissipative H-JEPA training, with physical acceptance at 4k and 20k.

The default is a read-only preflight. Staged training starts with a fresh A run,
then transfers its qualified 20k weights to independent single/multi branches.
The optional scratch route still starts each branch from random weights.
The upstream dissipative predictor and losses are unchanged. The 20k schedule
is fixed from the outset, including when execution stops for the 4k review.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dpbench.comparison_hjepa import default_hjepa_config
from dpbench.comparison_protocol import digest, json_digest, preflight, source_manifest, validate_config
from dpbench.damping_stage_data import validate_pair
from dpbench.evaluate_comparison import evaluate_checkpoint
from dpbench.hjepa_acceptance import qualify_evaluation
from dpbench.train_comparison import execute_run, validate_saved
from dpbench.train_dissipative import atomic_json, runtime_signature
from scripts.run_comparison import verify_evaluation

MILESTONES = (4000, 20000)
EXTRA_SOURCES = (
    "scripts/run_hjepa_retrain.py", "dpbench/hjepa_acceptance.py",
    "scripts/train_comparison.py", "scripts/evaluate_comparison.py", "scripts/run_comparison.py",
)


def validate_retrain_config(config):
    validate_config(config)
    if ((config["training_route"] == "scratch" and config["data"]["stage"] == "a")
            or config["fixture_only"] or config["model"] != default_hjepa_config()):
        raise ValueError("Require the unchanged full H-JEPA adapter; A uses staged, branches use staged or scratch")
    expected = {"steps": 20000, "batch_size": 128, "lr": 5e-5, "weight_decay": 1e-3,
                "curriculum": [{"until_step": 20000, "horizon": 5}]}
    for key, value in expected.items():
        if config["training"][key] != value:
            raise ValueError(f"H-JEPA retraining requires training.{key}={value!r}")
    return config


def extra_source_manifest():
    return {name: digest(ROOT / name) for name in EXTRA_SOURCES}


def prepare(config, data, data_a, device, parent_output=None):
    """Read-only scientific/data/source checks; no model, optimizer or output."""
    validate_retrain_config(config)
    stage = config["data"]["stage"]
    staged_branch = config["training_route"] == "staged" and stage != "a"
    if staged_branch != (parent_output is not None):
        raise ValueError("Exactly staged B requires --parent-output; A and scratch forbid it")
    data, data_a = Path(data).resolve(), Path(data_a).resolve()
    if stage == "a" and data != data_a:
        raise ValueError("Stage A must train on the supplied undamped --data-a path")
    report = preflight(data, config)
    paired = ({"status": "undamped_stage_a", "same_dataset": True} if stage == "a" else
              validate_pair(data_a, data, multi=stage == "multi"))
    manifest = {
        "schema_version": 2, "kind": "hjepa_dissipative_retraining",
        "config": config, "protocol_sha256": json_digest(config),
        "data": str(Path(data).resolve()), "data_sha256": report["dataset_sha256"],
        "data_a": str(data_a), "data_a_sha256": digest(data_a),
        "parent_output": str(Path(parent_output).resolve()) if staged_branch else None,
        "paired": paired, "sources": source_manifest(), "extra_sources": extra_source_manifest(),
        "device": device, "milestones": list(MILESTONES),
        "dissipation": "Official learned PSD dissipation active throughout; no disabling or freezing",
        "initialization": ("Qualified current-run A20k weights, including dissipation; fresh optimizer/RNG"
                           if staged_branch else "Random weights; no pretrained weights"),
        "evaluation": {"split": "val", "horizons": [8, 16, 32], "batch_size": 32,
                       "readout": "Own checkpoint; train288 fit, train96 selection; no test evaluation"},
    }
    return manifest, report


def _expected_parent_config(config):
    parent = copy.deepcopy(config)
    parent["training_route"] = "staged"
    parent["data"].update(stage="a", selected_damping=[0.])
    return parent


def _resolve_parent(config, manifest, report):
    """Read-only verification before a B optimizer or training output is created."""
    if manifest["parent_output"] is None:
        return None
    root = Path(manifest["parent_output"])
    parent_config = _expected_parent_config(config)
    expected_manifest, parent_report = prepare(parent_config, manifest["data_a"], manifest["data_a"], manifest["device"])
    marker = root / "pipeline.json"
    if json.loads(marker.read_text()) != expected_manifest:
        raise ValueError("Parent A pipeline must use this run's config, sources, wrapper and acceptance code")
    status = json.loads((root / "status.json").read_text())
    if (status.get("status") not in {"completed", "already_beyond_requested_milestone"}
            or status.get("qualified") is not True or status.get("completed_steps") != 20000):
        raise ValueError("Parent A must have completed and qualified at step20000")
    checkpoint = root / "milestones" / "step_020000.pt"
    saved = _load_checkpoint(checkpoint, parent_config, parent_report, expected_manifest)
    if (saved["step"] != 20000 or saved.get("validation_status") != "completed"
            or not isinstance(saved.get("validation"), dict)
            or not math.isfinite(saved["validation"].get("score", float("nan")))
            or saved.get("parent_config_sha256") != report.get("parent_config_sha256")):
        raise ValueError("Parent A checkpoint is not the matching validated20k endpoint")
    sha = digest(checkpoint)
    sidecar = {"step": 20000, "sha256": sha, "file": checkpoint.name,
               "dataset_sha256": manifest["data_a_sha256"], "protocol_sha256": json_digest(parent_config)}
    if json.loads(checkpoint.with_suffix(".json").read_text()) != sidecar:
        raise ValueError("Parent A milestone checksum/identity differs")
    evaluation = root / "evaluations" / "step_020000"
    verify_evaluation(evaluation, sha, manifest["data_a_sha256"], json_digest(parent_config), 20000)
    # Even a later read-only review of the 4k milestone cannot substitute for
    # rechecking this exact 20k physical evaluation and its acceptance record.
    acceptance = qualify_evaluation(evaluation)
    if acceptance.get("qualified") is not True:
        raise ValueError("Parent A fails current physical acceptance; B cannot start")
    if json.loads((root / "acceptance" / "step_020000.json").read_text()) != acceptance:
        raise ValueError("Parent A stored acceptance differs from rechecked evaluation")
    return {
        "checkpoint": str(checkpoint), "sha256": sha, "step": 20000,
        "optimizer_reset": True, "all_parameters_trainable": True,
        "independently_verified_data_sha256": manifest["data_a_sha256"],
        "parent_pipeline_sha256": digest(marker),
        "evaluation_complete_sha256": digest(evaluation / "complete.json"),
        "acceptance_sha256": digest(root / "acceptance" / "step_020000.json"),
    }


def _transfer_identity(parent):
    return {key: parent[key] for key in ("checkpoint", "sha256", "step", "optimizer_reset",
            "all_parameters_trainable", "independently_verified_data_sha256")}


def _load_checkpoint(path, config, report, manifest):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    validate_saved(saved, config, report, manifest["sources"], runtime_signature(manifest["device"]))
    parent = manifest.get("resolved_parent")
    expected_transfer = _transfer_identity(parent) if parent else None
    if saved.get("model_config") != config["model"] or saved.get("parent_transfer") != expected_transfer:
        raise ValueError("Checkpoint architecture or initialization provenance differs")
    return saved


def _seal_checkpoint(source, destination, step, config, report, manifest):
    """Atomically copy exact milestone bytes; never replace a conflicting copy."""
    saved = _load_checkpoint(source, config, report, manifest)
    if saved["step"] != step or saved.get("validation_status") != "completed":
        raise ValueError("Milestone requires its exact step and completed latent validation")
    checksum = digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if digest(destination) != checksum:
            raise ValueError(f"Refusing to overwrite different milestone bytes: {destination}")
    else:
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            shutil.copyfile(source, temporary)
            if digest(temporary) != checksum:
                raise ValueError("Milestone copy checksum mismatch")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    info = {"step": step, "sha256": checksum, "file": destination.name,
            "dataset_sha256": report["dataset_sha256"], "protocol_sha256": json_digest(config)}
    marker = destination.with_suffix(".json")
    if marker.exists() and json.loads(marker.read_text()) != info:
        raise ValueError("Existing milestone manifest differs")
    if not marker.exists():
        atomic_json(marker, info)
    return info


def _verify_milestone(path, step, config, report, manifest):
    saved = _load_checkpoint(path, config, report, manifest)
    if saved["step"] != step or saved.get("validation_status") != "completed":
        raise ValueError("Existing milestone is not the validated exact step")
    # Also verifies the sidecar, recovering it if interrupted immediately after copy.
    return _seal_checkpoint(path, path, step, config, report, manifest)


def _evaluate_and_qualify(root, checkpoint, info, config, report, manifest):
    step = info["step"]
    destination = root / "evaluations" / f"step_{step:06d}"
    identity = (info["sha256"], report["dataset_sha256"], json_digest(config), step)
    if destination.exists():
        verify_evaluation(destination, *identity)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        attempt = root / "evaluation_attempts" / f"step_{step:06d}_{uuid.uuid4().hex}"
        attempt.parent.mkdir(parents=True, exist_ok=True)
        evaluate_checkpoint(checkpoint, manifest["data"], attempt, device=manifest["device"],
                            batch_size=32, horizons=(8, 16, 32))
        verify_evaluation(attempt, *identity)
        os.replace(attempt, destination)
    result = qualify_evaluation(destination)
    if type(result.get("qualified")) is not bool:
        raise ValueError("Acceptance report lacks a boolean qualified result")
    acceptance = root / "acceptance" / f"step_{step:06d}.json"
    acceptance.parent.mkdir(parents=True, exist_ok=True)
    if acceptance.exists() and json.loads(acceptance.read_text()) != result:
        raise ValueError("Previously recorded acceptance differs; inspect instead of replacing it")
    if not acceptance.exists():
        atomic_json(acceptance, result)
    return result


def execute(output, config, manifest, report, until_step=20000):
    if until_step not in MILESTONES:
        raise ValueError("until_step must be 4000 or 20000")
    manifest = copy.deepcopy(manifest)
    parent = _resolve_parent(config, manifest, report)
    if parent is not None:
        manifest["resolved_parent"] = parent
    root = Path(output).resolve()
    if manifest["parent_output"] and (root == Path(manifest["parent_output"])
            or root.is_relative_to(Path(manifest["parent_output"]))):
        raise ValueError("B output must be separate from its immutable A output")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".pipeline.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = root / "pipeline.json"
        if marker.exists():
            if json.loads(marker.read_text()) != manifest:
                raise ValueError("Existing pipeline config/data/source identity differs")
        elif any(p.name != ".pipeline.lock" for p in root.iterdir()):
            raise FileExistsError("Refusing to mix unrecognized output artifacts")
        else:
            atomic_json(marker, manifest)
        run = root / "run"
        latest = run / "last.pt"
        try:
            current = _load_checkpoint(latest, config, report, manifest)["step"] if latest.exists() else 0
            for step in MILESTONES:
                if step > until_step:
                    break
                checkpoint = root / "milestones" / f"step_{step:06d}.pt"
                if current < step:
                    if checkpoint.exists():
                        raise ValueError("Milestone is ahead of latest training state; refusing a rollback")
                    atomic_json(root / "status.json", {"status": "training", "target_milestone": step,
                                                       "completed_steps": current, "total_steps": 20000})
                    transfer = {}
                    if parent is not None and not latest.exists():
                        if digest(parent["checkpoint"]) != parent["sha256"]:
                            raise ValueError("Parent checkpoint changed before transfer")
                        transfer = dict(init_from=Path(parent["checkpoint"]), expected_parent_step=20000,
                                        expected_parent_data_sha256=manifest["data_a_sha256"])
                    execute_run(manifest["data"], run, config, device=manifest["device"],
                                resume=latest if latest.exists() else None, stop_after_step=step, **transfer)
                    current = _load_checkpoint(latest, config, report, manifest)["step"]
                    if current != step:
                        raise ValueError("Trainer did not stop at the requested milestone")
                if current == step:
                    info = _seal_checkpoint(latest, checkpoint, step, config, report, manifest)
                elif checkpoint.exists():
                    info = _verify_milestone(checkpoint, step, config, report, manifest)
                else:
                    raise ValueError("Training progressed beyond an unpreserved milestone; cannot reconstruct it")
                atomic_json(root / "status.json", {"status": "evaluating", "milestone": step,
                                                   "completed_steps": current, "total_steps": 20000})
                acceptance = _evaluate_and_qualify(root, checkpoint, info, config, report, manifest)
                if not acceptance["qualified"]:
                    result = {"status": "not_qualified", "milestone": step, "qualified": False,
                              "completed_steps": current, "total_steps": 20000,
                              "continued_after_failed_acceptance": False}
                    atomic_json(root / "status.json", result)
                    return result
            status = ("completed" if until_step == 20000 else
                      "already_beyond_requested_milestone" if current > until_step else "qualified_at_4000")
            result = {"status": status, "qualified": True, "completed_steps": current,
                      "requested_milestone": until_step, "total_steps": 20000}
            atomic_json(root / "status.json", result)
            return result
        except BaseException as exc:
            atomic_json(root / "status.json", {"status": "failed", "error_type": type(exc).__name__,
                                               "error": str(exc), "total_steps": 20000})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-a", type=Path, required=True, help="Undamped data for A training or B pairing/parent audit")
    parser.add_argument("--parent-output", type=Path, help="Required only for staged B; qualified current A pipeline output")
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-parent", action="store_true",
                      help="Read-only staged B check of the completed A20k parent before GPU profiling")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--until-step", choices=MILESTONES, type=int, default=20000)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    manifest, report = prepare(config, args.data, args.data_a, args.device, args.parent_output)
    if not args.execute:
        if args.verify_parent and manifest["parent_output"] is None:
            parser.error("--verify-parent is only available for staged B with --parent-output")
        parent = _resolve_parent(config, manifest, report) if args.verify_parent else None
        print(json.dumps({"status": "preflight_passed", "training_started": False,
                          "optimizer_created": False, "until_step": args.until_step,
                          "parent_verified": parent is not None, "resolved_parent": parent,
                          "manifest": manifest, "preflight": report}, indent=2))
        return 0
    result = execute(args.output, config, manifest, report, args.until_step)
    print(json.dumps(result, indent=2))
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
