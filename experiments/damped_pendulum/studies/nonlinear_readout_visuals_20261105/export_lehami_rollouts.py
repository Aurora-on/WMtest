#!/usr/bin/env python3
"""Export frozen final-step LeHami-style validation rollouts for fixed readouts.

Only the initial 21 observed frames enter the predictor. This script does not
fit a readout, update a model, choose a checkpoint, or access the test split.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch


def verify_checkpoint(saved, checkpoint_hash, probe, source_hashes, data_hash):
    config = saved.get("config", {})
    if (saved.get("kind") != "dissipative" or saved.get("step") != 10000
            or saved.get("validation_status") != "completed"):
        raise ValueError("Requires the completed final step10000 checkpoint")
    if (config.get("objective_mode") != "lehami"
            or config.get("core", {}).get("damping_mode") != "none"
            or config.get("environment_context_dim") != 0):
        raise ValueError("Requires the undamped LeHami-style objective arm")
    arm = probe.get("arms", {}).get("lehami", {})
    if (probe.get("status") != "completed" or arm.get("checkpoint_step") != 10000
            or arm.get("objective_mode") != "lehami"
            or arm.get("checkpoint_sha256") != checkpoint_hash):
        raise ValueError("Checkpoint differs from the frozen readout diagnostic")
    if (saved.get("source_hashes") != source_hashes
            or probe.get("source_hashes") != source_hashes):
        raise ValueError("Checkpoint, probe and loaded training sources differ")
    if (saved.get("dataset_sha256") != data_hash
            or probe.get("dataset_sha256") != data_hash):
        raise ValueError("Dataset differs from checkpoint or readout diagnostic")


def load_helper(helper_dir):
    path = Path(helper_dir).resolve() / "evaluate_control.py"
    spec = importlib.util.spec_from_file_location("frozen_visualization_helper", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper: {path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper, path


def check_prefixes(features, energy, valid):
    """Keep genuine failed suffixes; never silently replace invalid predictions."""
    if features.shape[:2] != (64, 33) or energy.shape != (64, 33) or valid.shape != (64,):
        raise ValueError("Unexpected validation trajectory shapes")
    if np.any(valid < 0) or np.any(valid > 32):
        raise ValueError("Invalid valid_steps count")
    for i, count in enumerate(valid):
        if (not np.isfinite(features[i, :count + 1]).all()
                or not np.isfinite(energy[i, :count + 1]).all()):
            raise ValueError("Nonfinite value in a declared valid trajectory prefix")
        if count < 32 and not np.isnan(features[i, count + 1:]).all():
            raise ValueError("Failed trajectory suffix must remain NaN")


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    if args.batch_size != 32:
        raise ValueError("Use the existing batch32 evaluation convention")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    helper, helper_path = load_helper(args.helper_dir)
    probe = json.loads(args.probe_summary.read_text())
    if helper.sha256(helper_path) != probe.get("helper_script_sha256"):
        raise ValueError("Helper differs from the frozen readout diagnostic")
    evaluation_hashes = helper.verify_evaluation_sources(args.code_root, args.code_manifest)
    if evaluation_hashes != probe.get("evaluation_source_hashes"):
        raise ValueError("Evaluation helpers differ from the frozen readout diagnostic")
    Model, _, autonomous, pixels_tensor, source_hashes, diagnostic_names = helper.load_frozen_helpers(args.code_root)
    sources = source_hashes()
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    data_hash = helper.sha256(args.data)
    verify_checkpoint(saved, checkpoint_hash, probe, sources, data_hash)
    model = Model(saved["config"]).to(args.device).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.history != 21 or model.dt != .1:
        raise ValueError("Protocol requires history21 and dt=.1")
    before = helper.state_hash(model)
    if before != probe["arms"]["lehami"].get("model_hash_before"):
        raise ValueError("Model state differs from frozen readout calibration")
    print(json.dumps(dict(phase="val_autonomous_K32", elapsed_seconds=time.perf_counter() - started)), flush=True)

    trajectories, truths, valid, energies, failures = [], [], [], [], []
    diagnostics = {key: [] for key in diagnostic_names}
    with h5py.File(args.data, "r") as f:
        if f.attrs.get("training_scope") != "undamped_stage_a":
            raise ValueError("Requires undamped stage-A data")
        # Deliberately do not read splits/test or any test observations/labels.
        records = sorted([dict(episode=int(ep), group_id=int(f["group_id"][ep]),
                               energy_band=int(f["energy_band"][ep]))
                          for ep in f["splits/val"][:]], key=lambda r: r["group_id"])
        if len(records) != 64 or len({r["group_id"] for r in records}) != 64:
            raise ValueError("Requires 64 unique validation groups")
        selected_records = helper.select_animation_records(records)
        for start in range(0, len(records), args.batch_size):
            chunk = records[start:start + args.batch_size]
            images, actions, intervals = [], [], []
            for record in chunk:
                offset = int(f["ep_offset"][record["episode"]])
                if int(f["ep_len"][record["episode"]]) < 53:
                    raise ValueError("Episode cannot cover anchor20 + K32")
                images.append(pixels_tensor(f["pixels"][offset:offset + 21], "cpu"))
                actions.append(f["action"][offset:offset + 52])
                intervals.append(f["dt_frame"][offset:offset + 52])
                truths.append(f["state"][offset + 20:offset + 53])
            action = torch.as_tensor(np.stack(actions), device=args.device)
            dt = torch.as_tensor(np.stack(intervals), device=args.device)
            if not torch.all(action == 0) or not torch.allclose(dt.double(), torch.full_like(dt.double(), .1)):
                raise ValueError("This diagnostic requires passive data at dt=.1")
            # No future observation is passed to the autonomous rollout API.
            out = autonomous(model, torch.stack(images).to(args.device), action[:, :20], dt[:, :20],
                             action[:, 20:52], dt[:, 20:52], diagnostic_names=diagnostic_names)
            trajectories.append(out["features"])
            energies.append(out["latent_energy"])
            valid.extend(out["valid_steps"].tolist())
            for key in diagnostic_names:
                diagnostics[key].append(out["diagnostics"][key])
            for failure in out["failures"]:
                failures.append({**failure, "evaluation_index": start + failure["batch_index"],
                                 "episode": chunk[failure["batch_index"]]["episode"]})
        selected_indices = np.array([next(i for i, r in enumerate(records) if r["episode"] == s["episode"])
                                     for s in selected_records])
        # Future RGB is exported for display only, after the autonomous calls.
        true_rgb = np.stack([f["pixels"][int(f["ep_offset"][r["episode"]]) + 20:
                                            int(f["ep_offset"][r["episode"]]) + 53]
                             for r in selected_records])

    features = np.concatenate(trajectories)
    truth = np.asarray(truths)
    valid_steps = np.asarray(valid, dtype=np.int64)
    energy = np.concatenate(energies)
    check_prefixes(features, energy, valid_steps)
    if truth.shape != (64, 33, 2) or not np.isfinite(truth).all():
        raise ValueError("Unexpected reference state array")
    after = helper.state_hash(model)
    if before != after:
        raise RuntimeError("World-model parameters changed during evaluation")
    metadata = {key: np.array([r[key] for r in records], dtype=np.int64)
                for key in ("episode", "group_id", "energy_band")}
    for record, index in zip(selected_records, selected_indices):
        record.update(evaluation_index=int(index), valid_steps=int(valid_steps[index]))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "trajectories.npz", features=features, truth=truth,
                        valid_steps=valid_steps, latent_energy=energy, time=np.arange(33) * model.dt,
                        selected_indices=selected_indices, selected_true_rgb=true_rgb, **metadata,
                        **{"diagnostic_" + key: np.concatenate(value) for key, value in diagnostics.items()})
    summary = dict(status="completed", experiment_arm="lehami", checkpoint_step=10000,
                   checkpoint_sha256=checkpoint_hash, dataset_sha256=data_hash, source_hashes=sources,
                   evaluation_source_hashes=evaluation_hashes, helper_script_sha256=helper.sha256(helper_path),
                   script_sha256=helper.sha256(__file__), code_manifest_sha256=helper.sha256(args.code_manifest),
                   probe_summary_sha256=helper.sha256(args.probe_summary),
                   code_root=str(args.code_root.resolve()), device=args.device, torch=str(torch.__version__),
                   model_hash_before=before, model_hash_after=after, parameters_unchanged=True,
                   world_model_optimizer_created=False, world_model_updates=0, readout_refitted=False,
                   test_split_accessed=False, future_observations_passed_to_predictor=False,
                   val_groups=64, anchor_frame=20, history_frames=21, horizon_steps=32, horizon_seconds=3.2,
                   complete_trajectories=int((valid_steps == 32).sum()), failures=failures,
                   selected_records=selected_records, elapsed_seconds=time.perf_counter() - started,
                   scope="Fixed validation anchor20, K32; latent features for an unchanged train-calibrated readout. No physical score is computed here.")
    helper.write_json(args.output / "summary.json", summary)
    print(json.dumps(dict(phase="completed", complete_trajectories=summary["complete_trajectories"],
                         elapsed_seconds=summary["elapsed_seconds"])), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "code-manifest", "checkpoint", "data", "output", "probe-summary", "helper-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    return run(parser.parse_args())


if __name__ == "__main__":
    main()
