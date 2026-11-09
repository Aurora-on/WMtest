#!/usr/bin/env python3
"""Measure actual dissipative optimizer updates only with explicit --execute.

This is an isolated base-learning-rate stress profile, not a training resume or
performance evaluation. Every horizon starts from the same seed-0 weights.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpbench.dissipative_data import DissipativeWindowDataset, StepBatchSampler, fixed_validation_indices
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.train_dissipative import (atomic_json, config_digest, maximum_horizon, move_batch,
                                      preflight, read_config, runtime_signature, scalar_metrics,
                                      source_hashes, validation)


HORIZONS = (4, 8, 16, 32)


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def curriculum_counts(config):
    """Count each stage once, clipping to the configured optimization budget."""
    total, previous, counts = config["training"]["steps"], 0, {}
    for stage in config["training"]["curriculum"]:
        end = min(stage["until_step"], total)
        if end > previous:
            counts[stage["horizon"]] = counts.get(stage["horizon"], 0) + end - previous
        previous = end
        if end == total:
            break
    if sum(counts.values()) != total:
        raise ValueError("Curriculum does not cover the optimization budget")
    if set(counts) - set(HORIZONS):
        raise ValueError("This profiler measures only horizons 4,8,16,32")
    return counts


def make_plan(data, config, device, warmup, steps):
    counts = curriculum_counts(config)
    cfg = config["training"]
    return {"status": "plan_only", "optimizer_updates_enabled": False,
            "data": str(Path(data).resolve()), "device": device,
            "profile_seed": 0, "horizons": list(HORIZONS),
            "warmup_updates_per_horizon": warmup, "timed_updates_per_horizon": steps,
            "total_profile_optimizer_updates": len(HORIZONS) * (warmup + steps),
            "batch_size": cfg["batch_size"], "threads": cfg["threads"],
            "learning_rate_scope": "Full configured base LR in the profile; formal training starts fresh and uses its configured warmup/cosine schedule",
            "formal_optimizer_steps": cfg["steps"], "curriculum_updates": counts,
            "formal_validation_calls": math.ceil(cfg["steps"] / cfg["val_every"]),
            "validation_windows": cfg["val_windows"], "validation_horizons": cfg["val_horizons"],
            "validation_scope": "Full trainer.validation after the horizon-32 stress run; fixed validation anchors",
            "writes_files": False, "physical_performance_evaluation": False}


def estimate_runtime(config, measurements, validation_seconds):
    counts = curriculum_counts(config)
    calls = math.ceil(config["training"]["steps"] / config["training"]["val_every"])
    def estimate(stat):
        return sum(n * measurements[str(h)][stat] for h, n in counts.items()) + calls * validation_seconds
    return {"optimizer_steps": config["training"]["steps"], "curriculum_updates": counts,
            "validation_calls": calls, "one_full_validation_seconds": validation_seconds,
            "estimated_total_seconds": estimate("mean_seconds"),
            "observed_step_min_based_seconds": estimate("min_seconds"),
            "observed_step_max_based_seconds": estimate("max_seconds"),
            "estimated_validation_seconds": calls * validation_seconds,
            "scope": "Extrapolation from short fresh-model base-LR measurements, not a completion guarantee or statistical confidence interval. Excludes queueing, transfer, data generation, checkpoint serialization and initialization/preflight. Later Newton iteration counts and convergence can change."}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _optimizer(model, config):
    cfg = config["training"]
    encoder, dynamics = list(model.encoder_parameters()), list(model.dynamics_parameters())
    ids = [id(p) for p in encoder + dynamics]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Optimizer groups must partition all trainable parameters")
    return torch.optim.AdamW([
        {"params": encoder, "lr": cfg["encoder_lr"], "base_lr": cfg["encoder_lr"], "name": "encoder"},
        {"params": dynamics, "lr": cfg["dynamics_lr"], "base_lr": cfg["dynamics_lr"], "name": "dynamics"},
    ], weight_decay=cfg["weight_decay"])


def execute_profile(data, output, config, device, warmup=2, steps=4):
    """Run actual optimizer steps; caller must authorize this execution boundary."""
    if warmup < 1 or steps < 1:
        raise ValueError("warmup and steps must be positive")
    cfg = config["training"]
    report = make_plan(data, config, str(device), warmup, steps)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusively reserve the result path. Existing evidence is never replaced.
    with output.open("x") as handle:
        json.dump({"status": "starting", "optimizer_updates_completed": 0}, handle)
    train = val = None
    started = time.perf_counter()
    phase, horizon, completed = "preflight", None, 0
    report.update(status="running", writes_files=True, optimizer_updates_enabled=True,
                  measurements={}, optimizer_updates_completed=0)
    try:
        checked = preflight(data, config)
        device = torch.device(device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Only CPU/CUDA are supported")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if maximum_horizon(config) < max(HORIZONS):
            raise ValueError("Dataset windows/configuration must cover horizon 32")
        torch.set_num_threads(cfg["threads"])
        report.update(config_sha256=config_digest(config), dataset_sha256=checked["dataset_sha256"],
                      source_sha256=source_hashes(), runtime=runtime_signature(device),
                      effective_config=config, preflight_passed=True)
        report["source_sha256"]["scripts/profile_dissipative.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        model_config = checked["resolved_model_config"]
        kwargs = dict(history=model_config["history"], max_horizon=maximum_horizon(config),
                      train_frames=cfg["train_frames"])
        train, val = (DissipativeWindowDataset(data, split, **kwargs) for split in ("train", "val"))
        sampler = StepBatchSampler(train, cfg["batch_size"], seed=0)
        anchors = fixed_validation_indices(val, cfg["val_windows"], cfg["val_seed"])
        report["fixed_validation_anchor_sha256"] = hashlib.sha256(json.dumps(anchors).encode()).hexdigest()
        initial_hash = None
        for horizon in HORIZONS:
            phase = "initialize"
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = DissipativeWorldModel(config["model"]).to(device)
            optimizer = _optimizer(model, config)
            model.train()
            actual_hash = _state_hash(model)
            if initial_hash is None:
                initial_hash = actual_hash
            elif actual_hash != initial_hash:
                raise RuntimeError("Fresh horizon models did not start with identical weights/buffers")
            before = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            seconds, losses, residual_max = [], [], 0.
            for index in range(warmup + steps):
                phase = "warmup_update" if index < warmup else "timed_update"
                _sync(device)
                tick = time.perf_counter()
                batch = move_batch(train.batch(sampler.indices(index), horizon=horizon), device)
                optimizer.zero_grad(set_to_none=True)
                values = model.loss(batch, horizon=horizon)
                metrics = scalar_metrics(values)
                values["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                _sync(device)
                elapsed = time.perf_counter() - tick
                completed += 1
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite parameters after optimizer update")
                losses.append(metrics["loss"])
                residual_max = max(residual_max, abs(metrics.get("solver_residual", 0.)))
                if index >= warmup:
                    seconds.append(elapsed)
                report["optimizer_updates_completed"] = completed
            changed = sum(not torch.equal(before[name], p.detach().cpu()) for name, p in model.named_parameters())
            if not changed or not optimizer.state:
                raise RuntimeError("Profile did not demonstrate parameter changes and nonempty optimizer state")
            measurement = {"mean_seconds": statistics.mean(seconds), "median_seconds": statistics.median(seconds),
                           "min_seconds": min(seconds), "max_seconds": max(seconds),
                           "timed_updates": steps, "warmup_updates": warmup,
                           "losses_finite": all(math.isfinite(x) for x in losses),
                           "first_loss": losses[0], "last_loss": losses[-1],
                           "solver_residual_max": residual_max, "changed_parameter_tensors": changed,
                           "parameters_actually_changed": changed > 0, "optimizer_state_entries": len(optimizer.state),
                           "initial_state_sha256": actual_hash,
                           "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                           "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
            report["measurements"][str(horizon)] = measurement
            atomic_json(output, report)
            print(json.dumps({"horizon": horizon, **measurement}), flush=True)
            if horizon == HORIZONS[-1]:
                phase = "full_validation"
                _sync(device)
                tick = time.perf_counter()
                validation_report = validation(model, val, anchors, config, device)
                _sync(device)
                validation_seconds = time.perf_counter() - tick
                report["validation"] = {"seconds": validation_seconds, "windows": len(anchors),
                                         "horizons": cfg["val_horizons"], "groups": len(validation_report["groups"]),
                                         "gate_passed": validation_report["gate_passed"], "score": validation_report["score"],
                                         "scope": "After the fresh horizon-32 base-LR stress updates; internal latent validation, not physical accuracy"}
                report["runtime_estimate"] = estimate_runtime(config, report["measurements"], validation_seconds)
            del model, optimizer, before, batch, values
            gc.collect()
        report.update(status="completed", total_profile_wall_seconds=time.perf_counter() - started,
                      formal_training_started=False, checkpoint_saved=False,
                      timing_scope="Each timed interval includes data batch read/transfer, forward, scalar checks, backward, gradient clipping and AdamW update; CUDA synchronized at both ends. Model creation and parameter-copy audit excluded.")
        atomic_json(output, report)
        return report
    except BaseException as exc:
        report.update(status="failed", phase=phase, horizon=horizon, optimizer_updates_completed=completed,
                      error_type=type(exc).__name__, error=str(exc), elapsed_seconds=time.perf_counter() - started,
                      formal_training_started=False, checkpoint_saved=False)
        atomic_json(output, report)
        raise
    finally:
        for dataset in (train, val):
            if dataset is not None:
                dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=positive_int, default=2)
    parser.add_argument("--steps", type=positive_int, default=4)
    parser.add_argument("--threads", type=positive_int)
    parser.add_argument("--execute", action="store_true", help="Explicitly enable isolated optimizer updates")
    args = parser.parse_args(argv)
    config = copy.deepcopy(read_config(args.config))
    if args.threads is not None:
        config["training"]["threads"] = args.threads
    if args.execute:
        result = execute_profile(args.data, args.output, config, args.device, args.warmup, args.steps)
        print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                          "runtime_estimate": result["runtime_estimate"]}, indent=2))
    else:
        result = make_plan(args.data, config, args.device, args.warmup, args.steps)
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
