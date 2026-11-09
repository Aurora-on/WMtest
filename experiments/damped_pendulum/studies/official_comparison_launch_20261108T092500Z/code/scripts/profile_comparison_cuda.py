#!/usr/bin/env python3
"""Isolated CUDA timing on real comparison data; never writes training checkpoints.

Default coverage is A at K=4/8/16/32, single/multi at K=32, and the full
288-window K=8/32 latent validation for every stage. Use --all-stage-horizons
to measure every curriculum horizon on every dataset. The JSON is updated
after each measured step, so an OOM leaves an explicit failed record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from dpbench.comparison_models import build_comparison_model
from dpbench.comparison_protocol import digest, make_config, preflight, source_manifest
from dpbench.damping_stage_data import validate_pair
from dpbench.dissipative_data import (
    DissipativeWindowDataset, StepBatchSampler, fixed_validation_indices,
)
from dpbench.train_comparison import _seed, scalar_losses, validation
from dpbench.train_dissipative import atomic_json, runtime_signature


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def statistics_for(rows, key):
    values = [r[key] for r in rows]
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def timed_train_step(model, optimizer, dataset, sampler, index, horizon, config, device):
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    ids = sampler.indices(index)
    before_data = time.perf_counter()
    batch = dataset.batch(ids, horizon)
    data_seconds = time.perf_counter() - before_data
    before_transfer = time.perf_counter()
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    torch.cuda.synchronize(device)
    transfer_seconds = time.perf_counter() - before_transfer
    before_compute = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses = model.loss(batch, horizon, stage="train")
    values = scalar_losses(losses)
    losses["loss"].backward()
    gradient = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize(device)
    compute_seconds = time.perf_counter() - before_compute
    norm = float(gradient)
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient norm")
    return {"sample_step": index, "horizon": horizon, "losses": values,
            "gradient_norm_before_clip": norm, "data_seconds": data_seconds,
            "transfer_seconds": transfer_seconds, "compute_seconds": compute_seconds,
            "train_seconds": time.perf_counter() - started}


def estimate(stages):
    """Compute staged single-GPU and two-GPU estimates with explicit unknowns."""
    counts = {4: 1000, 8: 3000, 16: 6000, 32: 10000}
    a_times = {int(k): v["statistics"]["train_seconds"]["mean"]
               for k, v in stages["a"]["horizons"].items()}
    output = {}
    for stage, result in stages.items():
        measured = {int(k): v["statistics"]["train_seconds"]["mean"]
                    for k, v in result["horizons"].items()}
        scale = measured[32] / a_times[32]
        times = {k: measured.get(k, a_times[k] * scale) for k in counts}
        train_seconds = sum(counts[k] * times[k] for k in counts)
        validation_seconds = result["validation"]["elapsed_seconds"] * 80
        output[stage] = {
            "step_seconds_by_horizon": {str(k): v for k, v in times.items()},
            "measured_horizons": sorted(measured),
            "estimated_horizons": sorted(set(counts) - set(measured)),
            "unmeasured_horizon_rule": "A curve multiplied by this stage K32 / A K32",
            "optimizer_updates": 20000, "validation_calls": 80,
            "training_seconds": train_seconds,
            "latent_validation_seconds": validation_seconds,
            "subtotal_seconds": train_seconds + validation_seconds,
        }
    sequential = sum(item["subtotal_seconds"] for item in output.values())
    parallel = output["a"]["subtotal_seconds"] + max(
        output["single"]["subtotal_seconds"], output["multi"]["subtotal_seconds"])
    return {
        "stages": output,
        "single_gpu_sequential_subtotal_seconds": sequential,
        "single_gpu_sequential_subtotal_hours": sequential / 3600,
        "two_gpu_parallel_b_subtotal_seconds": parallel,
        "two_gpu_parallel_b_subtotal_hours": parallel / 3600,
        "formula_per_stage": "1000*t4 + 3000*t8 + 6000*t16 + 10000*t32 + 80*validation288_K8_K32",
        "excluded": ["queue time", "preflight/startup", "checkpoint and status file writes",
                     "three final physical readout evaluations", "contention and later-model numerical changes"],
        "not_a_completion_deadline": True,
        "interpretation": "Measured early-model compute/data estimate; add excluded costs and refine using formal history.jsonl",
    }


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new profile report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {
        "schema_version": 1, "status": "running", "method": args.method,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": digest(Path(__file__)), "warmup_steps_per_horizon": args.warmup,
        "measured_steps_per_horizon": args.measured, "stages": {},
        "scope": "Isolated fresh disposable model and optimizer per stage; no formal checkpoints or optimizer states read/written",
        "model_initialization": "Random seed0 on each stage; timings do not assess trained-model quality",
        "precision": "float32 as formal trainer; no AMP or compilation",
        "test_pixels_or_targets_read": False,
    }

    def publish(phase):
        report["phase"] = phase
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(output, report)
        print(f"{args.method}: {phase} ({report['elapsed_seconds']:.1f}s)", flush=True)

    publish("initializing")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; profiling never falls back to CPU")
        # Index 0 is relative to the Slurm-provided CUDA_VISIBLE_DEVICES.
        device = torch.device(args.device, 0)
        torch.cuda.set_device(device)
        torch.set_num_threads(4)
        report["runtime"] = runtime_signature(device)
        report["cuda_device"] = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
        report["sources"] = source_manifest()
        paths = {stage: getattr(args, "data_" + stage).resolve()
                 for stage in ("a", "single", "multi")}
        report["paired"] = {
            stage: validate_pair(paths["a"], paths[stage], multi=stage == "multi")
            for stage in ("single", "multi")}
        publish("paired_data_verified")
        for stage, path in paths.items():
            config = make_config(args.method, stage=stage, seed=0)
            result = {"config": config, "preflight": preflight(path, config), "horizons": {}}
            report["stages"][stage] = result
            train = val = model = optimizer = None
            try:
                _seed(0)
                model = build_comparison_model(config["model"], device)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                                              weight_decay=config["training"]["weight_decay"])
                result["parameters"] = sum(p.numel() for p in model.parameters())
                train = DissipativeWindowDataset(path, "train", 21, 32, 81, 2)
                val = DissipativeWindowDataset(path, "val", 21, 32, 81, 2)
                sampler = StepBatchSampler(train, 32, 0)
                horizons = (4, 8, 16, 32) if stage == "a" or args.all_stage_horizons else (32,)
                sample_step = 0
                for horizon in horizons:
                    rows = []
                    block = {"warmup_steps": args.warmup, "measured_steps": args.measured,
                             "batch_size": 32, "rows": rows}
                    result["horizons"][str(horizon)] = block
                    torch.cuda.reset_peak_memory_stats(device)
                    publish(f"{stage}_K{horizon}_warmup")
                    for _ in range(args.warmup):
                        timed_train_step(model, optimizer, train, sampler, sample_step, horizon, config, device)
                        sample_step += 1
                    for index in range(args.measured):
                        rows.append(timed_train_step(model, optimizer, train, sampler, sample_step,
                                                     horizon, config, device))
                        sample_step += 1
                        block["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                        block["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                        publish(f"{stage}_K{horizon}_measured_{index + 1}")
                    block["statistics"] = {key: statistics_for(rows, key) for key in
                                           ("train_seconds", "data_seconds", "transfer_seconds", "compute_seconds")}
                    block["status"] = "passed"
                indices = fixed_validation_indices(val, config["training"]["val_windows"],
                                                   config["training"]["val_seed"])
                publish(f"{stage}_validation288_K8_K32")
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                before = time.perf_counter()
                metrics = validation(model, val, indices, config, device)
                torch.cuda.synchronize(device)
                result["validation"] = {
                    "elapsed_seconds": time.perf_counter() - before, "windows": len(indices),
                    "horizons": [8, 32], "metrics": metrics,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    "scope": "Full formal latent validation, including permitted HDF5 reads; not final physical kNN evaluation",
                }
                result["status"] = "passed"
                publish(f"{stage}_passed")
            finally:
                if train is not None:
                    train.close()
                if val is not None:
                    val.close()
                del optimizer, model, train, val
                gc.collect()
                torch.cuda.empty_cache()
        report["estimate"] = estimate(report["stages"])
        report["status"] = "completed"
        publish("completed")
    except BaseException as exc:
        report["status"] = "failed"
        report["failed_phase"] = report.get("phase")
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
        publish("failed")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("lehami", "lewm", "hjepa"))
    for stage in ("a", "single", "multi"):
        parser.add_argument("--data-" + stage, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--warmup", type=positive, default=3)
    parser.add_argument("--measured", type=positive, default=10)
    parser.add_argument("--all-stage-horizons", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
