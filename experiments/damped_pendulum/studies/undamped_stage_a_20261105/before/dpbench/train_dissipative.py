"""Train the dissipative JEPA model only when --execute is explicitly supplied.

The default command is a read-only data/configuration preflight: it does not
construct an optimizer, update parameters, or create an output directory.
No server submission or subprocess training is performed by this entry point.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import time
import uuid
from pathlib import Path

import h5py
import numpy as np
import torch

from dpbench.dissipative_data import (
    DissipativeWindowDataset, MODEL_FIELDS, StepBatchSampler,
    fixed_validation_indices, preflight_dataset, window_stratum, stratum_label,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/dissipative_jepa.json"
CHECKPOINT_VERSION = 1


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_config(config):
    """Validate the training contract without constructing a model or optimizer."""
    if set(config) != {"model", "training"} or not isinstance(config["model"], dict):
        raise ValueError("Config must contain exactly model and training objects")
    cfg = config["training"]
    expected = {
        "steps", "batch_size", "train_frames", "seed", "threads", "encoder_lr",
        "dynamics_lr", "weight_decay", "warmup_steps", "minimum_lr_ratio",
        "gradient_clip", "curriculum", "val_horizons", "val_every", "val_windows",
        "val_seed", "minimum_target_std", "minimum_target_motion",
    }
    optional = {"validation_gate_time_bins", "prediction_motion_gate",
                "validation_motion_weight", "required_renderer"}
    if not expected <= set(cfg) or set(cfg) - expected - optional:
        raise ValueError(f"Training config keys differ: {sorted(set(cfg) ^ expected)}")
    if "required_renderer" in cfg and (not isinstance(cfg["required_renderer"], str)
                                      or cfg["required_renderer"] not in {"legacy", "smooth_v1"}):
        raise ValueError("required_renderer must be legacy or smooth_v1")
    motion_weight = cfg.get("validation_motion_weight", 0.0)
    if (isinstance(motion_weight, bool) or not isinstance(motion_weight, (int, float))
            or not math.isfinite(motion_weight) or motion_weight < 0):
        raise ValueError("validation_motion_weight must be finite and nonnegative")
    gate = cfg.get("prediction_motion_gate")
    if gate is not None:
        keys = {"min_rms_ratio", "max_rms_ratio", "max_relative_error",
                "min_cosine", "max_stalled_fraction"}
        if not isinstance(gate, dict) or set(gate) != keys:
            raise ValueError("prediction_motion_gate requires exactly " + str(sorted(keys)))
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in gate.values()):
            raise ValueError("prediction_motion_gate values must be finite numbers")
        if not (0 < gate["min_rms_ratio"] <= 1 <= gate["max_rms_ratio"]
                and 0 <= gate["max_relative_error"] < 1
                and 0 <= gate["min_cosine"] <= 1
                and 0 <= gate["max_stalled_fraction"] < 1):
            raise ValueError("Invalid prediction_motion_gate bounds")
    gate_bins = cfg.get("validation_gate_time_bins")
    if gate_bins is not None and (not isinstance(gate_bins, list) or not gate_bins
                                 or len(set(gate_bins)) != len(gate_bins)
                                 or any(type(x) is not int or x not in (0, 1, 2) for x in gate_bins)):
        raise ValueError("validation_gate_time_bins must contain unique time bins 0, 1 or 2")
    for key in ("steps", "batch_size", "train_frames", "threads", "val_every", "val_windows"):
        _positive_int(cfg[key], key)
    if cfg["batch_size"] < 2:
        raise ValueError("SIGReg training requires batch_size >= 2")
    for key in ("seed", "val_seed", "warmup_steps"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("encoder_lr", "dynamics_lr", "gradient_clip", "minimum_target_std", "minimum_target_motion"):
        if isinstance(cfg[key], bool) or not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(cfg["weight_decay"]) or cfg["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 <= cfg["minimum_lr_ratio"] <= 1:
        raise ValueError("minimum_lr_ratio must be in [0, 1]")
    if not isinstance(cfg["curriculum"], list) or not cfg["curriculum"]:
        raise ValueError("curriculum must be a nonempty list")
    previous = 0
    for stage in cfg["curriculum"]:
        if set(stage) != {"until_step", "horizon"}:
            raise ValueError("Each curriculum stage needs until_step and horizon")
        _positive_int(stage["until_step"], "until_step")
        _positive_int(stage["horizon"], "horizon")
        if stage["until_step"] <= previous:
            raise ValueError("Curriculum boundaries must strictly increase")
        previous = stage["until_step"]
    if previous < cfg["steps"]:
        raise ValueError("Curriculum must cover all optimizer steps")
    horizons = cfg["val_horizons"]
    if not isinstance(horizons, list) or not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("val_horizons must be nonempty and unique")
    for horizon in horizons:
        _positive_int(horizon, "val_horizon")
    history = config["model"].get("history", 5)
    _positive_int(history, "history")
    if history < 2 or history + maximum_horizon(config) > cfg["train_frames"]:
        raise ValueError("history + maximum horizon must fit inside the training prefix")
    return config


def maximum_horizon(config):
    cfg = config["training"]
    return max(cfg["val_horizons"] + [stage["horizon"] for stage in cfg["curriculum"]])


def horizon_at_step(config, step):
    """One-based optimizer step; boundaries are inclusive."""
    _positive_int(step, "step")
    if step > config["training"]["steps"]:
        raise ValueError("Step exceeds the configured training budget")
    return next(stage["horizon"] for stage in config["training"]["curriculum"]
                if step <= stage["until_step"])


def lr_factor(config, step):
    cfg = config["training"]
    _positive_int(step, "step")
    if step > cfg["steps"]:
        raise ValueError("Step exceeds the configured training budget")
    warmup = cfg["warmup_steps"]
    if warmup and step <= warmup:
        return step / warmup
    progress = (step - warmup) / max(1, cfg["steps"] - warmup)
    progress = min(1.0, max(0.0, progress))
    floor = cfg["minimum_lr_ratio"]
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def config_digest(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def source_hashes():
    names = ["dpbench/train_dissipative.py", "dpbench/dissipative_model.py",
             "dpbench/dissipative_core.py", "dpbench/dissipative_data.py",
             "dpbench/efficient_dissipative_core.py", "dpbench/automatic_discrete_gradient.py",
             "third_party/le-wm/module.py"]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"name": numpy_state[0], "keys": numpy_state[1].tolist(),
                  "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gaussian": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32),
                         n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology differs from the saved checkpoint")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def runtime_signature(device):
    dev = torch.device(device)
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda, "device_type": dev.type,
        "cuda_devices": ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                         if dev.type == "cuda" else []),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def validate_checkpoint(checkpoint, config, dataset_sha256, hashes, device, runtime=None):
    expected = {
        "format_version": CHECKPOINT_VERSION, "kind": "dissipative",
        "config_sha256": config_digest(config), "dataset_sha256": dataset_sha256,
        "source_hashes": hashes, "device": str(device),
    }
    if runtime is not None:
        expected["runtime"] = runtime
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Resume mismatch in {key}; use the original config, data and sources")
    if checkpoint.get("train_config") != config:
        raise ValueError("Resume train_config does not match its expected digest")
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= config["training"]["steps"]:
        raise ValueError("Invalid saved optimizer step")
    if checkpoint.get("sampler_next_step") != step:
        raise ValueError("Saved sampler position is inconsistent with completed optimizer steps")
    required = {"state_dict", "optimizer", "rng", "config", "validation_indices", "best_score", "best_step"}
    if required - set(checkpoint):
        raise ValueError(f"Checkpoint missing {sorted(required - set(checkpoint))}")


def move_batch(batch, device):
    return {key: batch[key].to(device=device, dtype=torch.float32) for key in MODEL_FIELDS}


def scalar_metrics(losses):
    metrics = {}
    for key, value in losses.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not torch.isfinite(value).all():
            raise FloatingPointError(f"Metric {key} must be a finite scalar tensor")
        metrics[key] = float(value.detach())
    return metrics


def prediction_motion_diagnostics(metrics, gate=None):
    """Pool vector moments before ratios; static targets impose no motion floor.

    Dynamic powers are full-batch means with inactive windows zeroed; counts
    are additive. Thus uneven batch partitions give the same result. Window
    activity and the 0.1 stall ratio are defined by model.loss, not test labels.
    """
    required = {"dynamic_window_count", "stalled_dynamic_window_count", "window_count",
                "dynamic_predicted_velocity_power", "dynamic_target_velocity_power",
                "dynamic_velocity_error_power", "dynamic_velocity_cross_power"}
    if required - set(metrics):
        raise ValueError("Prediction motion validation missing metrics: "
                         + str(sorted(required - set(metrics))))
    if not all(math.isfinite(metrics[k]) for k in required):
        raise ValueError("Prediction motion validation requires finite moments")
    n, dynamic, stalled = (metrics[k] for k in
                           ("window_count", "dynamic_window_count", "stalled_dynamic_window_count"))
    if not (0 <= stalled <= dynamic <= n and n > 0):
        raise ValueError("Inconsistent prediction motion window counts")
    if dynamic == 0:
        return dict(prediction_gate_applicable=False, prediction_gate_passed=True,
                    predicted_target_velocity_rms_ratio=None, relative_velocity_error=None,
                    velocity_cosine=None, stalled_dynamic_fraction=None,
                    dynamic_window_fraction=0.0)
    pp, tt, ee, cross = (metrics[k] for k in (
        "dynamic_predicted_velocity_power", "dynamic_target_velocity_power",
        "dynamic_velocity_error_power", "dynamic_velocity_cross_power"))
    if pp < 0 or tt <= 0 or ee < 0:
        raise ValueError("Invalid dynamic velocity powers")
    ratio = math.sqrt(pp / tt)
    relative = ee / tt
    cosine = max(-1.0, min(1.0, cross / math.sqrt(pp * tt))) if pp > 0 else 0.0
    stalled_fraction = stalled / dynamic
    passed = gate is None or (
        gate["min_rms_ratio"] <= ratio <= gate["max_rms_ratio"]
        and relative <= gate["max_relative_error"]
        and cosine >= gate["min_cosine"]
        and stalled_fraction <= gate["max_stalled_fraction"])
    return dict(prediction_gate_applicable=True, prediction_gate_passed=bool(passed),
                predicted_target_velocity_rms_ratio=ratio, relative_velocity_error=relative,
                velocity_cosine=cosine, stalled_dynamic_fraction=stalled_fraction,
                dynamic_window_fraction=dynamic / n)


def validation(model, dataset, indices, config, device):
    """Fixed validation prefixes; equal mode/time/horizon weighting.

    The score is an internal, normalized latent prediction diagnostic, not a
    physical metric. RNG state is restored so validation frequency does not
    change future SIGReg projections or other training randomness.
    """
    cfg = config["training"]
    motion_enabled = (cfg.get("prediction_motion_gate") is not None
                      or cfg.get("validation_motion_weight", 0) > 0)
    previous_mode = model.training
    rng = capture_rng()
    grouped = {}
    for index in indices:
        ref = dataset.index[index]
        grouped.setdefault(window_stratum(ref), []).append(index)
    results = {}
    try:
        random.seed(cfg["val_seed"])
        np.random.seed(cfg["val_seed"])
        torch.manual_seed(cfg["val_seed"])
        model.eval()
        with torch.no_grad():
            for horizon in cfg["val_horizons"]:
                for stratum, group in sorted(grouped.items()):
                    time_bin = stratum[-1]
                    sums, maxima, count = {}, {}, 0
                    count_keys = {"window_count", "dynamic_window_count",
                                  "stalled_dynamic_window_count", "momentum_point_count"}
                    maximum_keys = {"solver_residual", "chain_defect_max", "energy_residual",
                                    "numerical_balance_defect", "solver_residual_ratio",
                                    "solver_split_depth", "solver_replay_relative_error"}
                    for start in range(0, len(group), cfg["batch_size"]):
                        selected = group[start:start + cfg["batch_size"]]
                        batch = move_batch(dataset.batch(selected, horizon=horizon), device)
                        metrics = scalar_metrics(model.loss(batch, horizon=horizon))
                        for key, value in metrics.items():
                            sums[key] = sums.get(key, 0.0) + (value if key in count_keys else len(selected) * value)
                            if key in maximum_keys:
                                maxima[key] = max(maxima.get(key, 0.0), abs(value))
                        count += len(selected)
                    result = {key: value if key in count_keys else value / count
                              for key, value in sums.items()}
                    result.update(maxima)
                    # Pool sufficient statistics before normalization. Averaging
                    # per-batch variances would depend on the batch partition.
                    dimensions = sorted(key.removeprefix("target_mean_") for key in result
                                        if key.startswith("target_mean_"))
                    if not dimensions or "target_motion_mse" not in result:
                        raise ValueError("Validation requires target moments and target_motion_mse")
                    variances = [max(0.0, result[f"target_second_moment_{d}"]
                                     - result[f"target_mean_{d}"] ** 2) for d in dimensions]
                    std_min = math.sqrt(min(variances))
                    motion_rms = math.sqrt(max(0.0, result["target_motion_mse"]))
                    variance_floor = model.config.get("variance_floor", 1e-4)
                    result.update(samples=count, target_std_min=std_min,
                                  target_motion_rms=motion_rms,
                                  normalized_prediction_mse=result["prediction_mse"] / max(float(np.mean(variances)), variance_floor),
                                  gate_passed=(std_min >= cfg["minimum_target_std"]
                                               and motion_rms >= cfg["minimum_target_motion"]))
                    gate_bins = cfg.get("validation_gate_time_bins")
                    result["gate_required"] = gate_bins is None or time_bin in gate_bins
                    result["persistence_ratio"] = result["prediction_mse"] / max(result["persistence_mse"], 1e-8)
                    result["representation_gate_passed"] = result["gate_passed"]
                    if motion_enabled:
                        result.update(prediction_motion_diagnostics(result, cfg.get("prediction_motion_gate")))
                        result["gate_passed"] = (result["representation_gate_passed"]
                                                 and result["prediction_gate_passed"])
                    results[f"h{horizon}_{stratum_label(stratum)}"] = result
    finally:
        restore_rng(rng)
        model.train(previous_mode)
    if not results:
        raise ValueError("No validation groups")
    gate_rows = [row for row in results.values() if row["gate_required"]]
    if not gate_rows:
        raise ValueError("Validation selection contains no required gate time bin")
    position_score = float(np.mean([row["normalized_prediction_mse"] for row in results.values()]))
    active_motion_errors = [row["relative_velocity_error"] for row in results.values()
                            if row.get("prediction_gate_applicable", False)]
    motion_score = float(np.mean(active_motion_errors)) if active_motion_errors else 0.0
    # No observed dynamics is no evidence that the predictor preserves motion.
    # Individual truly settled windows remain exempt from a motion lower bound.
    dynamic_support = cfg.get("prediction_motion_gate") is None or bool(active_motion_errors)
    return {
        "score": position_score + cfg.get("validation_motion_weight", 0.0) * motion_score,
        "position_score": position_score, "motion_score": motion_score,
        "gate_passed": (dynamic_support and all(row["representation_gate_passed"] for row in gate_rows)
                        and all(row.get("prediction_gate_passed", True) for row in results.values())),
        "dynamic_validation_support": dynamic_support,
        "groups": results,
        "score_scope": "Within-run normalized latent position plus configured dynamic velocity error; not an independent physical metric",
        "sigreg_scope": "Diagnostic only; upstream statistic scales with batch size, including smaller final batches",
        "gate_scope": "Representation gates use configured time bins; enabled prediction gates check dynamic windows in ALL bins. Settled target windows have no motion floor.",
        "prediction_motion_gate": cfg.get("prediction_motion_gate"),
        "dynamic_target_velocity_rms_threshold": model.config.get("motion_scale_floor") if motion_enabled else None,
        "stall_velocity_rms_ratio_threshold": 0.1 if motion_enabled else None,
    }


def read_config(path, steps=None, fixed_horizon=None):
    config = json.loads(Path(path).read_text())
    if steps is not None:
        config["training"]["steps"] = steps
    if fixed_horizon is not None:
        config["training"]["curriculum"] = [{"until_step": config["training"]["steps"],
                                               "horizon": fixed_horizon}]
    return validate_config(config)


def preflight(data, config):
    from dpbench.dissipative_model import resolve_model_config

    validate_config(config)
    resolved_model = resolve_model_config(config["model"])
    report = preflight_dataset(data, history=resolved_model["history"],
                               max_horizon=maximum_horizon(config),
                               train_frames=config["training"]["train_frames"])
    if report["image_shape"] != [64, 64, 3] or report["action_dim"] != 1:
        raise ValueError("The visual adapter requires 64x64 RGB and exactly one action channel")
    if report["requires_environment_context"] and resolved_model.get("environment_context_dim", 0) < 1:
        raise ValueError("Multi-environment data requires environment_context_dim > 0 inferred from causal history")
    intervals = 0
    with h5py.File(data, "r") as handle:
        renderer = handle.attrs.get("renderer", "legacy")
        if isinstance(renderer, bytes):
            renderer = renderer.decode("utf-8")
        required_renderer = config["training"].get("required_renderer")
        if required_renderer is not None and renderer != required_renderer:
            raise ValueError(f"Dataset renderer {renderer!r} differs from required_renderer={required_renderer!r}; regenerate into a new directory")
        for split in ("train", "val"):
            for episode in handle[f"splits/{split}"][:]:
                start = int(handle["ep_offset"][episode])
                length = min(int(handle["ep_len"][episode]), report["train_frames"])
                dt = handle["dt_frame"][start:start + length - 1]
                if not np.allclose(dt, resolved_model["dt"], rtol=1e-6, atol=1e-9):
                    raise ValueError("Training/validation dt differs from model.dt; this dataset/evaluation adapter uses a fixed frame clock")
                intervals += len(dt)
    report.update(mode="preflight_only", optimizer_created=False, run_directory_created=False,
                  config=config, config_sha256=config_digest(config),
                  resolved_model_config=resolved_model,
                  renderer=renderer,
                  checked_frame_intervals=intervals,
                  validation_horizons_seconds=[h * resolved_model["dt"] for h in config["training"]["val_horizons"]],
                  validation_horizons_steps=config["training"]["val_horizons"],
                  note="No training runs without --execute. The step budget is unprofiled, not an optimality or cost claim.")
    return report


def execute_run(data, output, config, device="cpu", resume=None, stop_after_step=None):
    """Explicit execution boundary; unlike preflight, this updates parameters."""
    from dpbench.dissipative_model import DissipativeWorldModel

    report = preflight(data, config)
    cfg = config["training"]
    end_step = cfg["steps"] if stop_after_step is None else stop_after_step
    _positive_int(end_step, "stop_after_step")
    if end_step > cfg["steps"] or (end_step != cfg["steps"] and end_step % cfg["val_every"]):
        raise ValueError("stop_after_step must be a scheduled validation step within the full budget")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("The float64 dynamics implementation supports CPU/CUDA; MPS is not supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty run directory: {output}")
    hashes = source_hashes()
    runtime = runtime_signature(device)
    saved = None
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        validate_checkpoint(saved, config, report["dataset_sha256"], hashes, device, runtime)
        if end_step < saved["step"]:
            raise ValueError("stop_after_step precedes the saved optimizer step")
        if end_step == saved["step"] and saved.get("validation_status") == "completed":
            raise ValueError("No updates or pending validation remain within this run budget")
    torch.set_num_threads(cfg["threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = DissipativeWorldModel(config["model"]).to(device)
    encoder = list(model.encoder_parameters())
    dynamics = list(model.dynamics_parameters())
    ids = [id(parameter) for parameter in encoder + dynamics]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError("Optimizer groups must partition all trainable parameters exactly once")
    optimizer = torch.optim.AdamW([
        {"params": encoder, "lr": cfg["encoder_lr"], "base_lr": cfg["encoder_lr"], "name": "encoder"},
        {"params": dynamics, "lr": cfg["dynamics_lr"], "base_lr": cfg["dynamics_lr"], "name": "dynamics"},
    ], weight_decay=cfg["weight_decay"])
    kwargs = dict(history=model.config["history"], max_horizon=maximum_horizon(config),
                  train_frames=cfg["train_frames"])
    train = DissipativeWindowDataset(data, "train", **kwargs)
    val = DissipativeWindowDataset(data, "val", **kwargs)
    sampler = StepBatchSampler(train, cfg["batch_size"], cfg["seed"])
    val_indices = fixed_validation_indices(val, cfg["val_windows"], cfg["val_seed"])
    start_step, best, best_step, best_source = 0, None, None, None
    inherited_best = None
    if saved:
        if saved["config"] != model.config or saved["validation_indices"] != val_indices:
            raise ValueError("Resolved model config or validation anchors changed since checkpoint")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step, best, best_step = saved["step"], saved["best_score"], saved["best_step"]
        if best is not None:
            # Prefer a sibling to support moving the checkpoint pair together.
            sibling = Path(resume).resolve().parent / "best.pt"
            best_source = sibling if sibling.is_file() else Path(saved["best_checkpoint_path"])
            inherited_best = torch.load(best_source, map_location="cpu", weights_only=True)
            validate_checkpoint(inherited_best, config, report["dataset_sha256"], hashes, device, runtime)
            if (inherited_best["step"] != best_step or inherited_best["step"] > start_step
                    or inherited_best["best_score"] != best):
                raise ValueError("Saved best checkpoint does not match the resumed history")
        restore_rng(saved["rng"])
    output.mkdir(parents=True, exist_ok=True)
    if inherited_best is not None:
        atomic_torch_save(output / "best.pt", inherited_best)
        best_source = output / "best.pt"
    atomic_json(output / "run.json", {
        "kind": model.kind, "config": model.config, "train_config": config,
        "config_sha256": config_digest(config), "data_path": str(Path(data).resolve()),
        "dataset_sha256": report["dataset_sha256"], "source_hashes": hashes,
        "validation_indices": val_indices, "preflight": report, "provenance": model.provenance(),
        "device": str(device), "python": platform.python_version(), "torch": str(torch.__version__),
        "resume": str(Path(resume).resolve()) if resume else None,
        "stop_after_step": stop_after_step,
        "solver_migration": saved.get("solver_migration") if saved else None,
        "precision": "float32 vision, float64 dynamics", "training_started": True,
        "runtime": runtime,
        "resume_scope": "Same-runtime configuration, RNG and sampler continuation; bitwise determinism across hardware is not claimed. Move last.pt and best.pt together.",
        "physical_validation": False,
        "checkpoint_rule": "minimum configured position/motion validation score, only if representation and enabled prediction gates pass",
    })
    status = {"status": "running", "training_started": True, "completed_steps": start_step,
              "optimizer_steps_this_run": 0}
    atomic_json(output / "status.json", status)

    def checkpoint_at(step, validation_result):
        # An unvalidated last.pt is a recovery point only, never a best model.
        # Validation restores RNG, so either version resumes the same sampler,
        # optimizer and stochastic-training sequence after this completed step.
        return {
            "format_version": CHECKPOINT_VERSION, "kind": model.kind, "config": model.config,
            "train_config": config, "config_sha256": config_digest(config),
            "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "sampler_next_step": step, "rng": capture_rng(),
            "dataset_sha256": report["dataset_sha256"], "data_path": str(Path(data).resolve()),
            "source_hashes": hashes, "device": str(device), "seed": cfg["seed"],
            "runtime": runtime, "train_frames": cfg["train_frames"],
            "validation": validation_result,
            "validation_status": "not_run" if validation_result is None else "completed",
            "validation_indices": val_indices, "best_score": best, "best_step": best_step,
            "best_checkpoint_path": str(best_source) if best_source else None,
            "solver_migration": saved.get("solver_migration") if saved else None,
        }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def peak_memory():
        if device.type != "cuda":
            return {}
        return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}

    def validate_and_checkpoint(step, row):
        nonlocal best, best_step, best_source
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        validation_started = time.perf_counter()
        result = validation(model, val, val_indices, config, device)
        synchronize()
        row["validation_seconds"] = time.perf_counter() - validation_started
        row["validation_memory"] = peak_memory()
        row["validation"] = result
        improved = result["gate_passed"] and (best is None or result["score"] < best)
        if improved:
            best, best_step, best_source = result["score"], step, output / "best.pt"
        checkpoint_started = time.perf_counter()
        checkpoint = checkpoint_at(step, result)
        if improved:
            atomic_torch_save(output / "best.pt", checkpoint)
        atomic_torch_save(output / "last.pt", checkpoint)
        atomic_json(output / "status.json", status)
        row["checkpoint_seconds"] += time.perf_counter() - checkpoint_started

    model.train()
    started = time.perf_counter()
    try:
        with (output / "history.jsonl").open("x") as history:
            if (saved and saved.get("validation_status") == "not_run"
                    and (start_step % cfg["val_every"] == 0 or start_step == cfg["steps"])):
                # A validation failure must not silently skip model selection,
                # even if the interrupted checkpoint is the final update.
                recovery = {"event": "resumed_pending_validation", "step": start_step,
                            "checkpoint_seconds": 0.0, "optimizer_steps": 0}
                validate_and_checkpoint(start_step, recovery)
                atomic_json(output / "resume_validation.json", recovery)
                print(json.dumps(recovery, allow_nan=False), flush=True)
            for step in range(start_step + 1, end_step + 1):
                synchronize()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                tick = time.perf_counter()
                horizon = horizon_at_step(config, step)
                batch = move_batch(train.batch(sampler.indices(step - 1), horizon=horizon), device)
                factor = lr_factor(config, step)
                for group in optimizer.param_groups:
                    group["lr"] = group["base_lr"] * factor
                optimizer.zero_grad(set_to_none=True)
                losses = model.loss(batch, horizon=horizon)
                metrics = scalar_metrics(losses)
                losses["loss"].backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                synchronize()
                status.update(completed_steps=step, optimizer_steps_this_run=step - start_step)
                row = {"step": step, "horizon_steps": horizon, "horizon_seconds_at_model_dt": horizon * model.dt,
                       "losses": metrics, "gradient_norm": float(grad), "lr_factor": factor,
                       "step_seconds": time.perf_counter() - tick,
                       "validation_seconds": 0.0, "checkpoint_seconds": 0.0,
                       "training_memory": peak_memory()}
                scheduled_validation = step % cfg["val_every"] == 0 or step == cfg["steps"]
                if step == 1 or scheduled_validation:
                    checkpoint_started = time.perf_counter()
                    atomic_torch_save(output / "last.pt", checkpoint_at(step, None))
                    atomic_json(output / "status.json", status)
                    row["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
                if scheduled_validation:
                    validate_and_checkpoint(step, row)
                row["iteration_seconds"] = time.perf_counter() - tick
                history.write(json.dumps(row, allow_nan=False) + "\n")
                history.flush()
                print(json.dumps(row, allow_nan=False), flush=True)
        status.update(status="completed" if end_step == cfg["steps"] else "budget_completed",
                      configured_total_steps=cfg["steps"], stop_after_step=end_step,
                      full_budget_completed=end_step == cfg["steps"], best_score=best, best_step=best_step,
                      best_available=best is not None, elapsed_seconds=time.perf_counter() - started,
                      physical_validation=False)
        atomic_json(output / "status.json", status)
        return status
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error),
                      failed_samples_skipped=False, elapsed_seconds=time.perf_counter() - started)
        atomic_json(output / "status.json", status)
        raise
    finally:
        train.close()
        val.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable optimizer updates")
    parser.add_argument("--resume", type=Path, help="Resume into a new empty output directory; exact config/data/source match required")
    parser.add_argument("--stop-after-step", type=int,
                        help="Stop after this validation checkpoint without changing the full LR/curriculum schedule")
    parser.add_argument("--steps", type=int, help="Override optimizer step budget; does not enable execution")
    parser.add_argument("--fixed-horizon", type=int, help="Replace curriculum with a fixed rollout length")
    args = parser.parse_args(argv)
    config = read_config(args.config, steps=args.steps, fixed_horizon=args.fixed_horizon)
    if not args.execute:
        result = preflight(args.data, config)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return result
    if args.output is None:
        parser.error("--execute requires --output")
    return execute_run(args.data, args.output, config, args.device, args.resume, args.stop_after_step)


if __name__ == "__main__":
    main()
