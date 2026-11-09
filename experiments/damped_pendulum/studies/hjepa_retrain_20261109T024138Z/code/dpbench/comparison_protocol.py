"""Matched data/budget definitions and immutable source identities for comparisons."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from .dissipative_data import preflight_dataset

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("lehami", "lewm", "hjepa")
CONDITIONS = {"a": [0.], "single": [.2], "multi": [0., .05, .1, .2, .5, 1.]}
SCOPES = {"a": "undamped_stage_a", "single": "damping_stage_b_single", "multi": "damping_stage_b_multi"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_manifest():
    from .models import verify_sources
    verify_sources()
    paths = [ROOT / "third_party/sources.json", ROOT / "third_party/hjepa_source.json"]
    paths += sorted((ROOT / "dpbench").glob("comparison*.py"))
    paths += [ROOT / n for n in ("dpbench/train_comparison.py", "dpbench/evaluate_comparison.py",
              "dpbench/dissipative_data.py", "dpbench/train_dissipative.py", "dpbench/models.py",
              "dpbench/passive_family.py", "dpbench/damping_stage_data.py", "dpbench/generate.py",
              "dpbench/evaluate.py", "dpbench/evaluate_passive.py", "scripts/evaluate_staged_damping.py")]
    paths += [ROOT / "scripts" / name for name in (
        "run_comparison.py", "train_comparison.py", "evaluate_comparison.py",
        "prepare_comparison_data.py", "bootstrap_comparison.py", "slurm_comparison.sh")]
    # Hash executed upstream bytes, not just repository names or commit strings.
    for row in json.loads((ROOT / "third_party/sources.json").read_text()):
        paths += [ROOT / "third_party" / row["directory"] / f["path"] for f in row["files"]]
    paths += sorted(p for p in (ROOT / "third_party/hjepa").rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def default_training_config(kind, seed=0, steps=20000):
    return dict(steps=steps, batch_size=32, train_frames=81, seed=seed, threads=4,
                lr={"lehami": 5e-5, "lewm": 1e-4, "hjepa": 1e-4}[kind],
                weight_decay=1e-4, warmup_steps=500, minimum_lr_ratio=.1,
                gradient_clip=1., val_every=250, val_windows=288, val_seed=1009,
                val_horizons=[8, 32], checkpoint_every=250,
                curriculum=[dict(until_step=1000, horizon=4), dict(until_step=4000, horizon=8),
                            dict(until_step=10000, horizon=16), dict(until_step=max(20000, steps), horizon=32)])


def make_config(kind, stage="a", seed=0, steps=20000, tiny=False):
    from .comparison_models import default_comparison_model_config
    if stage not in CONDITIONS:
        raise ValueError("stage must be a, single or multi")
    return dict(schema_version=1, training_route="staged", model=default_comparison_model_config(kind, tiny=tiny),
                training=default_training_config(kind, seed, steps),
                data=dict(stage=stage, selected_damping=CONDITIONS[stage], renderer="smooth_v1",
                          history=21, dt=.1, train_frames=81,
                          expected_groups={"train": 384, "val": 64}),
                fixture_only=bool(tiny))


def validate_config(config):
    if set(config) != {"schema_version", "training_route", "model", "training", "data", "fixture_only"} or config["schema_version"] != 1:
        raise ValueError("Unsupported comparison training schema")
    if config["training_route"] not in {"staged", "scratch"}:
        raise ValueError("training_route must explicitly be staged or scratch")
    if type(config["fixture_only"]) is not bool or config["model"].get("kind") not in KINDS:
        raise ValueError("Invalid model / fixture identity")
    if config["model"].get("history")!=21 or config["model"].get("dt")!=.1:
        raise ValueError("Model must preserve history21 and physical dt=.1")
    if not config["fixture_only"] and config["model"].get("tiny_fixture",False):
        raise ValueError("Tiny software fixture cannot be labeled a scientific model")
    t, d = config["training"], config["data"]
    required = set(default_training_config(config["model"]["kind"]))
    if set(t) != required:
        raise ValueError(f"Training keys differ: {set(t)^required}")
    for key in ("steps", "batch_size", "train_frames", "threads", "val_every", "val_windows", "checkpoint_every"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if t["batch_size"] < 2:
        raise ValueError("BatchNorm/representation losses require batch_size >= 2")
    for key in ("seed", "val_seed", "warmup_steps"):
        if type(t[key]) is not int or t[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("lr", "gradient_clip"):
        if isinstance(t[key], bool) or not np.isfinite(t[key]) or t[key] <= 0:
            raise ValueError(f"{key} must be finite positive")
    if (not np.isfinite(t["weight_decay"]) or t["weight_decay"] < 0
            or not 0 <= t["minimum_lr_ratio"] <= 1):
        raise ValueError("Invalid optimizer schedule")
    previous = 0
    for stage in t["curriculum"]:
        if (set(stage) != {"until_step", "horizon"} or type(stage["until_step"]) is not int
                or stage["until_step"] <= previous or type(stage["horizon"]) is not int or stage["horizon"] < 1):
            raise ValueError("Invalid ordered curriculum")
        previous = stage["until_step"]
    if previous < t["steps"] or not t["val_horizons"] or any(type(h) is not int or h < 1 for h in t["val_horizons"]):
        raise ValueError("Invalid horizon coverage")
    if (d["stage"] not in CONDITIONS or d["selected_damping"] != CONDITIONS[d["stage"]]
            or d["history"] != 21 or d["dt"] != .1 or d["renderer"] != "smooth_v1"
            or d["train_frames"] != 81 or t["train_frames"] != 81):
        raise ValueError("Keep the established data/time/condition protocol")
    if 21 + maximum_horizon(config) + 2 > 81:
        raise ValueError("Training window exceeds allowed prefix")
    if not config["fixture_only"] and d["expected_groups"] != {"train": 384, "val": 64}:
        raise ValueError("Scientific comparison requires 384/64 paired groups")
    return config


def maximum_horizon(config):
    return max(config["training"]["val_horizons"] + [s["horizon"] for s in config["training"]["curriculum"]])


def preflight(data_path, config):
    validate_config(config)
    d = config["data"]
    result = preflight_dataset(data_path, history=21, max_horizon=maximum_horizon(config),
                               train_frames=81, target_tail_frames=2)
    with h5py.File(data_path, "r") as f:
        text = lambda x: x.decode() if isinstance(x, bytes) else x
        if text(f.attrs.get("renderer")) != d["renderer"] or text(f.attrs.get("training_scope")) != SCOPES[d["stage"]]:
            raise ValueError("Dataset renderer or stage scope differs")
        if result["image_shape"] != [64,64,3] or result["action_dim"] != 1:
            raise ValueError("Require 64x64 RGB and one passive action")
        for split in ("train", "val"):
            ids = f[f"splits/{split}"][:]
            groups = f["group_id"][:][ids]
            values = f["damping"][:][ids]
            if len(set(groups.tolist())) != d["expected_groups"][split]:
                raise ValueError(f"Unexpected {split} group count")
            for gid in np.unique(groups):
                if not np.array_equal(np.sort(values[groups == gid]), np.asarray(d["selected_damping"])):
                    raise ValueError("Each initial group must have each declared condition exactly once")
            for ep in ids:
                off, length = int(f["ep_offset"][ep]), min(81,int(f["ep_len"][ep]))
                dt = f["dt_frame"][off:off+length-1]
                act = f["action"][off:off+length-1]
                if not np.allclose(dt, .1, rtol=0, atol=1e-7) or not np.all(act == 0):
                    raise ValueError("Only passive data with unchanged dt=.1 are supported")
        from .passive_family import _parent_config_sha256
        parent=json.loads(text(f.attrs["parent_config_json"]))
        parent_hash=_parent_config_sha256(parent)
        if text(f.attrs["parent_config_sha256"]) != parent_hash:
            raise ValueError("Parent protocol JSON/hash mismatch")
        result["parent_config_sha256"] = parent_hash
    result.update(stage=d["stage"], selected_damping=d["selected_damping"],
                  protocol_sha256=json_digest(config), optimizer_created=False,
                  fixture_only=config["fixture_only"], full_history_available=True)
    return result
