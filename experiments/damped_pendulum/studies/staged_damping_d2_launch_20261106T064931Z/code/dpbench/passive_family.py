"""Passive multi-damping data; default CLI is a no-write plan, never training.

Initial-condition groups are split before expansion across constant damping
values. Unseen-damping files contain test groups only. All forces are zero;
physical state, damping and energy strata are metadata, not model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from .generate import derivative, energy, render, simulate, validate_renderer


SPLITS = ("train", "val", "test")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/passive_family.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_config(config):
    required = {"family_name", "groups", "split_seeds", "group_id_offset", "seen_damping",
                "interpolation_damping", "extrapolation_damping", "theta_range", "omega_range",
                "energy_edges", "history", "train_frames", "test_frames", "dt", "substeps",
                "image_size", "mass", "length", "gravity"}
    optional = {"renderer"}
    if not required <= set(config) or set(config) - required - optional:
        raise ValueError(f"Configuration keys differ: missing={required - set(config)}, "
                         f"unknown={set(config) - required - optional}")
    validate_renderer(config.get("renderer", "legacy"))
    if not isinstance(config["family_name"], str) or not config["family_name"]:
        raise ValueError("family_name must be nonempty")
    for key in ("history", "train_frames", "test_frames", "substeps", "image_size"):
        _integer(config[key], key)
    _integer(config["group_id_offset"], "group_id_offset", 0)
    if config["history"] < 3 or config["history"] >= config["train_frames"]:
        raise ValueError("Require 3 <= history < train_frames")
    if config["test_frames"] < config["train_frames"] or config["image_size"] != 64:
        raise ValueError("Require test_frames >= train_frames and 64x64 RGB")
    for key in ("dt", "mass", "length", "gravity"):
        if isinstance(config[key], bool) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("theta_range", "omega_range"):
        values = config[key]
        if len(values) != 2 or not all(math.isfinite(x) for x in values) or values[0] >= values[1]:
            raise ValueError(f"Invalid {key}")
    edges = config["energy_edges"]
    if len(edges) < 2 or edges[0] <= 0 or not all(math.isfinite(x) for x in edges):
        raise ValueError("Energy edges must be finite and positive")
    if any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("Energy edges must increase")
    bands = len(edges) - 1
    if set(config["groups"]) != set(SPLITS) or set(config["split_seeds"]) != set(SPLITS):
        raise ValueError("groups and split_seeds must specify train/val/test")
    for split in SPLITS:
        _integer(config["groups"][split], f"groups.{split}")
        _integer(config["split_seeds"][split], f"split_seeds.{split}", 0)
        if config["groups"][split] % bands:
            raise ValueError("Each split's group count must be divisible by the energy-band count")
    if len(set(config["split_seeds"].values())) != len(SPLITS):
        raise ValueError("Split seeds must be distinct")
    damping = []
    for key in ("seen_damping", "interpolation_damping", "extrapolation_damping"):
        values = config[key]
        if not values or not all(not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in values):
            raise ValueError(f"{key} must contain finite nonnegative coefficients")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError(f"{key} must be strictly increasing")
        damping.extend(values)
    if len(set(damping)) != len(damping) or len(config["seen_damping"]) < 2:
        raise ValueError("Damping partitions must be disjoint, with at least two seen values")
    low, high = min(config["seen_damping"]), max(config["seen_damping"])
    if any(not low < b < high for b in config["interpolation_damping"]):
        raise ValueError("Interpolation coefficients must lie strictly inside the seen range")
    if any(low <= b <= high for b in config["extrapolation_damping"]):
        raise ValueError("Extrapolation coefficients must lie outside the seen range")
    return config


def _parent_config_sha256(config):
    """Hash the complete parent protocol, independently of selected conditions."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan(config, *, undamped_only=False):
    """Report exact sizes and raw RGB storage; do not create files or sample data."""
    validate_config(config)
    if undamped_only and 0.0 not in config["seen_damping"]:
        raise ValueError("undamped-only requires damping 0 in the parent seen_damping partition")
    counts = config["groups"]
    seen = 1 if undamped_only else len(config["seen_damping"])
    unseen = 0 if undamped_only else len(config["interpolation_damping"]) + len(config["extrapolation_damping"])
    seen_frames = seen * ((counts["train"] + counts["val"]) * config["train_frames"]
                          + counts["test"] * config["test_frames"])
    unseen_frames = unseen * counts["test"] * config["test_frames"]
    result = {"mode": "plan_only", "training_started": False, "writes_files": False,
            "family_name": config["family_name"], "groups": counts,
            "renderer": config.get("renderer", "legacy"),
            "rendering_contract": "Versioned geometry; smooth_v1 is continuous before uint8 rounding, "
                                  "but stored pixels retain quantization boundaries. New version requires new data.",
            "seen_episodes": {s: seen * counts[s] for s in SPLITS},
            "unseen_episodes": {"interpolation": 0 if undamped_only else len(config["interpolation_damping"]) * counts["test"],
                                "extrapolation": 0 if undamped_only else len(config["extrapolation_damping"]) * counts["test"]},
            "seen_frames": seen_frames, "unseen_frames": unseen_frames,
            "raw_rgb_bytes": (seen_frames + unseen_frames) * config["image_size"]**2 * 3,
            "storage_note": "Raw RGB only; excludes metadata and compression. Actual size is measured after generation.",
            "history_seconds": (config["history"] - 1) * config["dt"],
            "test_rollout_seconds": (config["test_frames"] - config["history"]) * config["dt"],
            "model_fields": ["pixels", "action", "dt_frame"],
            "requires_environment_context": not undamped_only,
            "source": "Project-generated passive pendulum simulations, not an external dataset"}
    if undamped_only:
        result.update(selection_scope="undamped_subset", selected_damping=[0.0],
                      training_scope="undamped_stage_a",
                      parent_config_sha256=_parent_config_sha256(config),
                      output_files=["undamped.h5"],
                      pairing_contract="Reuse all parent initial groups and split identities; "
                                       "select b=0 before simulation/rendering. No resampling.")
    return result


def sample_groups(config):
    """Fixed-seed rejection samples in the original box, balanced by initial energy."""
    validate_config(config)
    result, group_offset = {}, config["group_id_offset"]
    bands = list(zip(config["energy_edges"], config["energy_edges"][1:]))
    physical = {key: config[key] for key in ("mass", "length", "gravity")}
    for split in SPLITS:
        rng = np.random.default_rng(config["split_seeds"][split])
        quota = config["groups"][split] // len(bands)
        samples, band_ids = [], []
        for band, (low, high) in enumerate(bands):
            selected, draws = [], 0
            while len(selected) < quota:
                n = max(256, 4 * (quota - len(selected)))
                states = np.column_stack((rng.uniform(*config["theta_range"], n),
                                          rng.uniform(*config["omega_range"], n)))
                e = energy(states, **physical) / (config["mass"] * config["gravity"] * config["length"])
                selected.extend(states[(e >= low) & (e < high)].tolist())
                draws += n
                if draws > 1_000_000:
                    raise ValueError(f"Energy band {band} cannot be filled from the declared initial box")
            samples.extend(selected[:quota])
            band_ids.extend([band] * quota)
        order = rng.permutation(len(samples))
        result[split] = {"initial": np.asarray(samples, dtype=np.float64)[order],
                         "energy_band": np.asarray(band_ids, dtype=np.int64)[order],
                         "group_id": np.arange(group_offset, group_offset + len(samples), dtype=np.int64)}
        group_offset += len(samples)
    all_initial = np.concatenate([result[s]["initial"] for s in SPLITS])
    if len(np.unique(all_initial, axis=0)) != len(all_initial):
        raise ValueError("Exact duplicate initial states occurred; no silent resampling is permitted")
    return result


def _reference_audit(initial, state, dissipation, bands, config, damping):
    """One full-horizon DOP853 trajectory per initial-energy band, per damping."""
    ids = [int(np.flatnonzero(bands == band)[0]) for band in range(len(config["energy_edges"]) - 1)]
    times = np.arange(state.shape[1], dtype=np.float64) * config["dt"]
    physical = {key: config[key] for key in ("mass", "length", "gravity")}
    state_error, diss_error = 0., 0.
    for ep in ids:
        ref = solve_ivp(lambda _, y: derivative(y, 0., **physical, damping=damping),
                        (0., float(times[-1])), np.r_[initial[ep], 0., 0.], t_eval=times,
                        method="DOP853", rtol=1e-11, atol=1e-12)
        if not ref.success or not np.isfinite(ref.y).all():
            raise RuntimeError(f"Reference solver failed: {ref.message}")
        state_error = max(state_error, float(np.abs(ref.y[:2].T - state[ep]).max()))
        cumulative = np.r_[0., np.cumsum(dissipation[ep, :-1])]
        diss_error = max(diss_error, float(np.abs(ref.y[3] - cumulative).max()))
    if max(state_error, diss_error) > 1e-5:
        raise RuntimeError("DOP853 audit exceeded 1e-5; increase RK4 substeps")
    return {"audited_episodes": ids, "energy_bands": len(ids), "frames_each": len(times),
            "state_max_abs": state_error, "cumulative_dissipation_max_abs": diss_error,
            "reference": "DOP853 rtol=1e-11 atol=1e-12; one full test trajectory per initial-energy band"}


def _records(config, groups, file_kind):
    conditions = [(b, "seen") for b in config["seen_damping"]]
    conditions += [(b, "interpolation") for b in config["interpolation_damping"]]
    conditions += [(b, "extrapolation") for b in config["extrapolation_damping"]]
    records = []
    for damping_id, (damping, regime) in enumerate(conditions):
        if (regime == "seen") != (file_kind == "seen"):
            continue
        for split in (SPLITS if file_kind == "seen" else ("test",)):
            for i, group in enumerate(groups[split]["group_id"]):
                records.append({"split": split, "initial": groups[split]["initial"][i],
                                "energy_band": int(groups[split]["energy_band"][i]),
                                "group_id": int(group), "damping": damping,
                                "damping_id": damping_id, "regime": regime,
                                "frames": config["test_frames"] if split == "test" else config["train_frames"]})
    return records


def _write_file(path, config, groups, kind, *, undamped_only=False):
    renderer = validate_renderer(config.get("renderer", "legacy"))
    records = _records(config, groups, kind)
    if undamped_only:
        if kind != "seen" or 0.0 not in config["seen_damping"]:
            raise ValueError("Undamped subset must select b=0 from the parent seen partition")
        # Filter metadata before any ODE integration or rendering. Do not build
        # the complete family and discard its other physical trajectories.
        records = [r for r in records if r["damping"] == 0.0]
    lengths = np.asarray([r["frames"] for r in records], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths[:-1])]
    total, episodes = int(lengths.sum()), len(records)
    numerical = []
    context_hashes = {s: {} for s in SPLITS}
    with h5py.File(path, "x") as f:
        f.attrs.update(schema_version=1, family_name=config["family_name"], renderer=renderer,
                       config_json=json.dumps(config, sort_keys=True),
                       requires_environment_context=not undamped_only, history_required=config["history"],
                       train_frames=config["train_frames"] if kind == "seen" else 0,
                       dt=config["dt"], mass=config["mass"], length=config["length"], gravity=config["gravity"],
                       damping_scope="Constant within each episode; hidden environment, never a model input",
                       ode="m L^2 theta_ddot = -m g L sin(theta) - b omega; external torque identically zero",
                       truth_integrator="float64 RK4 with integrated dissipation; per-band DOP853 audit")
        if undamped_only:
            f.attrs.update(selection_scope="undamped_subset", selected_damping_json=json.dumps([0.0]),
                           training_scope="undamped_stage_a",
                           parent_config_sha256=_parent_config_sha256(config),
                           parent_config_json=json.dumps(config, sort_keys=True))
        f.create_dataset("ep_len", data=lengths)
        f.create_dataset("ep_offset", data=offsets)
        for key in ("group_id", "damping_id", "energy_band"):
            f.create_dataset(key, data=np.asarray([r[key] for r in records], dtype=np.int64))
        # Sampler metadata only; identical indexing across seen and unseen files.
        f.create_dataset("regime_id", data=np.asarray([r["damping_id"] for r in records], dtype=np.int64))
        f.create_dataset("damping", data=np.asarray([r["damping"] for r in records]))
        f.create_dataset("initial_state", data=np.stack([r["initial"] for r in records]))
        f.create_dataset("mode", data=np.zeros(episodes, dtype=np.int64))
        for split in SPLITS:
            f.create_dataset("splits/" + split, data=np.asarray([i for i, r in enumerate(records) if r["split"] == split], dtype=np.int64))
        for regime in ("seen", "interpolation", "extrapolation"):
            f.create_dataset("evaluation/" + regime, data=np.asarray([i for i, r in enumerate(records) if r["regime"] == regime and r["split"] == "test"], dtype=np.int64))
        f.create_dataset("pixels", (total, config["image_size"], config["image_size"], 3), dtype="u1",
                         chunks=(1, config["image_size"], config["image_size"], 3), compression="lzf")
        for key, shape in (("state", (total, 2)), ("energy", (total,)), ("work", (total,)),
                           ("dissipation", (total,)), ("reference_residual", (total,))):
            f.create_dataset(key, shape, dtype="f8", compression="lzf")
        f.create_dataset("action", (total, 1), dtype="f4", fillvalue=0., compression="lzf")
        f.create_dataset("dt_frame", data=np.full((total, 1), config["dt"], dtype=np.float32), compression="lzf")
        f.create_dataset("valid_transition", data=np.ones(total, dtype=bool), compression="lzf")
        for damping_id in sorted({r["damping_id"] for r in records}):
            for split in SPLITS:
                ids = [i for i, r in enumerate(records) if r["damping_id"] == damping_id and r["split"] == split]
                if not ids:
                    continue
                damping, frames = records[ids[0]]["damping"], records[ids[0]]["frames"]
                initial = np.stack([records[i]["initial"] for i in ids])
                bands = np.asarray([records[i]["energy_band"] for i in ids])
                action = np.zeros((len(ids), frames - 1), dtype=np.float64)
                state, work, dissipation = simulate(initial, action, config["dt"], config["substeps"],
                                                   config["mass"], config["length"], config["gravity"], damping)
                en = energy(state, config["mass"], config["length"], config["gravity"])
                residual = np.diff(en, axis=1) + dissipation[:, :-1]
                if not all(np.isfinite(x).all() for x in (state, work, dissipation, en, residual)):
                    raise RuntimeError("Nonfinite simulation output")
                if work.any() or (dissipation < 0).any() or np.abs(residual).max() > 1e-6:
                    raise RuntimeError("Passive energy/work/dissipation audit failed")
                audit = {"damping": damping, "split": split, "episodes": len(ids),
                         "energy_balance_max_abs": float(np.abs(residual).max()),
                         "max_energy_increase": float(max(0., np.diff(en, axis=1).max())),
                         "work_exactly_zero": True, "dissipation_nonnegative": True}
                if split == "test":
                    audit["independent_reference"] = _reference_audit(initial, state, dissipation, bands, config, damping)
                    audit["independent_reference"]["file_episode_ids"] = [
                        ids[i] for i in audit["independent_reference"]["audited_episodes"]]
                numerical.append(audit)
                for local, ep in enumerate(ids):
                    start, end = int(offsets[ep]), int(offsets[ep] + lengths[ep])
                    images = np.stack([render(q, config["image_size"], renderer=renderer)
                                       for q in state[local, :, 0]])
                    f["pixels"][start:end] = images
                    for key, values in (("state", state), ("energy", en), ("work", work), ("dissipation", dissipation)):
                        f[key][start:end] = values[local]
                    f["reference_residual"][start:end] = np.r_[residual[local], 0.]
                    f["valid_transition"][end - 1] = False
                    key = hashlib.sha256(images[:config["history"]].tobytes()).hexdigest()
                    context_hashes[split].setdefault(key, []).append(ep)
        f.attrs["numerical_audit_json"] = json.dumps(numerical, sort_keys=True)
    collisions = []
    for split, by_hash in context_hashes.items():
        for ids in by_hash.values():
            by_group = {}
            for ep in ids:
                by_group.setdefault(records[ep]["group_id"], []).append(ep)
            for group, matching in by_group.items():
                if len(matching) > 1:
                    collisions.append({"split": split, "group_id": group,
                                       "episodes": matching,
                                       "damping": [records[ep]["damping"] for ep in matching]})
    result = {"file": path.name, "kind": kind, "renderer": renderer, "episodes": episodes, "frames": total,
            "sha256": sha256(path), "bytes": path.stat().st_size, "numerical_audit": numerical,
            "different_damping_same_group_identical_initial_context": {
                "collision_groups": len(collisions), "examples": collisions[:20],
                "scope": "Exact RGB equality across the complete initial history, within one initial group; no approximate-identifiability claim"}}
    if undamped_only:
        result.update(selection_scope="undamped_subset", selected_damping=[0.0],
                      training_scope="undamped_stage_a",
                      parent_config_sha256=_parent_config_sha256(config))
    return result, context_hashes


def _context_overlap(seen, unseen):
    """Exact initial RGB-context audit only; do not reject or search new seeds."""
    development = set(seen["train"]) | set(seen["val"])
    def hit_ids(target, reference):
        return sorted(ep for digest, ids in target.items() if digest in reference for ep in ids)
    return {"scope": "SHA256 of the first history frames only, all damping conditions combined; all actions zero. Not later-window or approximate-duplicate screening.",
            "seen_val_matches_train": hit_ids(seen["val"], set(seen["train"])),
            "seen_test_matches_development": hit_ids(seen["test"], development),
            "unseen_test_matches_development": hit_ids(unseen["test"], development),
            "unseen_test_matches_seen_test": hit_ids(unseen["test"], set(seen["test"])),
            "policy": "Report all collisions without deletion or silent resampling. Seen/unseen test groups are deliberately paired; their overlap is not training leakage."}


def _cross_condition_ambiguity(config, groups, seen, unseen):
    """Count identical test contexts across all b, including across file boundaries."""
    by_context_group = {}
    for kind, hashes in (("seen", seen), ("unseen", unseen)):
        records = _records(config, groups, kind)
        for digest, ids in hashes["test"].items():
            for ep in ids:
                r = records[ep]
                by_context_group.setdefault((digest, r["group_id"]), []).append(
                    {"file": kind + ".h5", "episode": ep, "damping": r["damping"]})
    collisions = [{"group_id": group, "members": members}
                  for (_, group), members in by_context_group.items() if len(members) > 1]
    return {"collision_context_groups": len(collisions),
            "distinct_initial_groups_affected": len({c["group_id"] for c in collisions}),
            "examples": collisions[:20],
            "scope": "Paired held-out initial conditions across all seen/interpolation/extrapolation b; exact RGB history equality only, not approximate identifiability",
            "interpretation": "A deterministic history encoder must return the same context for identical input histories. Do not claim exact hidden-b identification from these histories."}


def generate_family(config, output):
    """Generate data only. No model, optimizer, fitting, evaluation, or training."""
    summary = plan(config)
    groups = sample_groups(config)
    output = Path(output)
    # Reserve a wholly new directory; a partial previous generation is not overwritten.
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: sha256(ROOT / name) for name in ("dpbench/passive_family.py", "dpbench/generate.py")}
    (output / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    seen, seen_hashes = _write_file(output / "seen.h5", config, groups, "seen")
    unseen, unseen_hashes = _write_file(output / "unseen.h5", config, groups, "unseen")
    manifest = {"status": "generated_untrained", "training_started": False,
                "config": config, "renderer": summary["renderer"], "source_sha256": sources, "plan": summary,
                "datasets": [seen, unseen], "initial_context_overlap": _context_overlap(seen_hashes, unseen_hashes),
                "cross_condition_identical_test_contexts": _cross_condition_ambiguity(config, groups, seen_hashes, unseen_hashes),
                "source": "Self-generated passive point-mass pendulum; existing project renderer and RK4 simulator, new stratified initial samples",
                "split_contract": "Initial groups split before damping expansion; unseen damping uses test groups only; seen/unseen test initial states deliberately paired",
                "metadata_boundary": "damping/damping_id/regime_id/group_id/energy_band/state/energy/work/dissipation are never world-model inputs or training targets; physical quantities are retained for independent evaluation",
                "test_policy": "No model selection or hyperparameter adjustment from seen-test or unseen-test results. Once used for design, regenerate a new predeclared final test seed.",
                "limitations": "Constant damping within each trajectory; no state-dependent physical friction, no forced trajectories. Context may not identify nearby damping values from quantized short video."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def generate_undamped(config, output):
    """Generate only the parent's b=0 episodes, with identical groups and clocks.

    The parent remains a fully validated multi-damping protocol. This selection
    happens during data preparation; damping is never routed to model batches.
    Existing test seeds are preserved for paired development diagnostics, not
    rebranded as a fresh final holdout after their use in model design.
    """
    summary = plan(config, undamped_only=True)  # Reject absent b=0 before writes.
    groups = sample_groups(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {name: sha256(ROOT / name) for name in ("dpbench/passive_family.py", "dpbench/generate.py")}
    (output / "generation_config.json").write_text(json.dumps(config, indent=2) + "\n")
    dataset, contexts = _write_file(output / "undamped.h5", config, groups, "seen", undamped_only=True)
    development = set(contexts["train"]) | set(contexts["val"])
    matches = lambda target, reference: sorted(
        ep for digest, ids in target.items() if digest in reference for ep in ids)
    manifest = {
        "status": "generated_untrained", "training_started": False,
        "selection_scope": "undamped_subset", "selected_damping": [0.0],
        "training_scope": "undamped_stage_a",
        "parent_config_sha256": summary["parent_config_sha256"], "config": config,
        "renderer": summary["renderer"], "source_sha256": sources, "plan": summary,
        "datasets": [dataset], "requires_environment_context": False,
        "split_contract": "All parent train/val/test initial states and group IDs retained; "
                          "only b=0 is simulated. No resampling or cross-split reassignment.",
        "metadata_boundary": "Only pixels/action/dt_frame enter the model. Damping/state/energy/group "
                             "metadata remain outside training targets; regime_id=0 is sampler metadata only.",
        "initial_context_overlap": {
            "val_matches_train": matches(contexts["val"], set(contexts["train"])),
            "test_matches_development": matches(contexts["test"], development),
            "scope": "Exact RGB equality in initial histories; report without filtering or resampling."},
        "test_policy": "Keep the parent test split for paired development diagnostics. Once used in design, "
                       "it is not a fresh final holdout; preregister a separate final test seed later.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--generate", action="store_true", help="Write new data files only; never starts training")
    parser.add_argument("--undamped-only", action="store_true",
                        help="Select only b=0 from the complete parent protocol; default still only plans")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.generate:
        if args.output is None:
            parser.error("--generate requires --output pointing to a new directory")
        result = generate_undamped(config, args.output) if args.undamped_only else generate_family(config, args.output)
    else:
        result = plan(config, undamped_only=args.undamped_only)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
