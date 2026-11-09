"""Fresh, test-only pendulum data for a hash-frozen model/readout cohort.

No model imports, checkpoint selection, fitting, or scoring occur here. Exact
overlap checks cover initial inference contexts and training-length prefixes;
they do not certify absence of similar states or all future-window overlaps.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .generate import energy, reference_audit, render, simulate


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    with Path(path).open("x") as f:
        f.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def verify_frozen(root, manifest):
    for run in manifest["runs"]:
        for item in run["files"]:
            path = root / manifest["frozen_directory"] / item["relative_path"]
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError(f"Frozen artifact changed: {path}")


def sample_plan(config):
    """Same distributions and draw order as generate.make_plan, without splits."""
    n, frames = config["episodes"], config["frames"]
    if n < 2 or n % 2 or frames < 15:
        raise ValueError("Need an even episode count and at least 15 frames")
    rng = np.random.default_rng(config["data_seed"])
    initial = np.column_stack((rng.uniform(-2.5, 2.5, n), rng.uniform(-2., 2., n)))
    modes = np.arange(n) % 2
    actions = np.zeros((n, frames - 1))
    hold = config["action_hold_frames"]
    for ep in range(n):
        if modes[ep]:
            draws = rng.uniform(-config["torque_max"], config["torque_max"], (frames - 2) // hold + 1)
            actions[ep] = np.repeat(draws, hold)[:frames - 1]
            phase = np.arange(frames - 1) % config["drive_cycle_frames"]
            lo, hi = config["drive_off_frames"]
            actions[ep, (phase >= lo) & (phase < hi)] = 0.
    return initial, actions, modes


def write_dataset(path, config, damping, initial, actions, modes, protocol_hash):
    start = time.perf_counter()
    kwargs = {k: config[k] for k in ("dt", "substeps", "mass", "length", "gravity")}
    state, work, diss = simulate(initial, actions, **kwargs, damping=damping)
    en = energy(state, config["mass"], config["length"], config["gravity"])
    residual = np.diff(en, axis=1) - work[:, :-1] + diss[:, :-1]
    audit = reference_audit(initial, actions, state, work, diss, config, damping)
    audit["energy_balance_max_abs"] = float(np.abs(residual).max())
    if not all(np.isfinite(x).all() for x in (state, en, work, diss, residual)):
        raise ValueError("Non-finite ground truth")
    if audit["energy_balance_max_abs"] > 1e-6:
        raise ValueError(f"Energy balance failed: {audit}")
    n, frames = state.shape[:2]
    with h5py.File(path, "x") as f:
        f.attrs.update({k: config[k] for k in ("mass", "length", "gravity", "dt", "data_seed")})
        f.attrs.update(schema_version=1, damping=damping, train_frames=0,
                       role="frozen_model_independent_test_only", protocol_sha256=protocol_hash,
                       config_json=json.dumps(config), audit_json=json.dumps(audit),
                       ode="m L^2 theta_ddot = -m g L sin(theta) - b omega + torque",
                       truth_integrator="float64 RK4 with integrated W,D; independent DOP853 audit")
        f.create_dataset("ep_len", data=np.full(n, frames, dtype=np.int64))
        f.create_dataset("ep_offset", data=np.arange(n, dtype=np.int64) * frames)
        f.create_dataset("group_id", data=config["group_id_offset"] + np.arange(n, dtype=np.int64))
        f.create_dataset("mode", data=modes)
        for split in ("train", "val"):
            f.create_dataset("splits/" + split, data=np.empty(0, dtype=np.int64))
        f.create_dataset("splits/test_all", data=np.arange(n, dtype=np.int64))
        for key, value in (("state", state), ("energy", en), ("work", work), ("dissipation", diss)):
            shape = (n * frames, 2) if key == "state" else (n * frames,)
            f.create_dataset(key, data=value.reshape(shape), compression="lzf")
        f.create_dataset("action", data=np.pad(actions, ((0, 0), (0, 1))).reshape(-1, 1).astype(np.float32))
        f.create_dataset("dt_frame", data=np.full((n * frames, 1), config["dt"], dtype=np.float32))
        valid = np.ones((n, frames), dtype=bool)
        valid[:, -1] = False
        f.create_dataset("valid_transition", data=valid.ravel())
        f.create_dataset("reference_residual", data=np.pad(residual, ((0, 0), (0, 1))).ravel())
        size = config["image_size"]
        pixels = f.create_dataset("pixels", (n * frames, size, size, 3), dtype="u1",
                                  chunks=(1, size, size, 3), compression="lzf")
        for ep in range(n):
            pixels[ep * frames:(ep + 1) * frames] = np.stack([render(th, size) for th in state[ep, :, 0]])
    return {"numerical_audit": audit, "generation_seconds": time.perf_counter() - start}


# Each specification is (pixel window length, action interval relative to that
# pixel window). LeHami's selected raw window is common frames 1..14, with
# transition actions common 4..11. LeWM selects common 0..12, actions 2..11.
SPECS = {"context_5": (5, None), "context_4": (4, None),
         "lewm_clip": (13, (2, 12)), "lehami_clip": (14, (3, 11))}
NEW_START = {"context_5": 0, "context_4": 1, "lewm_clip": 0, "lehami_clip": 1}


def frame_hashes(pixels):
    return [hashlib.sha256(np.ascontiguousarray(frame).tobytes()).digest() for frame in pixels]


def window_key(hashes, actions, start, spec):
    length, interval = SPECS[spec]
    if start + length > len(hashes):
        raise ValueError("Window crosses episode end")
    h = hashlib.sha256(b"".join(hashes[start:start + length]))
    if interval is not None:
        lo, hi = interval
        h.update(np.asarray(actions[start + lo:start + hi], dtype="<f4").tobytes())
    return h.digest()


def physical_key(state, actions=None):
    h = hashlib.sha256(np.asarray(state, dtype="<f8").tobytes())
    if actions is not None:
        h.update(np.asarray(actions, dtype="<f4").tobytes())
    return h.digest()


def overlap_audit(reference, candidate):
    """Conservative old pool: first train_frames of train/val, all old test."""
    indexes = {key: {} for key in SPECS}
    initial_index, trajectory_index = {}, {}
    with h5py.File(reference, "r") as old, h5py.File(candidate, "r") as new:
        # Also compare across damping variants: the old development pool is
        # the union of both datasets, even though each model trained on one.
        for key in ("dt", "mass", "length", "gravity"):
            if float(old.attrs[key]) != float(new.attrs[key]):
                raise ValueError(f"Incompatible reference {key}")
        for split in ("train", "val", "test"):
            for ep in old["splits/" + split][:]:
                off, total = int(old["ep_offset"][ep]), int(old["ep_len"][ep])
                n = total if split == "test" else min(total, int(old.attrs["train_frames"]))
                pixels = old["pixels"][off:off + n]
                actions = old["action"][off:off + total]
                state = old["state"][off:off + total]
                hashes = frame_hashes(pixels)
                witness = {"split": split, "episode": int(ep), "damping": float(old.attrs["damping"])}
                initial_index.setdefault(physical_key(state[0]), witness)
                trajectory_index.setdefault(physical_key(state, actions), witness)
                for spec, (length, _) in SPECS.items():
                    for start in range(n - length + 1):
                        indexes[spec].setdefault(window_key(hashes, actions, start, spec),
                                                 dict(witness, start=start))
        hits = {key: [] for key in (*SPECS, "initial_state", "full_trajectory", "new_initial_repeat")}
        seen_initial = {}
        for ep in new["splits/test_all"][:]:
            off, n = int(new["ep_offset"][ep]), int(new["ep_len"][ep])
            state, actions = new["state"][off:off + n], new["action"][off:off + n]
            hashes = frame_hashes(new["pixels"][off:off + 15])
            for key, index, value in [(key, indexes[key], window_key(hashes, actions, NEW_START[key], key)) for key in SPECS] + [
                    ("initial_state", initial_index, physical_key(state[0])),
                    ("full_trajectory", trajectory_index, physical_key(state, actions)),
                    ("new_initial_repeat", seen_initial, physical_key(state[0]))]:
                if value in index:
                    hits[key].append({"new_episode": int(ep), "witness": index[value]})
            seen_initial.setdefault(physical_key(state[0]), {"episode": int(ep)})
    return {"counts": {k: len(v) for k, v in hits.items()}, "hits": hits}


def paired_subsets(audits, episodes):
    clip_bad, context_bad = set(), set()
    for audit in audits:
        for spec in ("lewm_clip", "lehami_clip"):
            clip_bad.update(h["new_episode"] for h in audit["hits"][spec])
        for spec in ("context_4", "context_5"):
            context_bad.update(h["new_episode"] for h in audit["hits"][spec])
        if any(audit["counts"][k] for k in ("initial_state", "full_trajectory", "new_initial_repeat")):
            raise ValueError("Exact physical duplicate detected; do not silently regenerate")
    ids = set(range(episodes))
    return {"test": sorted(ids - clip_bad),
            "test_strict_context": sorted(ids - clip_bad - context_bad)}


def union_audits(audits):
    """Count distinct new episodes; retain one reproducible witness per hit."""
    hits = {}
    for key in audits[0]["hits"]:
        unique = {}
        for audit in audits:
            for hit in audit["hits"][key]:
                unique.setdefault(hit["new_episode"], hit)
        hits[key] = list(unique.values())
    return {"counts": {k: len(v) for k, v in hits.items()}, "hits": hits}


def distribution_summary(initial, modes, ids, config):
    ids = np.asarray(ids, dtype=np.int64)
    data = initial[ids]
    fields = {"abs_theta0_rad": np.abs(data[:, 0]), "abs_omega0_rad_s": np.abs(data[:, 1]),
              "energy0_J": energy(data, config["mass"], config["length"], config["gravity"])}
    return {"episodes": len(ids), "passive": int((modes[ids] == 0).sum()), "driven": int((modes[ids] == 1).sum()),
            "quantiles_min_q25_median_q75_max": {k: np.quantile(v, [0, .25, .5, .75, 1]).tolist() if len(v) else [] for k, v in fields.items()}}


def generate_holdout(root):
    root = Path(root).resolve()
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    frozen_path = root / "frozen_manifest.json"
    if sha256(frozen_path) != protocol["frozen_manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    for relative, digest in protocol["source_sha256"].items():
        if sha256(Path(__file__).resolve().parents[1] / relative) != digest:
            raise ValueError(f"Generation source changed: {relative}")
    frozen = json.loads(frozen_path.read_text())
    verify_frozen(root, frozen)
    config = protocol["generation"]
    for ref in protocol["references"]:
        if sha256(ref["path"]) != ref["sha256"]:
            raise ValueError("Old reference data changed")
        with h5py.File(ref["path"], "r") as f:
            if int(f.attrs["data_seed"]) == config["data_seed"]:
                raise ValueError("New and old seeds must differ")
            if set(f["group_id"][:]) & set(config["group_id_offset"] + np.arange(config["episodes"])):
                raise ValueError("Group IDs overlap")
    if {r["sha256"] for r in protocol["references"]} != {r["training_data_sha256"] for r in frozen["runs"]}:
        raise ValueError("Reference data do not match the frozen training cohort")
    output = root / "data"
    output.mkdir(exist_ok=False)  # Never overwrite even a partially generated holdout.
    initial, actions, modes = sample_plan(config)
    entries = []
    for damping in config["damping_values"]:
        path = output / ("pendulum_b" + str(damping).replace(".", "p") + ".h5")
        result = write_dataset(path, config, damping, initial, actions, modes, sha256(protocol_path))
        checks = [dict(reference_sha256=r["sha256"], reference_damping=r["damping"],
                       **overlap_audit(r["path"], path)) for r in protocol["references"]]
        audit = union_audits(checks)
        entries.append(dict(file=str(path.relative_to(root)), damping=damping,
                            overlap_by_reference=checks, overlap=audit, **result))
        print(json.dumps({"file": path.name, "overlap_counts": audit["counts"], **result}), flush=True)
    subsets = paired_subsets([e["overlap"] for e in entries], config["episodes"])
    if not subsets["test"]:
        raise ValueError("No primary holdout episodes remain")
    for entry in entries:
        path = root / entry["file"]
        with h5py.File(path, "r+") as f:
            for name, ids in subsets.items():
                f.create_dataset("splits/" + name, data=np.asarray(ids, dtype=np.int64))
        entry.update(sha256=sha256(path), bytes=path.stat().st_size)
    verify_frozen(root, frozen)
    report = {"status": "generated_audited_unscored", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha256(protocol_path), "frozen_manifest_sha256": sha256(frozen_path),
              "datasets": entries, "paired_splits": dict(test_all=list(range(config["episodes"])), **subsets),
              "distribution": {k: distribution_summary(initial, modes, ids, config) for k, ids in
                               dict(test_all=list(range(config["episodes"])), **subsets).items()},
              "limits": protocol["overlap_check_scope"]}
    save_json(root / "audit.json", report)
    print(json.dumps({"status": report["status"], "distribution": report["distribution"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    generate_holdout(parser.parse_args().study)
