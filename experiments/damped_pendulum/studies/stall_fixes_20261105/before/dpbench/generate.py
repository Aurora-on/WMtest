"""Point-mass pendulum, vectorized RK4 truth with integrated work/dissipation.

Independent of either learned model/integrator. A DOP853 reference audit is
performed before data are written. SI units; actions[t] act on [t,t+dt).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.integrate import solve_ivp


def energy(state, mass, length, gravity):
    return .5 * mass * length**2 * state[..., 1]**2 + mass * gravity * length * (1 - np.cos(state[..., 0]))


def derivative(y, torque, mass, length, gravity, damping):
    theta, omega = y[..., 0], y[..., 1]
    return np.stack((omega, -gravity / length * np.sin(theta) + (torque - damping * omega) / (mass * length**2),
                     torque * omega, damping * omega**2), axis=-1)


def simulate(initial, actions, dt, substeps, mass=1., length=1., gravity=9.81, damping=.2):
    initial = np.atleast_2d(np.asarray(initial, dtype=np.float64))
    actions = np.atleast_2d(np.asarray(actions, dtype=np.float64))
    n, transitions = actions.shape
    state = np.empty((n, transitions + 1, 2), dtype=np.float64)
    work, diss = np.zeros((n, transitions + 1)), np.zeros((n, transitions + 1))
    y = np.concatenate((initial, np.zeros((n, 2))), axis=-1)
    state[:, 0] = initial
    h = dt / substeps
    for t in range(transitions):
        y[:, 2:] = 0.
        u = actions[:, t]
        for _ in range(substeps):
            k1 = derivative(y, u, mass, length, gravity, damping)
            k2 = derivative(y + .5*h*k1, u, mass, length, gravity, damping)
            k3 = derivative(y + .5*h*k2, u, mass, length, gravity, damping)
            k4 = derivative(y + h*k3, u, mass, length, gravity, damping)
            y += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        state[:, t+1] = y[:, :2]
        work[:, t], diss[:, t] = y[:, 2], y[:, 3]
    return state, work, diss


def reference_audit(initial, actions, state, work, diss, config, damping):
    """Audit first free/driven examples over their full saved horizon."""
    errors = {"state_max_abs": 0., "work_max_abs": 0., "dissipation_max_abs": 0.}
    kwargs = {k: config[k] for k in ("mass", "length", "gravity")}
    dt = config["dt"]
    for ep in range(min(2, len(initial))):
        y = np.r_[initial[ep], 0., 0.]
        for t, u in enumerate(actions[ep]):
            y[2:] = 0.
            sol = solve_ivp(lambda _, z: derivative(z, u, **kwargs, damping=damping),
                            (0., dt), y, method="DOP853", rtol=1e-11, atol=1e-12)
            if not sol.success:
                raise RuntimeError(sol.message)
            y = sol.y[:, -1]
            errors["state_max_abs"] = max(errors["state_max_abs"], float(np.max(np.abs(y[:2] - state[ep, t+1]))))
            errors["work_max_abs"] = max(errors["work_max_abs"], float(abs(y[2] - work[ep, t])))
            errors["dissipation_max_abs"] = max(errors["dissipation_max_abs"], float(abs(y[3] - diss[ep, t])))
    errors["audited_episodes"] = min(2, len(initial))
    errors["reference"] = "scipy.solve_ivp DOP853 rtol=1e-11 atol=1e-12; full horizon"
    if any(errors[k] > 1e-5 for k in ("state_max_abs", "work_max_abs", "dissipation_max_abs")):
        raise RuntimeError(f"Numerical audit failed: {errors}; increase substeps")
    return errors


def render(theta, size=64):
    """LeHami-style fixed-camera RGB drawing; point-mass model stated explicitly."""
    im = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    pivot, radius = (size/2, size/2), size*.36
    x, y = pivot[0] + radius*np.sin(theta), pivot[1] + radius*np.cos(theta)
    draw.line([pivot, (x, y)], fill=(50, 50, 50), width=2)
    draw.ellipse([x-3, y-3, x+3, y+3], fill=(200, 40, 40))
    draw.ellipse([pivot[0]-2, pivot[1]-2, pivot[0]+2, pivot[1]+2], fill=(50, 50, 50))
    return np.asarray(im)


def make_plan(config):
    rng = np.random.default_rng(config["data_seed"])
    n, frames = config["episodes"], config["frames"]
    initial = np.column_stack((rng.uniform(-2.5, 2.5, n), rng.uniform(-2., 2., n)))
    modes = np.arange(n) % 2  # alternating passive and driven; stratified split
    actions = np.zeros((n, frames-1))
    hold = config["action_hold_frames"]
    for ep in range(n):
        if modes[ep]:
            draws = rng.uniform(-config["torque_max"], config["torque_max"], (frames-2)//hold+1)
            actions[ep] = np.repeat(draws, hold)[:frames-1]
            phase = np.arange(frames-1) % config["drive_cycle_frames"]
            lo, hi = config["drive_off_frames"]
            actions[ep, (phase >= lo) & (phase < hi)] = 0.
    split = {"train": [], "val": [], "test": []}
    for mode in (0, 1):
        ids = rng.permutation(np.where(modes == mode)[0])
        ntrain = int(len(ids)*config["train_fraction"])
        nval = int(len(ids)*config["val_fraction"])
        if min(ntrain, nval, len(ids)-ntrain-nval) < 1:
            raise ValueError("Each split needs at least one episode per action mode")
        split["train"].extend(ids[:ntrain].tolist())
        split["val"].extend(ids[ntrain:ntrain+nval].tolist())
        split["test"].extend(ids[ntrain+nval:].tolist())
    return initial, actions, modes, {k: sorted(v) for k, v in split.items()}


def generate(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.h5")) or (output/"manifest.json").exists():
        raise FileExistsError("Refusing to overwrite existing dataset; choose a new output folder")
    if not 15 <= config["train_frames"] <= config["frames"]:
        raise ValueError("Need 15 <= train_frames <= frames")
    initial, actions, modes, split = make_plan(config)
    manifest = {"schema_version": 1, "config": config, "splits": split, "datasets": [],
                "claim": "New controlled dataset; not an original LeHami or CpAE benchmark",
                "action_contract": "action[t] in N m acts on [t,t+dt); final padded row invalid",
                "model_inputs": ["pixels", "action", "dt_frame"],
                "split_contract": "same group IDs / initial states / action traces across damping variants; episode split before windows"}
    for damping in config["damping_values"]:
        start = time.perf_counter()
        kwargs = {k: config[k] for k in ("dt", "substeps", "mass", "length", "gravity")}
        state, work, diss = simulate(initial, actions, **kwargs, damping=damping)
        simulation_sec = time.perf_counter()-start
        en = energy(state, config["mass"], config["length"], config["gravity"])
        residual = np.diff(en, axis=1) - work[:, :-1] + diss[:, :-1]
        audit = reference_audit(initial, actions, state, work, diss, config, damping)
        audit["energy_balance_max_abs"] = float(np.max(np.abs(residual)))
        if audit["energy_balance_max_abs"] > 1e-6:
            raise RuntimeError(f"Reference balance residual too large: {audit}")
        n, frames = state.shape[:2]
        name = "pendulum_b" + str(damping).replace(".", "p") + ".h5"
        path = output/name
        with h5py.File(path, "w") as f:
            f.attrs.update({k: config[k] for k in ("mass", "length", "gravity", "dt", "train_frames", "data_seed")})
            f.attrs.update(damping=damping, schema_version=1, config_json=json.dumps(config),
                           ode="m L^2 theta_ddot = -m g L sin(theta) - b omega + torque",
                           truth_integrator="float64 RK4 with integrated W,D; independent DOP853 audit",
                           audit_json=json.dumps(audit))
            f.create_dataset("ep_len", data=np.full(n, frames, dtype=np.int64))
            f.create_dataset("ep_offset", data=np.arange(n, dtype=np.int64)*frames)
            f.create_dataset("group_id", data=np.arange(n, dtype=np.int64))
            f.create_dataset("mode", data=modes)
            for key, ids in split.items():
                f.create_dataset("splits/"+key, data=np.asarray(ids, dtype=np.int64))
            for key, value in (("state", state), ("energy", en), ("work", work), ("dissipation", diss)):
                shape = (n*frames, 2) if key == "state" else (n*frames,)
                f.create_dataset(key, data=value.reshape(shape), compression="lzf")
            padded = np.pad(actions, ((0, 0), (0, 1)))
            f.create_dataset("action", data=padded.reshape(-1, 1).astype(np.float32))
            f.create_dataset("dt_frame", data=np.full((n*frames, 1), config["dt"], dtype=np.float32))
            valid = np.ones((n, frames), dtype=bool)
            valid[:, -1] = False
            f.create_dataset("valid_transition", data=valid.ravel())
            f.create_dataset("reference_residual", data=np.pad(residual, ((0, 0), (0, 1))).ravel())
            pixels = f.create_dataset("pixels", (n*frames, config["image_size"], config["image_size"], 3),
                                      dtype="u1", chunks=(1, config["image_size"], config["image_size"], 3), compression="lzf")
            for ep in range(n):
                pixels[ep*frames:(ep+1)*frames] = np.stack([render(th, config["image_size"]) for th in state[ep, :, 0]])
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024*1024), b""):
                digest.update(block)
        result = {"file": name, "damping": damping, "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                  "simulation_seconds": simulation_sec, "generation_total_seconds": time.perf_counter()-start,
                  "frames": n*frames, "valid_transitions": n*(frames-1), "numerical_audit": audit}
        manifest["datasets"].append(result)
        print(json.dumps(result), flush=True)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    generate(json.loads(Path(a.config).read_text()), a.output)


if __name__ == "__main__":
    main()
