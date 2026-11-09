"""Frozen 21-frame passive-family evaluation with explicit failed-rollout coverage.

Physical labels train only diagnostic readouts, never the world model. Test
histories initialize once; future images and damping values never enter dynamics.
A failed strict solve remains a failed prediction (NaN suffix), never a new solver.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from dpbench.dissipative_core import SolverError
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.evaluate import pixels_tensor, targets, fit_readout, decode_state
from dpbench.generate import energy
from dpbench.passive_family import sha256
from dpbench.train_dissipative import source_hashes


EVALUATION_DIAGNOSTICS = (
    "solver_residual", "energy_residual", "dissipation", "solver_iterations",
    "work", "residual_work", "chain_defect", "chain_defect_max",
    "solver_residual_ratio", "solver_split_depth", "solver_leaf_steps",
    "solver_failed_attempts", "solver_backtracks",
)


def evaluation_source_hashes():
    root = Path(__file__).resolve().parents[1]
    names = ("dpbench/evaluate_passive.py", "dpbench/evaluate.py", "dpbench/models.py",
             "dpbench/data.py", "dpbench/generate.py", "dpbench/passive_family.py")
    return {name: sha256(root / name) for name in names}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def calibration_pairs(path, split, history, cap, maximum, seed):
    if split not in {"train", "val"}:
        raise ValueError("Physical calibration can only use train or val")
    with h5py.File(path, "r") as f:
        pairs = [(int(ep), t) for ep in f["splits/" + split][:]
                 for t in range(history - 1, min(cap, int(f["ep_len"][ep])))]
    if not pairs:
        raise ValueError(f"No calibration data in {split}")
    chosen = np.random.default_rng(seed).choice(len(pairs), min(maximum, len(pairs)), replace=False)
    return [pairs[i] for i in sorted(chosen)]


@torch.no_grad()
def calibration_features(model, path, pairs, batch_size):
    device = next(model.parameters()).device
    features, labels = [], []
    with h5py.File(path, "r") as f:
        for start in range(0, len(pairs), batch_size):
            images, actions, intervals, y = [], [], [], []
            for ep, anchor in pairs[start:start + batch_size]:
                offset = int(f["ep_offset"][ep])
                lo, hi = offset + anchor - model.history + 1, offset + anchor + 1
                images.append(pixels_tensor(f["pixels"][lo:hi], "cpu"))
                actions.append(f["action"][lo:hi - 1])
                intervals.append(f["dt_frame"][lo:hi - 1])
                y.append(targets(f["state"][hi - 1:hi])[0])
            q, p, _, _ = model.initialize(torch.stack(images).to(device),
                torch.as_tensor(np.stack(actions), device=device),
                torch.as_tensor(np.stack(intervals), device=device))
            features.append(torch.cat((q, p), -1).cpu().numpy())
            labels.extend(y)
    return np.concatenate(features), np.asarray(labels)


def _safe_step(core, q, p, action, dt, context, ids, step, failures, diagnostic):
    """Recursively isolate a strict failure; never change dt/tolerance/integrator."""
    try:
        nq, np_, info = core(q, p, action, dt, context=context)
        if not (torch.isfinite(nq).all() and torch.isfinite(np_).all()):
            raise FloatingPointError("Nonfinite next state")
        diagnostic.append((ids, {k: v.detach().cpu().numpy() for k, v in info.items()}))
        return [(ids, nq.detach(), np_.detach())]
    except (SolverError, FloatingPointError) as exc:
        if len(ids) == 1:
            failures.append({"batch_index": int(ids[0]), "failed_step": step,
                             "error_type": type(exc).__name__, "error": str(exc)})
            return []
        middle = len(ids) // 2
        result = []
        for s in (slice(None, middle), slice(middle, None)):
            result += _safe_step(core, q[s], p[s], action[s], dt[s],
                None if context is None else context[s], ids[s], step, failures, diagnostic)
        return result


@torch.no_grad()
def autonomous_rollout(model, history, past_actions, past_dt, actions, future_dt, diagnostic_names=None):
    """Return all valid prefixes; no future observations are accepted by this API."""
    q, p, encoded, _ = model.initialize(history, past_actions, past_dt)
    context = model.infer_context(encoded, past_actions, past_dt)
    batch, steps = actions.shape[:2]
    features = np.full((batch, steps + 1, 2 * model.q_dim), np.nan)
    storage = np.full((batch, steps + 1), np.nan)
    diagnostics = {name: np.full((batch, steps), np.nan) for name in (diagnostic_names or
                   ("solver_residual", "energy_residual", "dissipation", "solver_iterations"))}
    valid_steps = np.zeros(batch, dtype=np.int64)
    failures, active = [], np.arange(batch)
    features[:, 0] = torch.cat((q, p), -1).cpu().numpy()
    storage[:, 0] = model.core.energy(q, p).cpu().numpy()
    for step in range(1, steps + 1):
        if not len(active):
            break
        pieces = []
        results = _safe_step(model.core, q, p, actions[active, step - 1],
            future_dt[active, step - 1], None if context is None else context[active],
            active, step, failures, pieces)
        for ids, info in pieces:
            for key in diagnostics:
                diagnostics[key][ids, step - 1] = info[key]
        if not results:
            break
        active = np.concatenate([x[0] for x in results])
        q, p = torch.cat([x[1] for x in results]), torch.cat([x[2] for x in results])
        features[active, step] = torch.cat((q, p), -1).cpu().numpy()
        storage[active, step] = model.core.energy(q, p).cpu().numpy()
        valid_steps[active] = step
    return dict(features=features, latent_energy=storage, diagnostics=diagnostics,
                context=None if context is None else context.cpu().numpy(),
                valid_steps=valid_steps, failures=failures)


def dataset_records(path, file_key, history, steps):
    records = []
    with h5py.File(path, "r") as f:
        family_for_ep = {}
        for family in f["evaluation"]:
            family_for_ep.update({int(ep): family for ep in f["evaluation/" + family][:]})
        for ep in f["splits/test"][:]:
            ep = int(ep)
            if int(f["ep_len"][ep]) < history + steps:
                raise ValueError(f"Episode {file_key}/{ep} too short")
            records.append(dict(file=file_key, episode=ep, family=family_for_ep[ep],
                damping=float(f["damping"][ep]), energy_band=int(f["energy_band"][ep]),
                group_id=int(f["group_id"][ep])))
    return records


def _group_masks(records):
    n = len(records)
    yield {"grouping": "all", "group": "all"}, np.ones(n, dtype=bool)
    for keys in (("family",), ("damping",), ("energy_band",), ("family", "energy_band"), ("damping", "energy_band")):
        values = sorted(set(tuple(r[k] for k in keys) for r in records))
        for value in values:
            yield {"grouping": "+".join(keys), **dict(zip(keys, value))}, np.asarray([
                tuple(r[k] for k in keys) == value for r in records])


def baseline_predictions(model_prediction, truth):
    """Keep the decoded anchor; a fixed future frame has zero future velocity."""
    persistence = np.repeat(model_prediction[:, :1], model_prediction.shape[1], axis=1)
    fixed_frame = persistence.copy()
    fixed_frame[:, 1:, 1] = 0.
    return {"model": model_prediction, "persistence": persistence,
            "fixed_frame": fixed_frame, "equilibrium": np.zeros_like(truth),
            "oracle_persistence": np.repeat(truth[:, :1], truth.shape[1], axis=1)}


def numerical_audit(storage, diagnostics, valid_steps):
    """Recompute the latent ledger on accepted prefixes, keeping failures visible.

    This is a numerical identity check, not a physical-energy accuracy test or
    an assertion that every finite-precision step decreases latent energy.
    """
    delta = np.diff(storage, axis=1)
    ledger_error = (delta - diagnostics["work"] + diagnostics["dissipation"]
                    - diagnostics["residual_work"] - diagnostics["chain_defect"])
    valid = np.arange(delta.shape[1])[None, :] < np.asarray(valid_steps)[:, None]
    finite = np.isfinite(storage[:, :-1]) & np.isfinite(storage[:, 1:])
    for value in diagnostics.values():
        finite &= np.isfinite(value)
    accepted = valid & finite
    scale = (1 + abs(storage[:, :-1]) + abs(storage[:, 1:])
             + abs(diagnostics["work"]) + abs(diagnostics["dissipation"])
             + abs(diagnostics["residual_work"]) + abs(diagnostics["chain_defect"]))
    count = int(accepted.sum())
    nonfinite_count = int((valid & ~finite).sum())
    violations = {
        "ledger_identity_violation_count": int((accepted & (abs(ledger_error) > 1e-10 * scale)).sum()),
        "negative_dissipation_violation_count": int((accepted & (diagnostics["dissipation"] < -1e-12 * scale)).sum()),
        "nonzero_passive_work_violation_count": int((accepted & (abs(diagnostics["work"]) > 1e-12 * scale)).sum()),
        "residual_ratio_violation_count": int((accepted & (diagnostics["solver_residual_ratio"] > 1 + 1e-10)).sum()),
    }
    def maximum(values):
        return float(np.max(values[accepted])) if count else None
    summary = dict(
        scope="Latent numerical balance on accepted prefixes only; not physical prediction accuracy",
        declared_valid_transitions=int(valid.sum()), audited_finite_transitions=count,
        invalid_prefix_diagnostic_count=nonfinite_count,
        passed_on_valid_prefixes=bool(count and not nonfinite_count and not any(violations.values())),
        max_abs_ledger_identity_error=maximum(abs(ledger_error)),
        max_scaled_ledger_identity_error=maximum(abs(ledger_error) / scale),
        minimum_dissipation=float(np.min(diagnostics["dissipation"][accepted])) if count else None,
        max_solver_residual_ratio=maximum(diagnostics["solver_residual_ratio"]),
        max_split_depth=maximum(diagnostics["solver_split_depth"]),
        latent_energy_increase_count=int((accepted & (delta > 1e-10 * scale)).sum()),
        ledger_scaled_tolerance=1e-10, passive_scaled_tolerance=1e-12,
        **violations,
    )
    return summary, ledger_error


def summary_metrics(predictions, truth, initial_energy, records, horizons, dt, physical, valid_steps):
    """Coverage plus pooled trajectory/time RMSE; matched-support baselines also saved."""
    rows = []
    true_energy = energy(truth, **physical)
    for horizon in horizons:
        end = round(horizon / dt)
        for method, pred in predictions.items():
            angle = np.arctan2(np.sin(pred[:, :end + 1, 0] - truth[:, :end + 1, 0]),
                               np.cos(pred[:, :end + 1, 0] - truth[:, :end + 1, 0]))
            omega = pred[:, :end + 1, 1] - truth[:, :end + 1, 1]
            pe = energy(pred[:, :end + 1], **physical)
            ediff = pe - true_energy[:, :end + 1]
            normalized_energy = ediff / initial_energy[:, None]
            good = np.isfinite(pred[:, :end + 1]).all(axis=(1, 2))
            for support in ("all_available", "model_complete"):
                eligible = good & ((valid_steps >= end) if support == "model_complete" else True)
                for description, mask in _group_masks(records):
                    selected = mask & eligible
                    count, total = int(selected.sum()), int(mask.sum())
                    row = dict(method=method, horizon_s=horizon, support=support, **description,
                               n_total=total, n_valid=count, coverage=count / total)
                    if count:
                        a, w, e, en = angle[selected], omega[selected], ediff[selected], normalized_energy[selected]
                        row.update(angle_rmse_rad=float(np.sqrt(np.mean(a[:, 1:] ** 2))),
                            omega_rmse_rad_s=float(np.sqrt(np.mean(w[:, 1:] ** 2))),
                            energy_rmse_J=float(np.sqrt(np.mean(e[:, 1:] ** 2))),
                            energy_normalized_rmse=float(np.sqrt(np.mean(en[:, 1:] ** 2))),
                            endpoint_angle_mae_rad=float(np.mean(abs(a[:, -1]))),
                            endpoint_omega_mae_rad_s=float(np.mean(abs(w[:, -1]))),
                            endpoint_energy_normalized_mae=float(np.mean(abs(en[:, -1]))),
                            anchor_angle_mae_rad=float(np.mean(abs(a[:, 0]))),
                            anchor_omega_mae_rad_s=float(np.mean(abs(w[:, 0]))),
                            physical_energy_increase_fraction=float(np.mean(np.diff(pe[selected], axis=1) > 1e-5)))
                    rows.append(row)
    return rows


def evaluate(args):
    torch.set_num_threads(args.threads)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Evaluation output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["kind"] != "dissipative":
        raise ValueError("Expected dissipative checkpoint")
    if checkpoint["dataset_sha256"] != sha256(args.seen):
        raise ValueError("Seen dataset does not match checkpoint SHA256")
    for name, expected in checkpoint["source_hashes"].items():
        if source_hashes().get(name) != expected:
            raise ValueError(f"Original model/trainer source changed: {name}")
    model = DissipativeWorldModel(checkpoint["config"]).to(args.device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dt = model.dt
    steps = round(max(args.horizons) / dt)
    if any(h <= 0 or not np.isclose(round(h / dt) * dt, h, atol=1e-9) for h in args.horizons):
        raise ValueError("Horizons must be positive whole frame intervals")
    with h5py.File(args.seen, "r") as seen, h5py.File(args.unseen, "r") as unseen:
        physical = {key: float(seen.attrs[key]) for key in ("mass", "length", "gravity")}
        for f in (seen, unseen):
            if int(f.attrs["history_required"]) != model.history or not np.isclose(f.attrs["dt"], dt):
                raise ValueError("History or frame interval differs from model")
            if any(float(f.attrs[k]) != physical[k] for k in physical):
                raise ValueError("Physical parameters differ across files")
        if len(unseen["splits/train"]) or len(unseen["splits/val"]):
            raise ValueError("Unseen file must be test-only")
        groups = [set(seen["group_id"][:][seen["splits/" + s][:]].tolist()) for s in ("train", "val", "test")]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Initial-condition groups cross seen splits")
        if not set(unseen["group_id"][:].tolist()) <= groups[2]:
            raise ValueError("Unseen test groups must pair with seen test")
    manifest = dict(status="calibrating", checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha256(args.checkpoint), checkpoint_step=checkpoint["step"],
        dataset_sha256={"seen": sha256(args.seen), "unseen": sha256(args.unseen)},
        evaluator_sha256=sha256(__file__), original_source_hashes=checkpoint["source_hashes"],
        evaluation_source_hashes=evaluation_source_hashes(),
        history=model.history, anchor_frame=model.history - 1, anchor_time_s=(model.history - 1) * dt,
        dt=dt, horizons_s=args.horizons, physical_parameters=physical,
        model_frozen=True, labels_scope="Seen train only for diagnostic readout; seen val for readout selection; no test fitting",
        failure_scope="Strict solver failures retain finite prefixes and NaN suffixes; no dt/tolerance/integrator change. Failed batches are recursively split to isolate individual failures.",
        seed=args.seed, probe_steps=args.probe_steps, batch_size=args.batch_size, device=args.device)
    write_json(args.output / "manifest.json", manifest)
    pairs = {split: calibration_pairs(args.seen, split, model.history, checkpoint["train_frames"], maximum, args.seed)
             for split, maximum in (("train", args.labels), ("val", args.val_labels))}
    x, y = calibration_features(model, args.seen, pairs["train"], args.batch_size)
    xv, yv = calibration_features(model, args.seen, pairs["val"], args.batch_size)
    probes, infos = {}, {}
    for mode in ("ridge", "mlp"):
        probes[mode], infos[mode] = fit_readout(x, y, xv, yv, mode, args.probe_steps, args.seed)
        decoded, truth = decode_state(probes[mode].predict(xv)), decode_state(yv)
        angle = np.arctan2(np.sin(decoded[:, 0] - truth[:, 0]), np.cos(decoded[:, 0] - truth[:, 0]))
        infos[mode].update(validation_angle_mae_rad=float(np.mean(abs(angle))),
                           validation_omega_mae_rad_s=float(np.mean(abs(decoded[:, 1] - truth[:, 1]))))
    selected = min(infos, key=lambda k: infos[k]["validation_standardized_mse"])
    write_json(args.output / "calibration.json", dict(pairs=pairs, probes=infos, selected=selected))
    torch.save({k: v.serialize() for k, v in probes.items()}, args.output / "readouts.pt")
    print(json.dumps({"phase": "readout_ready", "selected": selected, "probes": infos}), flush=True)
    paths = {"seen": args.seen, "unseen": args.unseen}
    records = sum([dataset_records(p, key, model.history, steps) for key, p in paths.items()], [])
    all_features, all_truth, all_energy0, all_storage, all_context, all_valid, all_failures = [], [], [], [], [], [], []
    all_diag = {}
    device = next(model.parameters()).device
    for file_key, path in paths.items():
        selected_records = [r for r in records if r["file"] == file_key]
        with h5py.File(path, "r") as f:
            for offset in range(0, len(selected_records), args.batch_size):
                chunk = selected_records[offset:offset + args.batch_size]
                images, past_actions, past_dt, actions, future_dt, truth, energy0 = [], [], [], [], [], [], []
                for r in chunk:
                    off = int(f["ep_offset"][r["episode"]]); anchor = off + model.history - 1
                    images.append(pixels_tensor(f["pixels"][off:anchor + 1], "cpu"))
                    past_actions.append(f["action"][off:anchor]); past_dt.append(f["dt_frame"][off:anchor])
                    actions.append(f["action"][anchor:anchor + steps]); future_dt.append(f["dt_frame"][anchor:anchor + steps])
                    truth.append(f["state"][anchor:anchor + steps + 1]); energy0.append(float(f["energy"][off]))
                if np.any(actions) or np.any(past_actions):
                    raise ValueError("This evaluator requires zero external forces")
                arrays = [torch.stack(images).to(device)] + [torch.as_tensor(np.stack(a), device=device)
                           for a in (past_actions, past_dt, actions, future_dt)]
                out = autonomous_rollout(model, *arrays, diagnostic_names=EVALUATION_DIAGNOSTICS)
                all_features.append(out["features"]); all_truth.append(np.stack(truth)); all_energy0.extend(energy0)
                all_storage.append(out["latent_energy"]); all_valid.append(out["valid_steps"])
                if out["context"] is not None:
                    all_context.append(out["context"])
                for k, v in out["diagnostics"].items():
                    all_diag.setdefault(k, []).append(v)
                for fail in out["failures"]:
                    all_failures.append({**chunk[fail["batch_index"]], **fail})
                print(json.dumps({"phase": "test", "file": file_key, "episodes_done": offset + len(chunk),
                    "file_episodes": len(selected_records), "failed_in_batch": len(out["failures"]),
                    "elapsed_seconds": time.perf_counter() - started}), flush=True)
                # Keep all completed batches recoverable even if a later unexpected failure occurs.
                np.savez_compressed(args.output / f"prefix_{file_key}_{offset:04d}.npz", **{
                    "features":out["features"], "truth":np.stack(truth), "valid_steps":out["valid_steps"],
                    "latent_energy":out["latent_energy"],
                    "episode":np.asarray([r["episode"] for r in chunk]), **out["diagnostics"]})
                write_json(args.output / "failures.json", all_failures)
    features, truth = np.concatenate(all_features), np.concatenate(all_truth)
    valid_steps, energy0 = np.concatenate(all_valid), np.asarray(all_energy0)
    predictions = baseline_predictions(decode_state(probes[selected].predict(features)), truth)
    storage = np.concatenate(all_storage)
    diagnostics = {key: np.concatenate(values) for key, values in all_diag.items()}
    audit, ledger_error = numerical_audit(storage, diagnostics, valid_steps)
    write_json(args.output / "numerical_audit.json", audit)
    metrics = summary_metrics(predictions, truth, energy0, records, args.horizons, dt, physical, valid_steps)
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "episodes.json", records)
    np.savez_compressed(args.output / "trajectories.npz", features=features, truth=truth,
        **{f"prediction_{k}":v for k,v in predictions.items()}, initial_energy=energy0,
        valid_steps=valid_steps, latent_energy=storage, ledger_identity_error=ledger_error,
        context=np.concatenate(all_context) if all_context else np.empty((len(records),0)),
        time=np.arange(steps+1)*dt, **diagnostics)
    manifest.update(status="completed", selected_readout=selected, episodes=len(records),
        completed_full_rollouts=int(np.sum(valid_steps == steps)), failed_rollouts=len(all_failures),
        readout_sha256=sha256(args.output / "readouts.pt"), elapsed_seconds=time.perf_counter()-started,
        numerical_audit=audit,
        metric_scope="Physical predictions after frozen readout. Model/latent structural balance is a separate diagnostic. Failures excluded from finite-error summaries and explicitly counted; compare model_complete support for matched baselines.",
        methods={"model":"Frozen checkpoint plus val-selected physical readout", "persistence":"Hold decoded last observed state (same visual initialization as model)",
                 "fixed_frame":"Keep decoded anchor at t=0; hold decoded angle and set omega=0 at all future steps",
                 "equilibrium":"theta=omega=0, known physical equilibrium", "oracle_persistence":"Hold true anchor state; privileged-label diagnostic, not an image-input baseline"})
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"phase":"completed", "episodes":len(records), "failed_rollouts":len(all_failures), "seconds":manifest["elapsed_seconds"]}), flush=True)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seen", type=Path, required=True)
    p.add_argument("--unseen", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--labels", type=int, default=20000)
    p.add_argument("--val-labels", type=int, default=4000)
    p.add_argument("--probe-steps", type=int, default=2000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizons", type=float, nargs="+", default=[1.,2.,4.,8.,16.,32.])
    args = p.parse_args(argv)
    if min(args.batch_size, args.labels, args.val_labels, args.probe_steps, args.threads) < 1:
        p.error("Budgets and batch size must be positive")
    return evaluate(args)


if __name__ == "__main__":
    main()
