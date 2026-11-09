"""Score hash-frozen models/readouts on a separately bound independent test set.

Prediction reads only five initial RGB frames and exogenous controls. Ground
truth is read separately after prediction. There is no fitting/optimizer path.
"""
import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .evaluate import Readout, decode_state, metrics, pixels_tensor
from .holdout import save_json, sha256, verify_frozen
from .models import load_checkpoint, synchronize


def tensor_digest(tree):
    h = hashlib.sha256()
    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value):
                h.update(str(key).encode()); visit(value[key])
        elif isinstance(value, torch.Tensor):
            x = value.detach().cpu().contiguous()
            h.update(str((x.dtype, tuple(x.shape))).encode()); h.update(x.numpy().tobytes())
        elif value is not None:
            h.update(str(value).encode())
    visit(tree)
    return h.hexdigest()


def load_readouts(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if set(saved) != {"ridge", "mlp"}:
        raise ValueError("Both original readouts must be present")
    probes = {}
    for mode in ("ridge", "mlp"):
        s = saved[mode]
        if s["mode"] != mode:
            raise ValueError("Readout mode mismatch")
        arrays = {k: s[k].numpy().copy() for k in ("mean", "scale", "ymean", "yscale")}
        if not all(np.isfinite(v).all() for v in arrays.values()) or any((arrays[k] <= 0).any() for k in ("scale", "yscale")):
            raise ValueError("Invalid frozen readout normalization")
        network, weights = None, None
        if mode == "mlp":
            network = torch.nn.Sequential(torch.nn.Linear(len(arrays["mean"]), 128), torch.nn.Tanh(),
                                          torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 3))
            network.load_state_dict(s["network"], strict=True)
            network.eval().requires_grad_(False)
        else:
            weights = s["weights"].numpy().copy()
        probes[mode] = Readout(mode=mode, network=network, weights=weights, **arrays)
    return probes


def input_batch(path, ids, steps):
    """Restricted input boundary: never reads future pixels or physical labels."""
    histories, past, actions = [], [], []
    with h5py.File(path, "r") as f:
        for ep in ids:
            off, n = int(f["ep_offset"][ep]), int(f["ep_len"][ep])
            if steps < 1 or steps + 5 > n:
                raise ValueError("Requested rollout exceeds the available episode")
            histories.append(f["pixels"][off:off + 5])
            past.append(f["action"][off:off + 4])
            actions.append(f["action"][off + 4:off + 4 + steps])
    return np.stack(histories), np.stack(past), np.stack(actions)


def predict(model, probes, inputs, device):
    history, past, actions = inputs
    history = torch.stack([pixels_tensor(p, device) for p in history])
    z = model.rollout_features(history, torch.as_tensor(actions, device=device),
                               past_actions=torch.as_tensor(past, device=device)).cpu().numpy()
    predictions = np.stack([decode_state(probe.predict(z)) for probe in probes.values()], axis=1)
    if not np.isfinite(z).all() or not np.isfinite(predictions).all():
        raise FloatingPointError("Non-finite rollout or frozen readout")
    return z, predictions


def canary_check(model, probes, data, ids, steps, device):
    """Real checkpoint: overwrite future imagery and ALL physical labels in a copy."""
    ids = list(ids)
    inputs = input_batch(data, ids, steps)
    before = predict(model, probes, inputs, device)
    with tempfile.TemporaryDirectory(prefix="holdout_canary_") as temporary:
        changed = Path(temporary) / "canary.h5"
        shutil.copyfile(data, changed)
        with h5py.File(changed, "r+") as f:
            for ep in ids:
                off, n = int(f["ep_offset"][ep]), int(f["ep_len"][ep])
                f["pixels"][off + 5:off + n] = 17
                f["state"][off:off + n] += 37.
                for key in ("energy", "work", "dissipation"):
                    f[key][off:off + n] = 999.
        altered = input_batch(changed, ids, steps)
        for left, right in zip(inputs, altered):
            np.testing.assert_array_equal(left, right)
        after = predict(model, probes, altered, device)
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)
        assert sha256(data) != sha256(changed)
    return {"episodes": ids, "steps": steps, "future_rgb_and_labels_replaced": True,
            "features_max_abs_difference": 0., "predictions_max_abs_difference": 0.}


def aggregate(rows, subsets):
    result = []
    for split, ids in subsets.items():
        ids = set(ids)
        for probe in ("ridge", "mlp"):
            for mode in ("passive", "driven"):
                for horizon in sorted({r["horizon_s"] for r in rows}):
                    chosen = [r for r in rows if r["episode"] in ids and
                              (r["probe"], r["mode"], r["horizon_s"]) == (probe, mode, horizon)]
                    if not chosen:
                        raise ValueError(f"Missing evaluation cell: {split}, {probe}, {mode}, {horizon}")
                    keys = [k for k in chosen[0] if k not in ("episode", "probe", "mode", "horizon_s") and chosen[0][k] is not None]
                    result.append(dict(split=split, probe=probe, mode=mode, horizon_s=horizon, episodes=len(chosen),
                                       mean={k: float(np.mean([r[k] for r in chosen])) for k in keys},
                                       episode_std={k: float(np.std([r[k] for r in chosen])) for k in keys}))
    return result


def evaluate_run(root, run, data_entry, output, device, batch_size, horizons):
    started = time.perf_counter()
    path = root / "frozen" / "runs/server_47111" / run["run"]
    model, checkpoint = load_checkpoint(path / "best.pt", device)
    model.requires_grad_(False)
    if (checkpoint["kind"], checkpoint["seed"], checkpoint["step"], checkpoint["dataset_sha256"]) != (
            run["model"], run["seed"], run["checkpoint_step"], run["training_data_sha256"]):
        raise ValueError("Original checkpoint provenance mismatch")
    probes = load_readouts(path / "evaluation/readouts.pt")
    original_model = tensor_digest(model.state_dict())
    original_probes = tensor_digest({k: p.serialize() for k, p in probes.items()})
    data = root / data_entry["file"]
    with h5py.File(data, "r") as f:
        if len(f["splits/train"]) or len(f["splits/val"]):
            raise ValueError("New data must be test-only")
        meta = {k: float(f.attrs[k]) for k in ("dt", "mass", "length", "gravity", "damping")}
        if meta["damping"] != run["damping"]:
            raise ValueError("Damping mismatch")
        subsets = {k: f["splits/" + k][:].tolist() for k in ("test", "test_all", "test_strict_context")}
        modes = f["mode"][:]
    if not all(h > 0 and np.isclose(round(h/meta["dt"])*meta["dt"], h) for h in horizons):
        raise ValueError("Invalid prediction horizons")
    steps, ids = round(max(horizons)/meta["dt"]), subsets["test_all"]
    canary_ids = [next(ep for ep in subsets["test"] if modes[ep] == mode) for mode in (0, 1)]
    canary = canary_check(model, probes, data, canary_ids, steps, device)
    # Compare proposed batching with original one-episode evaluation over the
    # full horizon. Fall back to batch=1 if differences exceed fixed tolerances.
    pilot_ids = ids[:batch_size]
    pilot = predict(model, probes, input_batch(data, pilot_ids, steps), device)
    diffs = []
    for i, ep in enumerate(pilot_ids[:2]):
        single = predict(model, probes, input_batch(data, [ep], steps), device)[1][0]
        difference = pilot[1][i] - single
        angle = np.arctan2(np.sin(difference[..., 0]), np.cos(difference[..., 0]))
        diffs.append((float(np.abs(angle).max()), float(np.abs(difference[..., 1]).max())))
    max_angle, max_omega = np.max(diffs, axis=0).tolist()
    actual_batch = batch_size if max_angle <= .001 and max_omega <= .01 else 1
    batch_check = dict(compared_episodes=pilot_ids[:2], proposed_batch=batch_size, actual_batch=actual_batch,
                       angle_max_abs_rad=max_angle, omega_max_abs_rad_s=max_omega,
                       tolerance_angle_rad=.001, tolerance_omega_rad_s=.01)
    predictions, features = [], []
    synchronize(device); inference_start = time.perf_counter()
    for i in range(0, len(ids), actual_batch):
        z, pred = predict(model, probes, input_batch(data, ids[i:i + actual_batch], steps), device)
        features.append(z); predictions.append(pred)
    synchronize(device); inference_seconds = time.perf_counter() - inference_start
    features, predictions = np.concatenate(features), np.concatenate(predictions)
    # The prediction phase has completed. Only now read labels for scoring.
    rows, truth_all = [], []
    with h5py.File(data, "r") as f:
        for i, ep in enumerate(ids):
            off = int(f["ep_offset"][ep]) + 4
            truth = f["state"][off:off + steps + 1]
            work, diss = f["work"][off:off + steps + 1], f["dissipation"][off:off + steps + 1]
            truth_all.append(truth)
            for j, probe in enumerate(probes):
                for h in horizons:
                    end = round(h/meta["dt"]) + 1
                    row = metrics(predictions[i, j, :end], truth[:end], work[:end], diss[:end], meta, modes[ep] == 0)
                    rows.append(dict(episode=int(ep), probe=probe, mode="passive" if modes[ep] == 0 else "driven", horizon_s=h, **row))
    if original_model != tensor_digest(model.state_dict()) or original_probes != tensor_digest({k: p.serialize() for k, p in probes.items()}):
        raise ValueError("Parameters/buffers/readouts changed during evaluation")
    assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    result = dict(run=run["run"], model=run["model"], damping=run["damping"], seed=run["seed"],
                  checkpoint_step=checkpoint["step"], training_data_sha256=checkpoint["dataset_sha256"],
                  test_data_sha256=data_entry["sha256"], frozen_model_digest=original_model,
                  frozen_readouts_digest=original_probes, canary=canary, batch_check=batch_check,
                  parameters_and_buffers_unchanged=True, readouts_refitted=False, probe_names=list(probes),
                  aggregate=aggregate(rows, subsets), episodes=rows, inference_seconds=inference_seconds,
                  total_seconds=time.perf_counter() - started)
    output.mkdir()
    np.savez_compressed(output / "trajectories.npz", features=features, predictions=predictions,
                        truth=np.stack(truth_all), episode_ids=np.asarray(ids), probe_names=np.asarray(list(probes)), dt=meta["dt"])
    save_json(output / "metrics.json", result)
    print(json.dumps({k: result[k] for k in ("run", "total_seconds", "inference_seconds", "canary", "batch_check")}), flush=True)
    return result


def run_matrix(args):
    torch.set_num_threads(args.threads)
    root, output = args.study.resolve(), args.output.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    audit = json.loads((root / "audit.json").read_text())
    frozen = json.loads((root / "frozen_manifest.json").read_text())
    if sha256(root / "protocol.json") != audit["protocol_sha256"] or sha256(root / "frozen_manifest.json") != protocol["frozen_manifest_sha256"]:
        raise ValueError("Sealed protocol/manifest changed")
    verify_frozen(root, frozen)
    for relative, digest in protocol["source_sha256"].items():
        if sha256(Path(__file__).resolve().parents[1] / relative) != digest:
            raise ValueError(f"Original source changed: {relative}")
    for entry in audit["datasets"]:
        if sha256(root / entry["file"]) != entry["sha256"]:
            raise ValueError("Independent test data changed")
    expected = {(model, b, seed) for model in ("lehami", "lewm") for b in (0., .2) for seed in (0, 1, 2)}
    if {(r["model"], r["damping"], r["seed"]) for r in frozen["runs"]} != expected or len(frozen["runs"]) != 12:
        raise ValueError("Incomplete frozen cohort")
    output.mkdir(parents=True, exist_ok=False)
    runtime = dict(device=args.device, torch_version=str(torch.__version__), threads=args.threads,
                   gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                   cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                   runner_sha256=sha256(__file__), protocol_sha256=sha256(root / "protocol.json"),
                   audit_sha256=sha256(root / "audit.json"), requested_batch_size=args.batch_size,
                   horizons=args.horizons, probes=["ridge", "mlp"], model_updates=False, readout_fitting=False)
    save_json(output / "evaluation_protocol.json", runtime)
    results = []
    for run in frozen["runs"]:
        entry = next(e for e in audit["datasets"] if e["damping"] == run["damping"])
        results.append(evaluate_run(root, run, entry, output / run["run"], args.device, args.batch_size, args.horizons))
    verify_frozen(root, frozen)
    for entry in audit["datasets"]:
        if sha256(root / entry["file"]) != entry["sha256"]:
            raise ValueError("Test data changed during evaluation")
    save_json(output / "completion.json", dict(status="COMPLETE", runs=len(results),
              all_canaries_passed=True, frozen_files_and_data_unchanged=True,
              total_run_seconds=sum(r["total_seconds"] for r in results),
              inference_seconds=sum(r["inference_seconds"] for r in results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--horizons", type=float, nargs="+", default=[4., 8., 16., 32.])
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and thread count must be positive")
    run_matrix(args)
