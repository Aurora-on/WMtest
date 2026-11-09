#!/usr/bin/env python3
"""Re-extract the original frozen damped-model calibration states.

Uses the exact saved train/validation pairs and original batch32 helper. No
world-model/readout optimizer is created; no test images or labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

EXPECTED_CHECKPOINT = "5163a6a293a7c773a90e9ef59a62353a4debdba64c0dc24814bcbdb3b47f9042"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def verify_identity(saved, manifest, checkpoint_hash, seen_hash, readout_hash, sources):
    if (checkpoint_hash != EXPECTED_CHECKPOINT or manifest.get("checkpoint_sha256") != checkpoint_hash
            or manifest.get("status") != "completed" or manifest.get("checkpoint_step") != 19250):
        raise ValueError("Requires the original completed step19250 evaluation checkpoint")
    if saved.get("kind") != "dissipative" or saved.get("step") != 19250:
        raise ValueError("Checkpoint kind/step mismatch")
    if saved.get("dataset_sha256") != seen_hash or manifest.get("dataset_sha256", {}).get("seen") != seen_hash:
        raise ValueError("Seen dataset differs from original checkpoint/evaluation")
    if manifest.get("readout_sha256") != readout_hash or manifest.get("selected_readout") != "mlp":
        raise ValueError("Requires the original frozen MLP-selected readout bundle")
    if saved.get("source_hashes") != sources or manifest.get("original_source_hashes") != sources:
        raise ValueError("Original training/model sources differ")


def load_original(code_root, expected_evaluation_hashes):
    code_root = Path(code_root).resolve()
    for name, module in sys.modules.items():
        if name == "dpbench" or name.startswith("dpbench."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(code_root):
                raise RuntimeError("dpbench already imported from a different checkout")
    actual = {name: sha256(code_root / name) for name in expected_evaluation_hashes}
    if actual != expected_evaluation_hashes:
        raise ValueError("Original evaluation helpers differ from saved manifest")
    sys.path.insert(0, str(code_root))
    from dpbench.dissipative_model import DissipativeWorldModel
    from dpbench.evaluate_passive import calibration_features
    from dpbench.train_dissipative import source_hashes
    return DissipativeWorldModel, calibration_features, source_hashes(), actual


def check_pairs(seen_path, pairs, history, train_frames):
    expected_counts = {"train": 20000, "val": 4000}
    if set(pairs) != set(expected_counts):
        raise ValueError("Calibration split names differ")
    pair_groups, split_groups = {}, {}
    with h5py.File(seen_path, "r") as handle:
        for split, count in expected_counts.items():
            if len(pairs[split]) != count:
                raise ValueError("Saved calibration pair budget differs")
            members = set(int(ep) for ep in handle["splits/" + split][:])
            split_groups[split] = {int(handle["group_id"][ep]) for ep in members}
            group_ids = []
            for pair in pairs[split]:
                if len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in pair):
                    raise ValueError("Expected integer episode/anchor pairs")
                ep, anchor = pair
                if ep not in members:
                    raise ValueError("Calibration pair lies outside its declared split")
                if not history - 1 <= anchor < min(train_frames, int(handle["ep_len"][ep])):
                    raise ValueError("Calibration anchor exceeds original observation bounds")
                group_ids.append(int(handle["group_id"][ep]))
            if len(set(map(tuple, pairs[split]))) != count:
                raise ValueError("Saved calibration pairs contain duplicates")
            pair_groups[split] = np.asarray(group_ids, dtype=np.int64)
    if split_groups["train"] & split_groups["val"]:
        raise ValueError("Train and validation initial-condition groups overlap")
    return pair_groups, {split: len(groups) for split, groups in split_groups.items()}


def verify_scalers(x, y, readouts):
    actual = {"mean": x.mean(0), "scale": np.maximum(x.std(0), 1e-5),
              "ymean": y.mean(0), "yscale": np.maximum(y.std(0), 1e-5)}
    checks = {}
    for mode in ("ridge", "mlp"):
        if mode not in readouts or readouts[mode]["mode"] != mode:
            raise ValueError("Original bundle must contain ridge and MLP readouts")
        checks[mode] = {}
        for key, value in actual.items():
            expected = readouts[mode][key].detach().cpu().numpy()
            matched = value.shape == expected.shape and np.allclose(value, expected, rtol=1e-5, atol=1e-5)
            checks[mode][key] = dict(max_abs_difference=float(np.max(np.abs(value - expected))),
                                    matches=bool(matched))
            if not matched:
                raise ValueError(f"Re-extracted calibration {mode}/{key} differs from frozen readout")
    return checks


def run(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Use a new calibration-export directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(args.threads)
    manifest_path, calibration_path = args.eval_results / "manifest.json", args.eval_results / "calibration.json"
    manifest = json.loads(manifest_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    if manifest.get("batch_size") != 32 or calibration.get("selected") != "mlp":
        raise ValueError("Original calibration convention differs")
    Model, extract, sources, evaluation_sources = load_original(args.code_root, manifest["evaluation_source_hashes"])
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    readout_bytes = args.readouts.read_bytes()
    readout_hash = hashlib.sha256(readout_bytes).hexdigest()
    readouts = torch.load(io.BytesIO(readout_bytes), map_location="cpu", weights_only=True)
    seen_hash = sha256(args.seen)
    verify_identity(saved, manifest, checkpoint_hash, seen_hash, readout_hash, sources)
    model = Model(saved["config"]).to(args.device).eval()
    model.load_state_dict(saved["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.history != 21 or model.dt != .1:
        raise ValueError("Original history/dt convention differs")
    pairs = calibration["pairs"]
    pair_groups, group_counts = check_pairs(args.seen, pairs, model.history, saved["train_frames"])
    before = state_hash(model)
    print(json.dumps(dict(phase="train20000_features", elapsed_seconds=time.perf_counter() - started)), flush=True)
    x, y = extract(model, args.seen, pairs["train"], 32)
    print(json.dumps(dict(phase="val4000_features", elapsed_seconds=time.perf_counter() - started)), flush=True)
    xv, yv = extract(model, args.seen, pairs["val"], 32)
    if not all(np.isfinite(a).all() for a in (x, y, xv, yv)):
        raise ValueError("Nonfinite calibration features or targets")
    scaler_checks = verify_scalers(x, y, readouts)
    after = state_hash(model)
    if before != after:
        raise RuntimeError("World-model parameters changed during export")
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "calibration_features.npz", x=x, y=y, xv=xv, yv=yv,
                        train_pairs=np.asarray(pairs["train"], dtype=np.int64),
                        val_pairs=np.asarray(pairs["val"], dtype=np.int64),
                        train_group_ids=pair_groups["train"], val_group_ids=pair_groups["val"])
    summary = dict(status="completed", checkpoint_step=int(saved["step"]), checkpoint_sha256=checkpoint_hash,
                   dataset_sha256=seen_hash, readout_sha256=readout_hash, selected_original_readout="mlp",
                   manifest_sha256=sha256(manifest_path), calibration_sha256=sha256(calibration_path),
                   source_hashes=sources, evaluation_source_hashes=evaluation_sources, script_sha256=sha256(__file__),
                   model_hash_before=before, model_hash_after=after, parameters_unchanged=True,
                   world_model_updates=0, optimizer_created=False, readout_refitted=False,
                   test_features_accessed=False, labels_scope="Original saved train20000/val4000 pairs only",
                   train_labels=len(x), val_labels=len(xv), unique_group_counts=group_counts,
                   feature_dim=x.shape[1], batch_size=32, scaler_checks=scaler_checks,
                   scaler_tolerance=dict(rtol=1e-5, atol=1e-5),
                   code_root=str(args.code_root.resolve()), device=args.device, torch=str(torch.__version__),
                   elapsed_seconds=time.perf_counter() - started,
                   scope="Calibration feature re-extraction only; no new autonomous rollout or test evaluation")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(phase="completed", elapsed_seconds=summary["elapsed_seconds"],
                         parameters_unchanged=True, scaler_checks=scaler_checks)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "checkpoint", "seen", "eval-results", "readouts", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return run(args)


if __name__ == "__main__":
    main()
