"""Weights-only stage-A to stage-B bootstrap, without modifying strict resume.

This creates a *new*, untrained Adam checkpoint at B step zero.  It never calls
optimizer.step(), loads A optimizer moments, or changes the source checkpoint.
The resulting seed is consumed by the ordinary trainer's strict --resume path.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from dpbench.dissipative_data import (
    DissipativeWindowDataset, StepBatchSampler, fixed_validation_indices,
)
from dpbench.dissipative_model import DissipativeWorldModel, resolve_model_config
from dpbench.train_dissipative import (
    CHECKPOINT_VERSION, atomic_json, atomic_torch_save,
    build_optimizer_parameter_groups, capture_rng, config_digest,
    configure_trainable_modules, maximum_horizon, preflight, restore_rng,
    runtime_signature, source_hashes, validate_checkpoint, validate_config,
)


_NEW_PREFIXES = ("core.damping_net.", "environment_encoder.")
_TRAINING_CHANGES = {
    "trainable_modules", "required_data_scope", "steps", "curriculum",
    "warmup_steps", "minimum_lr_ratio", "seed", "val_seed",
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_state_sha256(state_dict, keys=None):
    """Hash named tensors, including dtype/shape; optionally select shared keys.

Pass the manifest's ``shared_state_keys`` when hashing a trained B checkpoint.
Without keys all supplied tensors are included, so new D/context weights are
not silently ignored.
    """
    names = sorted(state_dict if keys is None else keys)
    if len(names) != len(set(names)) or any(name not in state_dict for name in names):
        raise ValueError("State hash keys must be unique and present")
    digest = hashlib.sha256()
    for name in names:
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"State value is not a tensor: {name}")
        value = value.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def transfer_manifest_path(checkpoint):
    checkpoint = Path(checkpoint)
    return checkpoint.with_suffix(checkpoint.suffix + ".transfer.json")


def _same_nested(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _same_nested(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _same_nested(a, b) for a, b in zip(left, right))
    return left == right


def _check_transition(saved, config_b, expected_a_step, expected_a_data_sha256):
    if type(expected_a_step) is not int or expected_a_step <= 0:
        raise ValueError("expected_a_step must be a positive integer")
    config_a = saved.get("train_config")
    if not isinstance(config_a, dict):
        raise ValueError("Stage A is missing train_config")
    validate_config(config_a)
    validate_config(config_b)
    data_hash = saved.get("dataset_sha256")
    if (not isinstance(data_hash, str) or len(data_hash) != 64
            or any(char not in "0123456789abcdef" for char in data_hash)):
        raise ValueError("Stage A has an invalid dataset_sha256")
    if expected_a_data_sha256 is not None and data_hash != expected_a_data_sha256:
        raise ValueError("Stage A dataset hash differs from the expected data")
    # A weights may be copied from a CUDA artifact on a CPU host. This is not
    # runtime continuation: only A's own device field is checked here.
    validate_checkpoint(saved, config_a, data_hash, source_hashes(), saved.get("device"))
    if (saved["step"] != expected_a_step or config_a["training"]["steps"] != expected_a_step
            or saved.get("validation_status") != "completed"
            or not isinstance(saved.get("validation"), dict)):
        raise ValueError("Stage A must be completed and validated at the exact final step")
    a = resolve_model_config(config_a["model"])
    b = resolve_model_config(config_b["model"])
    if saved["config"] != a:
        raise ValueError("Stage A resolved model config disagrees with train_config")
    if (a["q_dim"] != 2 or a["environment_context_dim"] != 0
            or a["objective_mode"] != "lehami" or a["core"]["damping_mode"] != "none"):
        raise ValueError("Stage A must be the two-dimensional undamped LeHami-objective model")
    if set(config_a["training"].get("trainable_modules", [])) != {"encoder", "initializer", "potential"}:
        raise ValueError("Stage A trainable modules must be encoder, initializer and potential")
    if b["core"]["damping_mode"] != "position_velocity":
        raise ValueError("Stage B requires position_velocity damping")
    expected_modules = ["damping", "environment"] if b["environment_context_dim"] else ["damping"]
    if config_b["training"].get("trainable_modules") != expected_modules:
        raise ValueError(f"Stage B trainable_modules must be {expected_modules}")
    expected_model = copy.deepcopy(a)
    expected_model["core"]["damping_mode"] = "position_velocity"
    expected_model["environment_context_dim"] = b["environment_context_dim"]
    if b != expected_model:
        raise ValueError("Stage B may change only damping mode and damping-only environment context")
    changed = {key for key in set(config_a["training"]) | set(config_b["training"])
               if config_a["training"].get(key) != config_b["training"].get(key)}
    if changed - _TRAINING_CHANGES:
        raise ValueError(f"Unapproved stage B training changes: {sorted(changed - _TRAINING_CHANGES)}")
    if not config_b["training"].get("required_data_scope"):
        raise ValueError("Stage B requires an explicit data scope")
    return a, b, {key: {"before": config_a["training"].get(key),
                       "after": config_b["training"].get(key)} for key in sorted(changed)}


def _copy_shared_state(model, source):
    target = model.state_dict()
    shared = sorted(source)
    if set(source) - set(target):
        raise ValueError("Stage A contains unexpected state keys")
    added = sorted(set(target) - set(source))
    expected_added = sorted(name for name in target if name.startswith(_NEW_PREFIXES))
    if added != expected_added or any(name.startswith(_NEW_PREFIXES) for name in source):
        raise ValueError("Only new damping/context tensors may be absent from stage A")
    for name in shared:
        value = source[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target[name].shape
                or value.dtype != target[name].dtype or not torch.isfinite(value).all()):
            raise ValueError(f"Incompatible/nonfinite shared state tensor: {name}")
        target[name] = value.detach().clone()
    model.load_state_dict(target, strict=True)
    for name in shared:
        if not torch.equal(model.state_dict()[name].detach().cpu(), source[name].detach().cpu()):
            raise ValueError(f"Stage A tensor changed during transfer: {name}")
    return shared, added


def create_damping_seed(stage_a_checkpoint, data_b, output_checkpoint, config_b,
                        device, expected_a_step=20000, damping_init=0.01,
                        expected_a_data_sha256=None):
    """Create/verify an immutable B step-0 seed and a provenance sidecar.

    ``config_b`` may be a model/training dictionary or a JSON path. The single
    environment branch trains D only; the multi-environment branch trains D
    plus a new context encoder. All A state tensors remain exactly frozen.
    Existing valid seeds are checked and reused, never overwritten. An orphan
    seed with matching embedded provenance can have its missing sidecar rebuilt.
    Return a JSON-compatible manifest with ``reused_existing`` indicating reuse.
    """
    if isinstance(damping_init, bool) or not isinstance(damping_init, (int, float)) or not math.isfinite(damping_init) or damping_init <= 0:
        raise ValueError("damping_init must be finite and strictly positive")
    if isinstance(config_b, (str, Path)):
        config_b = json.loads(Path(config_b).read_text())
    config_b = copy.deepcopy(config_b)
    source_path, data_path, output = map(lambda p: Path(p).resolve(),
                                         (stage_a_checkpoint, data_b, output_checkpoint))
    if source_path == output:
        raise ValueError("The stage B seed must not overwrite stage A")
    source_digest = _file_sha256(source_path)
    saved = torch.load(source_path, map_location="cpu", weights_only=True)
    _, resolved_b, changes = _check_transition(saved, config_b, expected_a_step, expected_a_data_sha256)
    report = preflight(data_path, config_b)
    if bool(resolved_b["environment_context_dim"]) != bool(report["requires_environment_context"]):
        raise ValueError("Damping context must match the dataset's multi-environment requirement")
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Stage B seed requires available CPU or CUDA")
    cfg, hashes, runtime = config_b["training"], source_hashes(), runtime_signature(device)
    before_rng = capture_rng()
    try:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        model = DissipativeWorldModel(config_b["model"]).to(device)
        shared, added = _copy_shared_state(model, saved["state_dict"])
        # L must not start identically zero: d(LL^T)/dL would then vanish.
        # Zero output weights make initial D exactly damping_init * I. Hidden
        # D/context gradients become active after the first output-layer update.
        with torch.no_grad():
            last = model.core.damping_net[-1]
            last.weight.zero_()
            rows, cols = torch.tril_indices(model.q_dim, model.q_dim, device=device)
            last.bias.copy_((rows == cols).to(last.bias) * math.sqrt(damping_init))
        trainability = configure_trainable_modules(model, cfg)
        optimizer = torch.optim.AdamW(build_optimizer_parameter_groups(model, cfg),
                                      weight_decay=cfg["weight_decay"])
        if optimizer.state:
            raise ValueError("Stage B optimizer must start without moments")
        kwargs = dict(history=model.history, max_horizon=maximum_horizon(config_b),
                      train_frames=cfg["train_frames"], target_tail_frames=model.config["target_tail_frames"])
        train = DissipativeWindowDataset(data_path, "train", **kwargs)
        val = DissipativeWindowDataset(data_path, "val", **kwargs)
        try:
            val_indices = fixed_validation_indices(val, cfg["val_windows"], cfg["val_seed"])
            first_indices = StepBatchSampler(train, cfg["batch_size"], cfg["seed"]).indices(0)
        finally:
            train.close()
            val.close()
        state = model.state_dict()
        frozen_parameters = sorted(name for name, value in model.named_parameters() if not value.requires_grad)
        trainable_parameters = sorted(name for name, value in model.named_parameters() if value.requires_grad)
        provenance = dict(
            kind="weights_only_damping_bootstrap", format_version=1,
            source_checkpoint=str(source_path), source_checkpoint_sha256=source_digest,
            source_step=saved["step"], source_data_sha256=saved["dataset_sha256"],
            source_config_sha256=saved["config_sha256"], source_hashes=hashes,
            bootstrap_source_sha256=_file_sha256(__file__),
            target_checkpoint=str(output), target_data=str(data_path),
            target_data_sha256=report["dataset_sha256"], target_config_sha256=config_digest(config_b),
            target_device=str(device), target_runtime=runtime,
            branch="multi_environment" if model.environment_context_dim else "single_environment",
            environment_context_dim=model.environment_context_dim, damping_init=float(damping_init),
            shared_state_keys=shared, new_state_keys=added,
            shared_state_sha256=shared_state_sha256(state, shared),
            frozen_parameter_names=frozen_parameters, trainable_parameter_names=trainable_parameters,
            frozen_parameters_sha256=shared_state_sha256(state, frozen_parameters),
            trainability=trainability, training_changes=changes,
            validation_indices=val_indices, initial_train_window_indices=first_indices,
            optimizer_state_entries=0, optimizer_updates=0, target_step=0,
            source_optimizer_loaded=False, source_rng_loaded=False, source_best_loaded=False,
            source_gate_passed=saved["validation"].get("gate_passed"),
            physical_acceptance_claimed=False,
            scope="New B optimizer/sampler/RNG; not continuation of A optimizer history.")
        checkpoint = dict(
            format_version=CHECKPOINT_VERSION, kind=model.kind, config=model.config,
            train_config=config_b, config_sha256=config_digest(config_b), state_dict=state,
            optimizer=optimizer.state_dict(), step=0, sampler_next_step=0, rng=capture_rng(),
            dataset_sha256=report["dataset_sha256"], data_path=str(data_path), source_hashes=hashes,
            device=str(device), seed=cfg["seed"], runtime=runtime, train_frames=cfg["train_frames"],
            trainability=trainability, validation=None, validation_status="not_run",
            validation_indices=val_indices, best_score=None, best_step=None, best_checkpoint_path=None,
            solver_migration=None, stage_transfer=provenance)
        validate_checkpoint(checkpoint, config_b, report["dataset_sha256"], hashes, device, runtime)
    finally:
        restore_rng(before_rng)
    if _file_sha256(source_path) != source_digest:
        raise ValueError("Stage A checkpoint changed during bootstrap; use an immutable exact-step snapshot")
    sidecar = transfer_manifest_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(output.suffix + ".lock")
    # Exclusive construction/revalidation protects against duplicate orchestrators.
    # Keep the lock inode: unlinking it could let two processes lock different
    # inodes for the same path. The OS releases flock after process termination.
    with lock.open("a") as locked:
        fcntl.flock(locked.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        existed = output.exists()
        if existed:
            existing = torch.load(output, map_location="cpu", weights_only=True)
            validate_checkpoint(existing, config_b, report["dataset_sha256"], hashes, device, runtime)
            if (existing.get("stage_transfer") != provenance or existing["step"] != 0
                    or existing.get("validation_status") != "not_run"
                    or not _same_nested(existing["optimizer"], checkpoint["optimizer"])
                    or not _same_nested(existing["rng"], checkpoint["rng"])
                    or existing.get("trainability") != checkpoint["trainability"]
                    or existing["validation_indices"] != val_indices
                    or existing["config"] != checkpoint["config"]
                    or any(existing.get(key) is not None for key in ("best_score", "best_step", "best_checkpoint_path"))
                    or shared_state_sha256(existing["state_dict"]) != shared_state_sha256(state)):
                raise ValueError("Existing B seed differs from the requested pristine transfer")
        elif sidecar.exists():
            raise FileExistsError("Transfer manifest exists without its seed checkpoint")
        else:
            atomic_torch_save(output, checkpoint)
        manifest = dict(provenance, checkpoint_sha256=_file_sha256(output),
                        manifest_path=str(sidecar), new_optimizer_created=True, training_started=False)
        if sidecar.exists():
            if json.loads(sidecar.read_text()) != manifest:
                raise ValueError("Existing transfer manifest or seed hash does not match")
        else:
            atomic_json(sidecar, manifest)
        return dict(manifest, reused_existing=existed)
