#!/usr/bin/env python3
"""Bounded diagnostic replay from a frozen checkpoint; capture the first strict failure.

The original trainer/model/core source stays unchanged. This wrapper observes
_step inputs and re-raises its original SolverError. At most --max-updates
optimizer updates are permitted; the original total-step config is not changed.
The default is a read-only plan. --execute explicitly permits diagnostic replay.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpbench.dissipative_core import DissipativeCore, SolverError
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.train_dissipative import atomic_json, atomic_torch_save, capture_rng, execute_run, read_config, source_hashes


class ReplayBudgetReached(RuntimeError):
    """The bounded diagnostic ended without reproducing a strict failure."""


def cpu_copy(value):
    return None if value is None else value.detach().cpu().clone()


@contextmanager
def capture_failed_substeps(output, resume_step, max_updates=60, provenance=None):
    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    output = Path(output)
    original_step = DissipativeCore._step
    original_loss = DissipativeWorldModel.loss
    state = dict(training_loss_calls=0, training_step=resume_step, phase=None,
                 substep_call_in_loss=0, captured=False)

    @functools.wraps(original_loss)
    def observed_loss(self, *args, **kwargs):
        is_training = self.training and torch.is_grad_enabled()
        state["phase"] = "training" if is_training else "validation"
        state["substep_call_in_loss"] = 0
        if is_training:
            if state["training_loss_calls"] >= max_updates:
                raise ReplayBudgetReached(
                    f"Diagnostic limit reached after {max_updates} optimizer updates; "
                    "expected strict failure did not recur within this budget")
            state["training_loss_calls"] += 1
            state["training_step"] = resume_step + state["training_loss_calls"]
        return original_loss(self, *args, **kwargs)

    @functools.wraps(original_step)
    def observed_step(self, q, p, action, h, track_gradient, context=None):
        state["substep_call_in_loss"] += 1
        try:
            return original_step(self, q, p, action, h, track_gradient, context=context)
        except SolverError as exc:
            if not state["captured"]:
                # All observation and serialization happen only after failure.
                # No additional forward, RNG draw or tensor copy occurs beforehand.
                state["captured"] = True
                metadata = dict(format_version=1, diagnostic_only=True,
                    core_config=dict(self.config), track_gradient=bool(track_gradient),
                    error_type=type(exc).__name__, error=str(exc), resume_step=resume_step,
                    training_step=state["training_step"], phase=state["phase"],
                    training_loss_calls=state["training_loss_calls"],
                    completed_replay_updates=max(0, state["training_loss_calls"] - int(state["phase"] == "training")),
                    substep_call_in_loss=state["substep_call_in_loss"],
                    max_replay_updates=max_updates, input_device=str(q.device), input_dtype=str(q.dtype),
                    runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                        cuda_build=torch.version.cuda, device_name=torch.cuda.get_device_name(q.device) if q.is_cuda else "cpu"),
                    provenance=provenance or {},
                    reconstruction="DissipativeCore(**core_config); load core_state_dict; use saved input with original _step. h is one actual substep, not a whole frame interval.")
                payload = dict(metadata, core_state_dict={k:cpu_copy(v) for k,v in self.state_dict().items()},
                    input=dict(q=cpu_copy(q), p=cpu_copy(p), action=cpu_copy(action), h=cpu_copy(h), context=cpu_copy(context)),
                    rng=capture_rng())
                try:
                    if (output / "failure_capture.pt").exists():
                        raise FileExistsError("Refusing to replace an existing failure capture")
                    atomic_torch_save(output / "failure_capture.pt", payload)
                    atomic_json(output / "failure_capture.json", metadata)
                    print(json.dumps({"event":"strict_failure_captured", "training_step":state["training_step"],
                        "substep_call_in_loss":state["substep_call_in_loss"],
                        "capture":str(output / "failure_capture.pt")}), flush=True)
                except Exception as capture_error:
                    # Preserve the original numerical failure even if storage fails.
                    print(f"Failure capture could not be saved: {capture_error!r}", file=sys.stderr, flush=True)
            raise

    DissipativeCore._step = observed_step
    DissipativeWorldModel.loss = observed_loss
    try:
        yield state
    finally:
        DissipativeCore._step = original_step
        DissipativeWorldModel.loss = original_loss


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--max-updates", type=int, default=60)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    if not 1 <= args.max_updates <= 60:
        p.error("Diagnostic update budget must be between 1 and 60")
    config = read_config(args.config)
    saved = torch.load(args.resume, map_location="cpu", weights_only=True)
    plan = dict(mode="bounded_failure_replay" if args.execute else "plan_only",
        resume=str(args.resume.resolve()), resume_step=int(saved["step"]),
        max_optimizer_updates=args.max_updates, latest_allowed_optimizer_step=int(saved["step"]) + args.max_updates,
        expected_failure_step=2795, source_files_unchanged=True, original_tolerances_unchanged=True,
        output=str(args.output.resolve()), device=args.device, optimizer_updates_enabled=args.execute,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_source_hashes=source_hashes())
    print(json.dumps(plan), flush=True)
    if not args.execute:
        return plan
    if saved["train_config"] != config:
        raise ValueError("Replay configuration differs from checkpoint")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Replay output must be a new empty directory")
    try:
        with capture_failed_substeps(args.output, int(saved["step"]), args.max_updates, plan):
            return execute_run(args.data, args.output, config, args.device, args.resume)
    finally:
        if args.output.is_dir():
            atomic_json(args.output / "replay_protocol.json", plan)


if __name__ == "__main__":
    main()
