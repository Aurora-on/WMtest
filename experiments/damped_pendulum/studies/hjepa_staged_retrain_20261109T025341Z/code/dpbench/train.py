"""Train or time the pinned official baseline models on the common HDF5 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import resource
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from dpbench.data import WindowDataset
from dpbench.models import build_model, synchronize, validate_device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def memory_stats(device):
    # Darwin ru_maxrss is bytes; Linux is KiB. MPS reports current, not peak.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out = {"process_peak_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024}
    if device.type == "cuda":
        out["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        out["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    if device.type == "mps":
        out["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        out["mps_current_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return out


def move_batch(batch, device):
    # Explicit whitelist prevents state/energy/parameter labels entering training.
    return {k: batch[k].to(device=device, dtype=torch.float32)
            for k in ("pixels", "action", "dt_frame")}


def validate(model, loader, device):
    cpu_rng = torch.get_rng_state()
    accelerator_rng = None
    if device.type == "cuda":
        accelerator_rng = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        accelerator_rng = torch.mps.get_rng_state()
    torch.manual_seed(192837)
    model.eval()
    total = count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                batch = move_batch(batch, device)
                value = model.loss(batch, "val")["loss"]
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite validation loss")
                n = batch["pixels"].shape[0]
                total += float(value) * n
                count += n
    finally:
        torch.set_rng_state(cpu_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state(accelerator_rng, device)
        elif device.type == "mps":
            torch.mps.set_rng_state(accelerator_rng)
        model.train()
    if not count:
        raise ValueError("Empty validation loader")
    return total / count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", choices=["lehami", "lewm"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--steps", type=int, default=2000, help="Total optimizer steps, not epochs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-frames", type=int, default=81)
    p.add_argument("--unroll", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--val-every", type=int, default=200)
    p.add_argument("--val-windows", type=int, default=256)
    p.add_argument("--benchmark", action="store_true", help="Timing only; no scientific result/checkpoint")
    p.add_argument("--warmup", type=int, default=3, help="Initial steps excluded from timing statistics")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.val_every < 1 or args.val_windows < 1:
        p.error("steps/val-every/val-windows must be positive and batch-size >= 2")
    if args.steps <= args.warmup:
        p.error("steps must exceed warmup for a usable timing estimate")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = validate_device(args.device)
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    dataset_sha256 = file_sha256(args.data)
    with h5py.File(args.data, "r") as f:
        dt = float(f["dt_frame"][0, 0])
        source_attrs = {k: str(v) for k, v in f.attrs.items()}
    model = build_model(args.model, dt=dt, device=device, unroll=args.unroll)
    train_ds = WindowDataset(args.data, "train", model.window, train_frames=args.train_frames)
    val_ds = WindowDataset(args.data, "val", model.window, train_frames=args.train_frames)
    if len(train_ds) < args.batch_size:
        raise ValueError(f"Only {len(train_ds)} training windows; reduce batch-size explicitly")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.workers, generator=generator)
    val_indices = torch.randperm(len(val_ds), generator=torch.Generator().manual_seed(1009))[:args.val_windows].tolist()
    val_loader = DataLoader(Subset(val_ds, val_indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metadata = dict(model.provenance(), args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    torch_version=str(torch.__version__), python=platform.python_version(),
                    device=str(device), machine=platform.machine(), dataset_attributes=source_attrs,
                    dataset_sha256=dataset_sha256,
                    train_windows=len(train_ds), validation_windows_used=len(val_indices),
                    precision="float32", scheduler="constant lr", gradient_clip_norm=1.0,
                    checkpoint_rule="minimum fixed-subset validation objective (model-specific, not cross-model score)",
                    benchmark_only=args.benchmark, scientific_validation=False)
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    iterator = iter(train_loader)
    best = float("inf")
    timings, compute_timings = [], []
    synchronize(device)
    started = time.perf_counter()
    with (args.output / "history.jsonl").open("w") as history:
        for step in range(1, args.steps + 1):
            synchronize(device)
            tick = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = move_batch(batch, device)
            synchronize(device)
            compute_tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(batch, "train")
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            losses["loss"].backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            done = time.perf_counter()
            row = dict(step=step, **{k: float(v.detach()) for k, v in losses.items()},
                       grad_norm=float(grad), wall_step_seconds=done-tick,
                       compute_step_seconds=done-compute_tick)
            if step > args.warmup:
                timings.append(done-tick)
                compute_timings.append(done-compute_tick)
            if not args.benchmark and (step % args.val_every == 0 or step == args.steps):
                row["validation_loss"] = validate(model, val_loader, device)
                checkpoint = dict(kind=args.model, config=model.config,
                                  state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                                  step=step, seed=args.seed, data_path=str(args.data.resolve()),
                                  dataset_sha256=dataset_sha256,
                                  train_frames=args.train_frames, validation_loss=row["validation_loss"])
                torch.save(checkpoint, args.output / "last.pt")
                if row["validation_loss"] < best:
                    best = row["validation_loss"]
                    torch.save(checkpoint, args.output / "best.pt")
            history.write(json.dumps(row) + "\n")
            history.flush()
            print(json.dumps(row), flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    summary = dict(model=args.model, device=str(device), steps=args.steps,
                   warmup_steps=args.warmup, measured_steps=len(timings), batch_size=args.batch_size,
                   wall_seconds=elapsed, step_seconds_mean=float(np.mean(timings)),
                   step_seconds_median=float(np.median(timings)),
                   step_seconds_p90=float(np.percentile(timings,90)),
                   compute_seconds_mean=float(np.mean(compute_timings)),
                   clips_per_second=args.batch_size / float(np.mean(timings)),
                   best_validation_loss=None if args.benchmark else best,
                   extrapolation_caveat="Short-run estimate excludes future validation, checkpoint I/O, thermal effects and queue time.",
                   **memory_stats(device))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
