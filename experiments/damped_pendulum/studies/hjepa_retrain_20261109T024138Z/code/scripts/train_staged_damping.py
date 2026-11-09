#!/usr/bin/env python3
"""A(20k, undamped) -> parallel B(single, multi), with immutable milestones.

Default: print the experiment plan without generating data or training.
Execution needs --execute and two visible GPUs (--gpus 0,1), or --device cpu.
Rerun the same command/output after interruption; completed work is verified
and reused, while each resumed trainer writes a new segment directory.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
A_STEPS = 20000
BRANCHES = ("single", "multi")


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_copy(source, target):
    source, target = Path(source), Path(target)
    expected = digest(source)
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f"Refusing to overwrite a different artifact: {target}")
        return expected
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        if digest(temporary) != expected or digest(source) != expected:
            raise ValueError(f"Source changed during copy: {source}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return expected


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        yield


def build_configs(config_a, damping_steps=20000):
    """Preserve A's raw config exactly, including implicit q_dim=2 defaults."""
    a = copy.deepcopy(config_a)
    m, t = a["model"], a["training"]
    if (m.get("q_dim", 2) != 2 or m.get("objective_mode") != "lehami"
            or m["core"].get("damping_mode") != "none"
            or m.get("environment_context_dim", 0) != 0 or t["steps"] != A_STEPS
            or t.get("required_data_scope") != "undamped_stage_a"
            or t.get("trainable_modules") != ["encoder", "initializer", "potential"]):
        raise ValueError("A must be the 2D, LeHami, undamped, full-20000-step configuration")
    if (m.get("history") != 21 or m.get("velocity_history") != 5 or m.get("dt", .1) != .1
            or m.get("target_tail_frames") != 2 or m.get("momentum_reference_time") != .1
            or m["core"].get("potential_mode") != "raw" or m["core"].get("substeps") != 3
            or m["core"].get("integrator") != "frozen_dg"
            or t.get("train_frames") != 81 or t.get("required_renderer") != "smooth_v1"):
        raise ValueError("A must preserve the current physical-time/renderer/evaluation protocol")
    if type(damping_steps) is not int or damping_steps < 1:
        raise ValueError("damping_steps must be a positive integer")
    result = {"a": a}
    for name in BRANCHES:
        b = copy.deepcopy(a)
        b["model"]["core"]["damping_mode"] = "position_velocity"
        b["model"]["environment_context_dim"] = 4 if name == "multi" else 0
        b["training"].update(
            steps=damping_steps,
            trainable_modules=["damping", "environment"] if name == "multi" else ["damping"],
            required_data_scope="damping_stage_b_" + name,
        )
        # A larger optional budget keeps K32 for the extension; physical time is unchanged.
        b["training"]["curriculum"][-1]["until_step"] = max(A_STEPS, damping_steps)
        result[name] = b
    return result


def experiment_plan(args):
    configs = build_configs(read_json(args.config_a), args.damping_steps)
    parent = read_json(args.parent_data_config)
    if (parent.get("groups") != {"train": 384, "val": 64, "test": 128}
            or parent.get("seen_damping") != [0., .05, .1, .2, .5, 1.]
            or parent.get("dt") != .1 or parent.get("history") != 21
            or parent.get("train_frames") != 81 or parent.get("test_frames") != 341
            or parent.get("substeps") != 20 or parent.get("renderer") != "smooth_v1"):
        raise ValueError("Keep the established paired-data/time protocol and calibration group counts")
    if (not math.isfinite(args.damping) or args.damping <= 0
            or args.damping not in parent["seen_damping"]):
        raise ValueError("Single damping must be a positive member of parent seen_damping")
    if not math.isfinite(args.damping_init) or args.damping_init <= 0:
        raise ValueError("damping_init must be strictly positive")
    path = lambda p: str(Path(p).resolve()) if p else None
    return {
        "schema_version": 1, "configs": configs, "parent_data_config": parent,
        "inputs": {"data_a": path(args.data_a), "data_single": path(args.data_single),
                   "data_multi": path(args.data_multi), "resume_a": path(args.resume_a)},
        "damping": args.damping, "damping_init": args.damping_init,
        "device": args.device, "gpus": args.gpus,
        "stage_a_total_steps": A_STEPS,
        "stage_b_steps_each": args.damping_steps,
        "stage_b_launch": "parallel_after_A_exact_20000_checkpoint_and_readout",
        "seen_damping": parent["seen_damping"],
        "readout": "fit once on A20000 train groups; freeze and reuse for both B branches",
        "selection_policy": "exact last checkpoint at target step; earlier best is diagnostic only",
        "quality_policy": "record physical quality without manual gate; execution/provenance failure stops dependent work",
        "test_policy": "train/val only; unseen test damping is never used for fitting or selection",
        "optimizer_policy": "A resumes optimizer/RNG/sampler; each B starts fresh at step0",
        "time_settings": {"dt": configs["a"]["model"].get("dt", .1),
                          "history": configs["a"]["model"]["history"],
                          "curriculum": configs["a"]["training"]["curriculum"],
                          "val_horizons": configs["a"]["training"]["val_horizons"]},
    }


def source_snapshot():
    # Include imported experiment helpers, not only the trainer's seven files.
    files = list((ROOT / "dpbench").glob("*.py"))
    files += [Path(__file__), ROOT / "scripts/evaluate_staged_damping.py",
              ROOT / "third_party/le-wm/module.py"]
    files += list((ROOT / "studies/dimension_geometry_ablation_20261105").glob("*.py"))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}


def verify_sources(manifest):
    for relative, expected in manifest["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Experiment source changed: {relative}")


def verify_artifacts(directory, manifest):
    for relative, expected in manifest["artifacts"].items():
        if digest(Path(directory) / relative) != expected:
            raise ValueError(f"Saved artifact changed: {directory}/{relative}")


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def check_checkpoint(path, config, data_hash, device, target=None):
    from dpbench.train_dissipative import source_hashes, validate_checkpoint
    saved = load_checkpoint(path)
    validate_checkpoint(saved, config, data_hash, source_hashes(), device)
    if target is not None and (saved["step"] != target or saved.get("validation_status") != "completed"):
        raise ValueError(f"Expected validated exact step {target}: {path}")
    return saved


def select_resume(stage_dir, initial, config, data_hash, device):
    """Latest segment is authoritative; corrupt later checkpoints never fall back silently."""
    paths = sorted((Path(stage_dir) / "segments").glob("*/last.pt"))
    candidate = paths[-1] if paths else initial
    if candidate is not None:
        check_checkpoint(candidate, config, data_hash, device)
    return candidate


def train_to_target(stage_dir, data, config, device, initial=None):
    from dpbench.train_dissipative import execute_run
    stage_dir = Path(stage_dir)
    data_hash, target = digest(data), config["training"]["steps"]
    resume = select_resume(stage_dir, initial, config, data_hash, device)
    if resume and check_checkpoint(resume, config, data_hash, device).get("validation_status") == "completed":
        if load_checkpoint(resume)["step"] == target:
            return Path(resume)
    segments = stage_dir / "segments"
    segments.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.name) for p in segments.iterdir() if p.is_dir() and p.name.isdigit()]
    output = segments / f"{max(numbers, default=0) + 1:04d}"
    execute_run(data, output, config, device=device, resume=resume, stop_after_step=target)
    check_checkpoint(output / "last.pt", config, data_hash, device, target)
    return output / "last.pt"


def seal_checkpoint(source, output, config, data_hash, device):
    """Publish last.pt at the exact budget, independently of model selection."""
    output = Path(output)
    target = config["training"]["steps"]
    check_checkpoint(source, config, data_hash, device, target)
    marker = output / "checkpoint_manifest.json"
    if marker.exists():
        saved = read_json(marker)
        verify_artifacts(output, saved)
        if digest(source) != saved["artifacts"]["last.pt"]:
            raise ValueError("Milestone checkpoint does not match current completed run")
        return output / "last.pt"
    artifacts = {"last.pt": immutable_copy(source, output / "last.pt")}
    for name in ("best.pt", "history.jsonl", "run.json", "status.json", "resume_validation.json"):
        path = Path(source).parent / name
        if path.is_file():
            artifacts[name] = immutable_copy(path, output / name)
    atomic_json(marker, {"exact_step": target, "source": str(Path(source).resolve()),
                         "best_is_not_handoff": True, "artifacts": artifacts})
    return output / "last.pt"


def verify_frozen_state(parent, child):
    import torch
    a, b = load_checkpoint(parent), load_checkpoint(child)
    # A contains no damping/context tensors: every A tensor must be identical in B.
    for key, tensor in a["state_dict"].items():
        if key not in b["state_dict"] or not torch.equal(tensor, b["state_dict"][key]):
            raise ValueError(f"Frozen A tensor changed in B: {key}")
    return {"all_parent_tensors_identical": True, "tensor_count": len(a["state_dict"]),
            "parent_sha256": digest(parent), "child_sha256": digest(child)}


def evaluate(checkpoint, data, output, stage, expected_step, device, readout=None):
    output = Path(output)
    request = {"checkpoint_sha256": digest(checkpoint), "data_sha256": digest(data),
               "stage": stage, "expected_step": expected_step,
               "readout_sha256": digest(readout) if readout else None}
    marker = output / "complete.json"
    if marker.exists():
        done = read_json(marker)
        if done["request"] != request:
            raise ValueError("Evaluation request differs from saved result")
        verify_artifacts(output, done)
        return output
    # Failed evaluations live in separate attempts; a retry cannot mix partial results.
    attempts = output.parent / (output.name + "_attempts")
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / uuid.uuid4().hex
    command = [sys.executable, str(ROOT / "scripts/evaluate_staged_damping.py"),
               "--checkpoint", str(checkpoint), "--data", str(data), "--output", str(attempt),
               "--stage", stage, "--expected-step", str(expected_step),
               "--code-root", str(ROOT), "--device", device]
    if readout:
        command += ["--readout", str(readout)]
    subprocess.run(command, check=True, cwd=ROOT)
    required = ["summary.json", "trajectories.npz", "readout.npz", "readout_provenance.json"]
    for name in required:
        if not (attempt / name).is_file():
            raise ValueError(f"Evaluation did not produce {name}")
    if output.exists():
        raise ValueError(f"Incomplete published evaluation requires inspection: {output}")
    artifacts = {str(p.relative_to(attempt)): digest(p) for p in sorted(attempt.rglob("*")) if p.is_file()}
    atomic_json(attempt / "complete.json", {"request": request, "artifacts": artifacts})
    os.replace(attempt, output)
    return output


def milestone(root, name, steps):
    return Path(root) / "milestones" / (f"undamped_step{steps}" if name == "a" else f"damping_{name}_step{steps}")


def qualify_stage_a(root, manifest):
    """Read-only check in A's actual single-GPU runtime, before B data generation."""
    import torch
    from dpbench.dissipative_model import resolve_model_config
    from dpbench.passive_family import _parent_config_sha256
    from dpbench.train_dissipative import preflight, runtime_signature, source_hashes, validate_checkpoint
    root, plan = Path(root), manifest["plan"]
    verify_sources(manifest)
    prepared_path = root / "prepared.json"
    if prepared_path.exists():
        prepared = read_json(prepared_path)
        data, initial = prepared["data"]["a"]["path"], prepared.get("resume_a")
        if digest(data) != prepared["data"]["a"]["sha256"]:
            raise ValueError("Prepared A data changed")
        for relative, expected in prepared.get("input_artifacts", {}).items():
            if digest(root / relative) != expected:
                raise ValueError("Copied initial checkpoint changed")
    else:
        data, initial = plan["inputs"]["data_a"], plan["inputs"]["resume_a"]
    config, device = plan["configs"]["a"], plan["device"]
    report = preflight(data, config)
    data_manifest = read_json(Path(data).parent / "manifest.json")
    entry = next((row for row in data_manifest.get("datasets", []) if row["file"] == Path(data).name), None)
    if (data_manifest.get("config") != plan["parent_data_config"]
            or data_manifest.get("parent_config_sha256") != _parent_config_sha256(plan["parent_data_config"])
            or entry is None or entry.get("sha256") != report["dataset_sha256"]):
        raise ValueError("A data/generation manifest differs from this experiment's parent protocol")
    for name in ("dpbench/passive_family.py", "dpbench/generate.py"):
        if data_manifest.get("source_sha256", {}).get(name) != digest(ROOT / name):
            raise ValueError("A generator source differs from the planned B generator")
    runtime = runtime_signature(device)
    resume = select_resume(root / "stage_a", initial, config, report["dataset_sha256"], device)
    saved = None
    if resume:
        saved = load_checkpoint(resume)
        continuing = saved["step"] < A_STEPS or saved.get("validation_status") != "completed"
        validate_checkpoint(saved, config, report["dataset_sha256"], source_hashes(), device,
                            runtime if continuing else None)
        if saved["config"] != resolve_model_config(config["model"]):
            raise ValueError("A resolved model configuration differs from the experiment")
        if continuing:
            before_rng = torch.cuda.device_count() if device == "cuda" else 0
            if len(saved["rng"].get("torch_cuda", [])) != before_rng:
                raise ValueError("A saved CUDA RNG topology differs from the worker runtime")
        if any(not torch.isfinite(tensor).all() for tensor in saved["state_dict"].values()):
            raise ValueError("A resume contains nonfinite model tensors")
        if continuing and saved.get("best_score") is not None:
            best_path = Path(resume).parent / "best.pt"
            if not best_path.exists():
                best_path = Path(saved["best_checkpoint_path"])
            best = load_checkpoint(best_path)
            validate_checkpoint(best, config, report["dataset_sha256"], source_hashes(), device, runtime)
            if (best["step"] != saved["best_step"] or best["step"] > saved["step"]
                    or best["best_score"] != saved["best_score"]):
                raise ValueError("A best checkpoint does not match the resumed history")
    verify_sources(manifest)
    atomic_json(root / "stage_a_qualification.json", {
        "status": "qualified", "training_started": False, "runtime": runtime,
        "data_sha256": report["dataset_sha256"], "resume": str(resume) if resume else None,
        "resume_sha256": digest(resume) if resume else None,
        "resume_step": saved["step"] if saved else 0, "optimizer_created": False})


def run_worker(root, name):
    root = Path(root)
    manifest = read_json(root / "pipeline.json")
    if name == "qualify_a":
        qualify_stage_a(root, manifest)
        return
    plan, prepared = manifest["plan"], read_json(root / "prepared.json")
    config, device = plan["configs"][name], plan["device"]
    stage_dir = root / ("stage_a" if name == "a" else "stage_b_" + name)
    stage_dir.mkdir(parents=True, exist_ok=True)
    with lock(stage_dir / ".worker.lock"):
        try:
            verify_sources(manifest)
            data = Path(prepared["data"][name]["path"])
            if digest(data) != prepared["data"][name]["sha256"]:
                raise ValueError("Prepared training data changed")
            atomic_json(stage_dir / "worker_status.json", {"status": "running", "branch": name})
            initial = prepared.get("resume_a") if name == "a" else None
            a_milestone = milestone(root, "a", A_STEPS)
            if name != "a":
                a_done = read_json(a_milestone / "evaluation/complete.json")
                verify_artifacts(a_milestone / "evaluation", a_done)
                verify_artifacts(a_milestone, read_json(a_milestone / "checkpoint_manifest.json"))
                from dpbench.staged_damping import create_damping_seed
                seed = stage_dir / "initialization/seed.pt"
                seed.parent.mkdir(parents=True, exist_ok=True)
                create_damping_seed(a_milestone / "last.pt", data, seed, config,
                                    device=device, expected_a_step=A_STEPS,
                                    damping_init=plan["damping_init"],
                                    expected_a_data_sha256=prepared["data"]["a"]["sha256"])
                initial = seed
            checkpoint = train_to_target(stage_dir, data, config, device, initial)
            verify_sources(manifest)
            final_dir = milestone(root, name, config["training"]["steps"])
            checkpoint = seal_checkpoint(checkpoint, final_dir, config, digest(data), device)
            readout = None
            if name != "a":
                frozen = verify_frozen_state(a_milestone / "last.pt", checkpoint)
                atomic_json(final_dir / "frozen_state_audit.json", frozen)
                readout = a_milestone / "evaluation/readout.npz"
            evaluate(checkpoint, data, final_dir / "evaluation", "a" if name == "a" else "b",
                     config["training"]["steps"], device, readout)
            verify_sources(manifest)
            atomic_json(stage_dir / "worker_status.json", {"status": "completed", "branch": name,
                        "completed_steps": config["training"]["steps"], "milestone": str(final_dir)})
        except BaseException as exc:
            atomic_json(stage_dir / "worker_status.json", {"status": "failed", "branch": name,
                        "error": f"{type(exc).__name__}: {exc}"})
            raise


def prepare(root, manifest):
    from dpbench.damping_stage_data import generate, validate_pair, plan as data_plan
    from dpbench.train_dissipative import preflight
    root, plan = Path(root), manifest["plan"]
    marker = root / "prepared.json"
    if marker.exists():
        prepared = read_json(marker)
        for item in prepared["data"].values():
            if digest(item["path"]) != item["sha256"]:
                raise ValueError("Prepared data hash changed")
        for relative, expected in prepared.get("input_artifacts", {}).items():
            if digest(root / relative) != expected:
                raise ValueError("Copied initial checkpoint changed")
        return prepared
    data = {"a": Path(plan["inputs"]["data_a"])}
    for name in BRANCHES:
        supplied = plan["inputs"]["data_" + name]
        directory = root / ("data_" + name)
        if supplied:
            data[name] = Path(supplied)
        else:
            # Generation is independently transactional, so interrupted preparation is recoverable.
            if not (directory / "manifest.json").exists():
                if directory.exists():
                    raise ValueError(f"Incomplete generated dataset: {directory}")
                temporary = root / (".data_" + name + "." + uuid.uuid4().hex)
                generate(plan["parent_data_config"], temporary, damping=plan["damping"], multi=name == "multi")
                os.replace(temporary, directory)
            data[name] = directory / ("seen.h5" if name == "multi" else "damping.h5")
    audits = {name: validate_pair(data["a"], data[name], multi=name == "multi") for name in BRANCHES}
    for name in BRANCHES:
        expected = data_plan(plan["parent_data_config"], damping=plan["damping"], multi=name == "multi")
        if (audits[name]["selected_damping"] != expected["selected_damping"]
                or audits[name]["parent_config_sha256"] != expected["parent_config_sha256"]):
            raise ValueError(f"Supplied {name} data differs from this experiment's declared conditions/parent")
    reports = {name: preflight(path, plan["configs"][name]) for name, path in data.items()}
    prepared = {"data": {name: {"path": str(path.resolve()), "sha256": digest(path)} for name, path in data.items()},
                "pair_audits": audits, "preflights": reports, "resume_a": None, "input_artifacts": {}}
    initial = plan["inputs"]["resume_a"]
    if initial:
        saved = check_checkpoint(initial, plan["configs"]["a"], prepared["data"]["a"]["sha256"], plan["device"])
        copied = root / "inputs/stage_a/last.pt"
        prepared["input_artifacts"][str(copied.relative_to(root))] = immutable_copy(initial, copied)
        prepared["resume_a"] = str(copied)
        if saved.get("best_score") is not None:
            best = Path(initial).parent / "best.pt"
            if not best.exists():
                best = Path(saved["best_checkpoint_path"])
            target = copied.parent / "best.pt"
            prepared["input_artifacts"][str(target.relative_to(root))] = immutable_copy(best, target)
    atomic_json(marker, prepared)
    return prepared


def gpu_environments(device, gpus):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONPATH=str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
        return [dict(env), dict(env)]
    import torch
    ids = [int(x) for x in gpus.split(",")]
    if len(ids) != 2 or len(set(ids)) != 2 or min(ids) < 0 or max(ids) >= torch.cuda.device_count():
        raise ValueError("Parallel B branches require two distinct allocated visible GPUs (--gpus 0,1)")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
    return [dict(env, CUDA_VISIBLE_DEVICES=physical[i]) for i in ids]


def launch_workers(root, names, environments):
    """Start all siblings before waiting. One failure doesn't erase the other result."""
    processes, handles = [], []
    try:
        for name, env in zip(names, environments):
            log_dir = Path(root) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            handle = (log_dir / (name + ".log")).open("a")
            handles.append(handle)
            command = [sys.executable, str(Path(__file__).resolve()), "--output", str(root),
                       "--worker", name, "--execute"]
            processes.append((name, subprocess.Popen(command, cwd=ROOT, env=env,
                                                     stdout=handle, stderr=subprocess.STDOUT)))
        failures = {name: code for name, process in processes if (code := process.wait()) != 0}
        if failures:
            raise RuntimeError(f"Training/evaluation workers failed: {failures}; see logs and worker_status.json")
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        raise
    finally:
        for handle in handles:
            handle.close()


def execute(args, plan):
    if not args.output or not args.data_a:
        raise ValueError("Execution requires --output and --data-a")
    environments = gpu_environments(args.device, args.gpus)
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".pipeline.lock"):
        manifest_path = root / "pipeline.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["plan"] != plan:
                raise ValueError("Existing output belongs to a different experiment plan")
            verify_sources(manifest)
        else:
            if any(p.name != ".pipeline.lock" for p in root.iterdir()):
                raise ValueError("New pipeline output must be empty")
            manifest = {"plan": plan, "source_sha256": source_snapshot()}
            atomic_json(manifest_path, manifest)
        try:
            atomic_json(root / "status.json", {"status": "qualifying_stage_a"})
            launch_workers(root, ["qualify_a"], [environments[0]])
            atomic_json(root / "status.json", {"status": "preparing"})
            prepare(root, manifest)
            atomic_json(root / "status.json", {"status": "running_stage_a"})
            launch_workers(root, ["a"], [environments[0]])
            verify_sources(manifest)
            # Both B subprocesses derive from the same published A milestone.
            atomic_json(root / "status.json", {"status": "running_stage_b_parallel", "branches": list(BRANCHES)})
            launch_workers(root, list(BRANCHES), environments)
            verify_sources(manifest)
            atomic_json(root / "status.json", {"status": "completed", "stage_a_steps": A_STEPS,
                                               "stage_b_steps_each": args.damping_steps})
        except BaseException as exc:
            atomic_json(root / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config-a", type=Path, default=ROOT / "configs/dissipative_loss_pair_lehami_v1.json")
    p.add_argument("--parent-data-config", type=Path, default=ROOT / "configs/passive_family_smooth_v1.json")
    p.add_argument("--data-a", type=Path)
    p.add_argument("--data-single", type=Path)
    p.add_argument("--data-multi", type=Path)
    p.add_argument("--resume-a", type=Path, help="A last.pt (e.g. step10000), with sibling best.pt")
    p.add_argument("--output", type=Path)
    p.add_argument("--damping", type=float, default=.2)
    p.add_argument("--damping-init", type=float, default=.01, help="Initial D=lambda I; lambda>0 avoids dead factor gradients")
    p.add_argument("--damping-steps", type=int, default=20000, help="Independent B optimizer budget for EACH branch")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--gpus", default="0,1", help="Two distinct GPU indices within the current allocation")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker", choices=("qualify_a", "a", *BRANCHES), help=argparse.SUPPRESS)
    return p


def main(argv=None):
    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    args = parser().parse_args(argv)
    if args.worker:
        if not args.execute or not args.output:
            raise ValueError("Internal worker requires --execute --output")
        run_worker(args.output, args.worker)
        return
    plan = experiment_plan(args)
    if not args.execute:
        print(json.dumps({"mode": "plan_only", "training_started": False, "plan": plan}, indent=2, ensure_ascii=False))
        return
    execute(args, plan)


if __name__ == "__main__":
    main()
