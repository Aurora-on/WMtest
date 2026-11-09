"""Same data and workload as the M5 pilot, on one Slurm-allocated CUDA GPU."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this GPU profile inside a Slurm allocation")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated job has no usable CUDA device")
    out = a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    data = ROOT/"data/basic/pendulum_b0p2.h5"
    manifest = json.loads((data.parent/"manifest.json").read_text())
    expected = next(d["sha256"] for d in manifest["datasets"] if d["file"] == data.name)
    if sha256(data) != expected:
        raise RuntimeError("Transferred data hash mismatch")
    subprocess.run([sys.executable, str(ROOT/"scripts/bootstrap_upstream.py"), "--verify"], check=True)
    props = torch.cuda.get_device_properties(0)
    versions = {}
    for package in ("torch", "numpy", "scipy", "h5py", "Pillow", "omegaconf", "einops", "PyYAML"):
        versions[package] = importlib.metadata.version(package)
    meta = dict(job_id=os.environ["SLURM_JOB_ID"], python=platform.python_version(),
                hostname=platform.node(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
                cuda_runtime=torch.version.cuda, versions=versions,
                cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                dataset_sha256=expected, steps=500, warmup_for_statistics=20, batch_size=32,
                protocol="Same fixed b=.2 data, seed0, FP32 parameters, 500 updates, validation every250 with256 windows, full64-episode evaluation and fixed4096/1024 readout labels. No autocast; backend TF32 defaults recorded separately.")
    (out/"hardware.json").write_text(json.dumps(meta, indent=2))
    records = []
    for model in ("lehami", "lewm"):
        run = out/model
        phases = {
            "train": [sys.executable, "-m", "dpbench.train", "--data", str(data), "--model", model,
                      "--output", str(run), "--device", "cuda", "--steps", "500", "--warmup", "20",
                      "--batch-size", "32", "--seed", "0", "--train-frames", "81", "--val-every", "250",
                      "--val-windows", "256", "--threads", "4", "--workers", "0"],
            "evaluate": [sys.executable, "-m", "dpbench.evaluate", "--data", str(data),
                         "--checkpoint", str(run/"best.pt"), "--output", str(run/"evaluation"),
                         "--device", "cuda", "--labels", "4096", "--val-labels", "1024", "--probe-steps", "500",
                         "--seed", "0", "--threads", "4", "--horizons", "4", "8", "16", "32"]}
        for phase, command in phases.items():
            print(json.dumps(dict(model=model, phase=phase, command=command)), flush=True)
            tick = time.perf_counter()
            with (out/f"{model}_{phase}.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            records.append(dict(model=model, phase=phase, process_wall_seconds=time.perf_counter()-tick,
                                returncode=result.returncode))
            (out/"process_times.json").write_text(json.dumps(records, indent=2))
            if result.returncode:
                raise RuntimeError(f"{model} {phase} failed; inspect its log")
    (out/"COMPLETE.json").write_text(json.dumps(dict(success=True, records=records), indent=2))


if __name__ == "__main__":
    main()
