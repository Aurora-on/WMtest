"""Sequential local baseline matrix; no remote submission or hidden resume."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT/"data/basic")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--models", nargs="+", choices=["lehami", "lewm"], default=["lehami", "lewm"])
    p.add_argument("--horizons", type=float, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--probe-steps", type=int, default=500)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.output = a.output.resolve(); a.data_dir = a.data_dir.resolve()
    manifest = json.loads((a.data_dir/"manifest.json").read_text())
    for seed in a.seeds:
        for item in manifest["datasets"]:
            data = a.data_dir/item["file"]
            for kind in a.models:
                out = a.output/f"{kind}_b{item['damping']}_seed{seed}"
                train = [sys.executable, "-m", "dpbench.train", "--data", str(data), "--model", kind,
                         "--output", str(out), "--device", a.device, "--steps", str(a.steps),
                         "--batch-size", str(a.batch_size), "--seed", str(seed), "--warmup", "3",
                         "--train-frames", str(manifest["config"]["train_frames"]), "--val-every", "250"]
                evaluate = [sys.executable, "-m", "dpbench.evaluate", "--data", str(data),
                            "--checkpoint", str(out/"best.pt"), "--output", str(out/"evaluation"),
                            "--device", a.device, "--seed", str(seed), "--probe-steps", str(a.probe_steps),
                            "--horizons", *map(str, a.horizons)]
                for phase, command in (("train", train), ("evaluate", evaluate)):
                    print(json.dumps({"phase": phase, "command": command}), flush=True)
                    if not a.dry_run:
                        log_dir = a.output/"logs"
                        log_dir.mkdir(parents=True, exist_ok=True)
                        with (log_dir/f"{out.name}_{phase}.log").open("x") as log:
                            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
