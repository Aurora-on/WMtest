"""Plot observed pilot errors; never infer convergence from this figure."""
import argparse
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--runs", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
with h5py.File(a.data, "r") as f:
    mode = f["mode"][:]
for name, label, color in (("lehami", "LeHami", "#b34a43"), ("lewm", "LeWM adaptation", "#2863a5")):
    d = np.load(a.runs/f"{name}_b0.2_seed0/evaluation/trajectories.npz")
    index = list(d["probe_names"]).index("mlp")
    pred = d["predictions"][:, index, :, 0]
    truth = d["truth"][:, :, 0]
    error = np.arctan2(np.sin(pred-truth), np.cos(pred-truth))
    times = np.arange(error.shape[1])*float(d["dt"])
    for m, ax in enumerate(axes):
        selected = mode[d["episode_ids"]] == m
        curve = np.sqrt(np.mean(error[selected]**2, axis=0))
        ax.plot(times, curve, color=color, label=label, lw=1.4)
        ax.set(xlabel="Autonomous forecast horizon (s)", title=("Free decay" if m == 0 else "Driven / released / driven"))
        ax.grid(alpha=.2)
axes[0].set_ylabel("Angular RMSE across episodes (rad)")
axes[1].legend(frameon=False, loc="lower right")
fig.suptitle("Damped pendulum b=0.2 | 500 training steps, seed 0 | NOT converged comparison", fontsize=11)
a.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.output, dpi=170)
