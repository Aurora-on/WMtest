"""End-to-end frozen-only evaluation against the unchanged original evaluator."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from dpbench import evaluate as original
from dpbench.evaluate_holdout import evaluate_run, input_batch
from dpbench.generate import generate
from dpbench.holdout import sha256
from dpbench.models import build_model


@pytest.mark.parametrize("kind", ["lehami", "lewm"])
def test_frozen_evaluator_matches_original_without_fitting(kind, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    config = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())
    config.update(episodes=16, frames=21, train_frames=17, damping_values=[.2])
    manifest = generate(config, tmp_path / "old_data")
    old = tmp_path / "old_data" / manifest["datasets"][0]["file"]
    run_name = f"{kind}_b0p2_seed0"
    run_dir = tmp_path / "frozen/runs/server_47111" / run_name
    run_dir.mkdir(parents=True)
    torch.manual_seed(1729)
    model = build_model(kind, unroll=8).eval()
    checkpoint = dict(kind=kind, seed=0, step=1, config=model.config, state_dict=model.state_dict(),
                      train_frames=17, dataset_sha256=sha256(old))
    torch.save(checkpoint, run_dir / "best.pt")
    reference = original.evaluate(SimpleNamespace(
        data=old, checkpoint=run_dir / "best.pt", output=run_dir / "evaluation", device="cpu", threads=2,
        labels=32, val_labels=16, seed=17, probes=["ridge", "mlp"], probe_steps=25, horizons=[.8, 1.6], max_episodes=0))
    new = tmp_path / "holdout.h5"
    shutil.copyfile(old, new)
    with h5py.File(new, "r+") as f:
        ids = f["splits/test"][:]
        for split in ("train", "val"):
            del f["splits/" + split]
            f.create_dataset("splits/" + split, data=np.array([], dtype=np.int64))
        f.attrs["train_frames"] = 0
        f.create_dataset("splits/test_all", data=ids)
        f.create_dataset("splits/test_strict_context", data=ids)
    def forbidden(*args, **kwargs):
        raise AssertionError("A frozen evaluation must never fit a readout")
    monkeypatch.setattr(original, "fit_readout", forbidden)
    result = evaluate_run(tmp_path, dict(run=run_name, model=kind, seed=0, checkpoint_step=1,
                                         training_data_sha256=sha256(old), damping=.2),
                          dict(file=new.name, sha256=sha256(new)), tmp_path / "result", "cpu", 1, [.8, 1.6])
    with np.load(run_dir / "evaluation/trajectories.npz") as old_result, np.load(tmp_path / "result/trajectories.npz") as new_result:
        for key in ("features", "predictions", "truth", "episode_ids"):
            np.testing.assert_allclose(old_result[key], new_result[key], rtol=1e-6, atol=1e-6)
    for before, after in zip(reference["aggregate"], [r for r in result["aggregate"] if r["split"] == "test"]):
        assert before["probe"] == after["probe"] and before["mode"] == after["mode"]
        for key in before["mean"]:
            np.testing.assert_allclose(before["mean"][key], after["mean"][key], rtol=1e-6, atol=1e-6)
    assert result["canary"]["predictions_max_abs_difference"] == 0.
    assert result["parameters_and_buffers_unchanged"] and not result["readouts_refitted"]
    with pytest.raises(ValueError, match="exceeds"):
        input_batch(new, ids, 17)
    # Deliberate positive control: changing a legitimate initial image must
    # reach the restricted model input loader.
    before = input_batch(new, ids, 16)
    with h5py.File(new, "r+") as f:
        off = int(f["ep_offset"][ids[0]])
        f["pixels"][off:off + 5] = 0
    after = input_batch(new, ids, 16)
    assert not np.array_equal(before[0], after[0])
