"""End-to-end leakage regressions through the production evaluate() entry point.

All datasets/checkpoints/results are private pytest temporary files. Checkpoints
use fixed synthetic weights in the real upstream architectures, not the study's
trained weights. These tests check information flow, not predictive accuracy or
train/test content deduplication. See LEAKAGE_TESTS.md for scope and usage.
"""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from dpbench import evaluate as evaluation
from dpbench.generate import generate
from dpbench.models import build_model


@pytest.fixture(scope="module", autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    rng = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_evaluation(folder, data, checkpoint_template):
    """Use unchanged evaluator, including its hash gate, probe fit and scoring."""
    folder.mkdir()
    checkpoint = dict(checkpoint_template, data_path=str(data.resolve()),
                      dataset_sha256=digest(data))
    # Bind only this synthetic checkpoint's provenance to the intentional
    # canary dataset. Its learned tensors are identical across every variant.
    checkpoint_path = folder / "fixture.pt"
    torch.save(checkpoint, checkpoint_path)
    output = folder / "evaluation"
    args = SimpleNamespace(
        data=data, checkpoint=checkpoint_path, output=output, device="cpu",
        threads=2, labels=32, val_labels=16, seed=17, probes=["ridge", "mlp"],
        probe_steps=25, horizons=[.8, 1.6], max_episodes=0,
    )
    result = evaluation.evaluate(args)
    # Reject non-finite scores even if an equality helper could accept matching
    # NaNs. Some local NumPy BLAS builds also emit floating-point warnings.
    json.dumps(result, allow_nan=False)
    with np.load(output / "trajectories.npz") as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    for key in ("features", "predictions", "truth"):
        assert np.isfinite(arrays[key]).all(), f"Non-finite {key} cannot pass this audit"
    readouts = torch.load(output / "readouts.pt", weights_only=True)
    calibration = json.loads((output / "calibration.json").read_text())
    return dict(arrays=arrays, readouts=readouts, calibration=calibration,
                metrics=result)


@pytest.fixture(scope="module", params=["lehami", "lewm"])
def experiment(request, tmp_path_factory):
    folder = tmp_path_factory.mktemp("no_future_leakage_" + request.param)
    config = json.loads((Path(__file__).parents[1] / "configs/smoke.json").read_text())
    config.update(episodes=16, frames=21, train_frames=17, damping_values=[.2])
    manifest = generate(config, folder / "data")
    data = folder / "data" / manifest["datasets"][0]["file"]
    with torch.random.fork_rng():
        torch.manual_seed(1729)
        model = build_model(request.param, unroll=8).eval()
        # Upstream LeWM starts AdaLN conditioning at exactly zero. Exercise a
        # nonzero action path without training or changing any upstream code.
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, "adaLN_modulation"):
                    module.adaLN_modulation[-1].weight.normal_(0., .01)
    checkpoint = dict(kind=request.param, config=model.config,
                      state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
                      step=0, seed=1729, train_frames=17)
    baseline = run_evaluation(folder / "baseline", data, checkpoint)
    return SimpleNamespace(folder=folder, data=data, checkpoint=checkpoint,
                           baseline=baseline, kind=request.param)


def variant_file(experiment, tmp_path, variant):
    """Change only test episodes; train/val data and episode lengths stay fixed."""
    path = tmp_path / "variant.h5"
    shutil.copyfile(experiment.data, path)
    rng = np.random.default_rng(8675309)
    with h5py.File(path, "r+") as f:
        for episode in f["splits/test"][:]:
            off, n = int(f["ep_offset"][episode]), int(f["ep_len"][episode])
            future = slice(off + 5, off + n)
            if variant in ("random_future_pixels", "pixels_and_labels"):
                f["pixels"][future] = rng.integers(
                    0, 256, size=f["pixels"][future].shape, dtype=np.uint8)
            elif variant == "shuffled_future_pixels":
                f["pixels"][future] = f["pixels"][future][rng.permutation(n - 5)]
            if variant in ("test_labels", "pixels_and_labels"):
                # Include anchor labels: no physical state, even at the
                # anchor, is an authorized model or readout-calibration input.
                f["state"][off:off + n] = np.column_stack(
                    (np.linspace(-2.7, 2.7, n), np.full(n, 7.)))
                for key in ("energy", "work", "dissipation"):
                    f[key][off:off + n] = rng.uniform(10., 20., size=n)
            if variant == "initial_pixels":
                f["pixels"][off:off + 5] = 255 - f["pixels"][off:off + 5]
            if variant == "late_actions":
                # Change action 10 onward; transitions 4..9 must be identical.
                f["action"][off + 10:off + n - 1] += 3.
    assert digest(path) != digest(experiment.data), "Canary did not alter the input file"
    return path


def assert_same_tree(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_tree(left[key], right[key])
    elif isinstance(left, torch.Tensor):
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=False)
    else:
        assert left == right


def assert_predictions_unchanged(reference, changed):
    for key in ("features", "predictions", "episode_ids", "probe_names", "dt"):
        np.testing.assert_array_equal(reference["arrays"][key], changed["arrays"][key])
    assert reference["calibration"] == changed["calibration"]
    assert_same_tree(reference["readouts"], changed["readouts"])


@pytest.mark.parametrize("variant", ["random_future_pixels", "shuffled_future_pixels",
                                     "test_labels", "pixels_and_labels"])
def test_test_truth_cannot_change_predictions_or_fitted_readouts(experiment, tmp_path, variant):
    data = variant_file(experiment, tmp_path, variant)
    changed = run_evaluation(tmp_path / "run", data, experiment.checkpoint)
    assert_predictions_unchanged(experiment.baseline, changed)
    if variant in ("test_labels", "pixels_and_labels"):
        assert not np.array_equal(experiment.baseline["arrays"]["truth"], changed["arrays"]["truth"])
        before = experiment.baseline["metrics"]["aggregate"][0]["mean"]["omega_rmse_rad_s"]
        after = changed["metrics"]["aggregate"][0]["mean"]["omega_rmse_rad_s"]
        assert before != after, "Scoring must actually use the altered test truth"


@pytest.mark.parametrize("variant", ["initial_pixels", "late_actions"])
def test_authorized_inputs_do_change_predictions(experiment, tmp_path, variant):
    data = variant_file(experiment, tmp_path, variant)
    changed = run_evaluation(tmp_path / "run", data, experiment.checkpoint)
    before, after = experiment.baseline["arrays"], changed["arrays"]
    assert not np.array_equal(before["features"], after["features"])
    assert not np.array_equal(before["predictions"], after["predictions"])
    if variant == "late_actions":
        # Anchor=4. States through frame 10 precede the first altered action.
        for key in ("features", "predictions"):
            axis = 1 if key == "features" else 2
            np.testing.assert_array_equal(np.take(before[key], range(7), axis=axis),
                                          np.take(after[key], range(7), axis=axis))
    # Test imagery/actions must not affect fitting or selection of the probes.
    assert_same_tree(experiment.baseline["readouts"], changed["readouts"])


def test_full_evaluator_encodes_only_initial_history_and_freezes_model(experiment, tmp_path, monkeypatch):
    actual_load = evaluation.load_checkpoint
    models, calls = [], []

    def checked_load(*args, **kwargs):
        model, checkpoint = actual_load(*args, **kwargs)
        original_rollout = model.rollout_features

        def checked_rollout(history, actions, past_actions=None):
            assert history.shape[1] == 5
            assert not any(module.training for module in model.modules())
            seen = []
            spatial = model.model.encoder if experiment.kind == "lehami" else model.model.encoder.stem
            hook = spatial.register_forward_pre_hook(
                lambda module, inputs: seen.append(inputs[0].detach().clone()))
            try:
                result = original_rollout(history, actions, past_actions=past_actions)
            finally:
                hook.remove()
            expected = history[:, 1:] if experiment.kind == "lehami" else history
            expected = model.normalize(expected).reshape(-1, *history.shape[2:])
            assert len(seen) == 1, "Encoder called again during autonomous rollout"
            torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
            calls.append(actions.shape[1])
            return result

        monkeypatch.setattr(model, "rollout_features", checked_rollout)
        models.append(model)
        return model, checkpoint

    monkeypatch.setattr(evaluation, "load_checkpoint", checked_load)
    result = run_evaluation(tmp_path / "run", experiment.data, experiment.checkpoint)
    assert_predictions_unchanged(experiment.baseline, result)
    assert calls.count(16) == len(result["arrays"]["episode_ids"])
    assert calls.count(0) > 0  # The guard also covers probe calibration.
    assert len(models) == 1
    assert_same_tree(experiment.checkpoint["state_dict"], models[0].state_dict())
    with h5py.File(experiment.data, "r") as f:
        splits = {s: set(f["splits/" + s][:].tolist()) for s in ("train", "val", "test")}
    assert not (splits["train"] & splits["val"] or splits["train"] & splits["test"]
                or splits["val"] & splits["test"])
    for split in ("train", "val"):
        assert all(ep in splits[split] and 4 <= t < 17
                   for ep, t in result["calibration"][split])
    assert set(result["arrays"]["episode_ids"]) == splits["test"]


def test_oracle_detects_deliberately_injected_future_frame(experiment, tmp_path, monkeypatch):
    """A deliberately leaky loader must fail the very same invariance oracle."""
    original_loader = evaluation.load_episode

    def leaky_loader(*args, **kwargs):
        episode = original_loader(*args, **kwargs)
        episode["pixels"][4] = episode["pixels"][5]  # WRONG, only inside this test.
        return episode

    monkeypatch.setattr(evaluation, "load_episode", leaky_loader)
    reference = run_evaluation(tmp_path / "leaky_reference", experiment.data, experiment.checkpoint)
    data = variant_file(experiment, tmp_path, "random_future_pixels")
    changed = run_evaluation(tmp_path / "leaky_canary", data, experiment.checkpoint)
    with pytest.raises(AssertionError):
        assert_predictions_unchanged(reference, changed)
