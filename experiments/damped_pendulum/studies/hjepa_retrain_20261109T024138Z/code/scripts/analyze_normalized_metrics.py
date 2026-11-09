"""Post-hoc amplitude-normalized diagnostics for saved pendulum predictions.

No training, model selection, future-frame re-encoding, or changes to original
metrics. The reference RMS is a test-set diagnostic, not a training statistic.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import numpy as np


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def errors(pred, truth):
    difference = pred - truth
    difference[..., 0] = wrap(difference[..., 0])
    return difference


def score(pred, truth):
    """Pool squared errors over episodes/time, separately for theta and omega."""
    difference = errors(pred, truth)
    equilibrium_error = errors(np.zeros_like(truth), truth)
    mse = np.mean(difference**2, axis=(0, 1))
    reference_mse = np.mean(equilibrium_error**2, axis=(0, 1))
    rmse = np.sqrt(mse)
    signal = np.sqrt(reference_mse)
    # Undefined reference amplitudes remain undefined; no arbitrary epsilon.
    ratios = [float(r/s) if s > 0 else None for r, s in zip(rmse, signal)]
    return {
        name: dict(pooled_rmse=float(rmse[i]), equilibrium_reference_rms=float(signal[i]),
                   mean_episode_rmse=float(np.mean(np.sqrt(np.mean(difference[..., i]**2, axis=1)))),
                   nrmse=ratios[i], mse_skill_vs_equilibrium=None if ratios[i] is None else 1-ratios[i]**2)
        for i, name in enumerate(("angle", "omega"))
    }


def self_check():
    truth = np.array([[[3.13, 2.], [-3.13, -2.]], [[.4, 1.], [-.4, -1.]]])
    exact, zero = score(truth, truth), score(np.zeros_like(truth), truth)
    assert all(exact[k]["nrmse"] == 0 for k in exact)
    assert all(np.isclose(zero[k]["nrmse"], 1) and np.isclose(zero[k]["mse_skill_vs_equilibrium"], 0) for k in zero)
    assert abs(wrap(-3.13-3.13)) < .024
    empty = score(np.ones((2, 3, 2)), np.zeros((2, 3, 2)))
    assert all(empty[k]["nrmse"] is None for k in empty)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    self_check()
    rows, source_files, reference_truth, reference_modes, identifiers = [], {}, {}, {}, set()
    reference_dt = None
    matched_metric_rows = 0
    for run in sorted(args.runs_root.iterdir()):
        if not (run/"run.json").exists():
            continue
        metadata = json.loads((run/"run.json").read_text())
        metrics = json.loads((run/"evaluation/metrics.json").read_text())
        model, seed = metadata["args"]["model"], int(metadata["args"]["seed"])
        damping = float(metadata["dataset_attributes"]["damping"])
        identity = (model, damping, seed)
        assert identity not in identifiers
        identifiers.add(identity)
        path = run/"evaluation/trajectories.npz"
        source_files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as saved:
            prediction, truth = saved["predictions"], saved["truth"]
            ids, probes, dt = saved["episode_ids"], saved["probe_names"].tolist(), float(saved["dt"])
        assert truth.shape == (64, 321, 2) and prediction.shape == (64, 2, 321, 2)
        if reference_dt is None:
            reference_dt = dt
        assert dt == reference_dt
        assert np.isfinite(truth).all() and np.isfinite(prediction).all()
        if damping in reference_truth:
            ref_ids, ref_truth = reference_truth[damping]
            assert np.array_equal(ids, ref_ids) and np.array_equal(truth, ref_truth)
        else:
            reference_truth[damping] = ids, truth
        modes = {r["episode"]: r["mode"] for r in metrics["episodes"]}
        if damping in reference_modes:
            assert modes == reference_modes[damping]
        else:
            reference_modes[damping] = modes
        original = {(r["probe"], r["mode"], float(r["horizon_s"])): r for r in metrics["aggregate"]}
        for mode, horizon, probe in itertools.product(("passive", "driven"), (8., 32.), probes):
            index = [j for j, ep in enumerate(ids) if modes[int(ep)] == mode]
            assert len(index) == 32
            steps = round(horizon/dt)
            y = truth[index, 1:steps+1]
            yhat = prediction[index, probes.index(probe), 1:steps+1]
            value = score(yhat, y)
            for key, original_key in (("angle", "angle_rmse_rad"), ("omega", "omega_rmse_rad_s")):
                assert np.isclose(value[key]["mean_episode_rmse"], original[(probe, mode, horizon)]["mean"][original_key], rtol=1e-10, atol=1e-12)
                assert value[key]["pooled_rmse"] + 1e-12 >= value[key]["mean_episode_rmse"]
                matched_metric_rows += 1
            # A privileged diagnostic: persists the true initial theta/omega,
            # whereas the learned model receives images rather than that state.
            persistence = score(np.broadcast_to(truth[index, :1], y.shape), y)
            rows.append(dict(model=model, damping=damping, seed=seed, probe=probe, mode=mode,
                             horizon_s=horizon, episodes=len(index), metrics=value,
                             oracle_true_initial_state_persistence=persistence))
    assert identifiers == set(itertools.product(("lehami", "lewm"), (0., .2), (0, 1, 2)))
    assert np.array_equal(reference_truth[0.][0], reference_truth[.2][0])
    assert reference_modes[0.] == reference_modes[.2]
    motion = []
    for damping, (ids, truth) in sorted(reference_truth.items()):
        for mode, horizon in itertools.product(("passive", "driven"), (8., 32.)):
            index = [j for j, ep in enumerate(ids) if reference_modes[damping][int(ep)] == mode]
            y = truth[index, :round(horizon/reference_dt)+1]
            motion.append(dict(damping=damping, mode=mode, horizon_s=horizon, episodes=len(index),
                               episodes_with_abs_unwrapped_theta_above_pi=int(np.any(np.abs(y[..., 0])>np.pi, axis=1).sum()),
                               episodes_with_unwrapped_angle_range_above_two_pi=int((np.ptp(y[..., 0], axis=1)>2*np.pi).sum()),
                               max_abs_unwrapped_angle_rad=float(np.abs(y[..., 0]).max()),
                               last_quarter_omega_rms=float(np.sqrt(np.mean(y[:, -((len(y[0])-1)//4):, 1]**2)))))
    aggregate = []
    for model, damping, probe, mode, horizon in itertools.product(("lehami", "lewm"), (0., .2), ("ridge", "mlp"), ("passive", "driven"), (8., 32.)):
        selected = [r for r in rows if (r["model"], r["damping"], r["probe"], r["mode"], r["horizon_s"]) == (model, damping, probe, mode, horizon)]
        assert len(selected) == 3
        scores = {}
        for quantity, metric in itertools.product(("angle", "omega"), selected[0]["metrics"]["angle"]):
            values = [r["metrics"][quantity][metric] for r in selected]
            scores.setdefault(quantity, {})[metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values)) if all(v is not None for v in values) else dict(mean=None, sample_sd=None)
        aggregate.append(dict(model=model, damping=damping, probe=probe, mode=mode, horizon_s=horizon, seeds=[0, 1, 2], metrics=scores))
    result = dict(schema_version=1, original_metrics_unchanged=True, source_sha256=source_files,
                  checks=dict(self_tests=True, all_12_runs_present=True, identical_truth_across_models_and_seeds_per_damping=True,
                              same_test_episode_ids_across_damping=True, reconstructed_original_metric_values=matched_metric_rows,
                              all_reference_amplitudes_nonzero=all(v["nrmse"] is not None for r in rows for v in r["metrics"].values())),
                  definitions={
                      "angle_error": "wrap(predicted theta - true theta) in [-pi,pi]",
                      "pooled_rmse": "sqrt(mean(error squared over all selected episodes and future times)); excludes anchor",
                      "mean_episode_rmse": "mean over episodes of sqrt(mean time error squared); matches original report and differs from pooled_rmse",
                      "equilibrium_reference_rms": "RMSE of always theta=0, omega=0; angle denominator uses wrap(true theta), not unwrapped multi-turn angle",
                      "nrmse": "pooled_rmse / equilibrium_reference_rms; values below1 beat the equilibrium predictor,1 tie,above1 worse",
                      "mse_skill_vs_equilibrium": "1 - nrmse squared; zero ties equilibrium,positive improves,negative worsens",
                      "oracle_true_initial_state_persistence": "constant true theta/omega at anchor; privileged diagnostic, not a visually initialized competing learned model",
                      "seed_aggregation": "Each seed contributes one metric; mean and sample SD across3 seeds, including readout randomness",
                      "normalization_boundary": "Condition/horizon test-truth RMS is a post-hoc descriptive scale, not a training statistic or a replacement primary metric. For future prespecified comparisons, additionally use a common scale fixed from training data and report absolute RMSE plus baseline skill.",
                      "pointwise_relative_error": "Not used: theta and omega cross zero and decay toward zero, so per-time division is singular or unstable.",
                      "paired_design_boundary": "Same test episode IDs and data-seed configuration across damping; anchor is0.4s after initialization and therefore its true state already differs with damping."},
                  true_motion_diagnostics=motion, aggregate=aggregate, per_seed=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), runs=len(identifiers), checks=result["checks"])))
    for r in aggregate:
        if r["model"] == "lewm" and r["probe"] == "mlp":
            print(r["damping"], r["mode"], r["horizon_s"], {k: {m: round(v["mean"], 4) for m, v in values.items()} for k, values in r["metrics"].items()})


if __name__ == "__main__":
    main()
