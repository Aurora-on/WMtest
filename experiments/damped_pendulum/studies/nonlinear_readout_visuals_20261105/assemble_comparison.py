"""Apply previously selected frozen readouts to saved observation/rollout states."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "control_visuals_20261105"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physics(y):
    return np.stack((np.arctan2(y[..., 0], y[..., 1]), y[..., 2]), axis=-1)


def metrics(prediction, truth, omega_scale):
    angle = np.arctan2(np.sin(prediction[..., 0] - truth[..., 0]),
                       np.cos(prediction[..., 0] - truth[..., 0]))
    omega_rmse = float(np.sqrt(np.mean((prediction[..., 1] - truth[..., 1]) ** 2)))
    return dict(angle_mae_rad=float(np.mean(np.abs(angle))),
                omega_rmse_rad_s=omega_rmse, omega_nrmse=omega_rmse / omega_scale)


def main():
    probe_path = PREVIOUS / "readout_diagnostic/summary.json"
    probe = json.loads(probe_path.read_text())
    rollout_meta_path = HERE / "rollout/summary.json"
    rollout_meta = json.loads(rollout_meta_path.read_text())
    old = np.load(PREVIOUS / "evaluation/trajectories.npz", allow_pickle=False)
    new = np.load(HERE / "rollout/trajectories.npz", allow_pickle=False)
    reference = np.load(PREVIOUS / "evaluation/animation_inputs.npz", allow_pickle=False)
    control_meta_path = PREVIOUS / "evaluation/summary.json"
    control_meta = json.loads(control_meta_path.read_text())
    assert probe["arms"]["control"]["checkpoint_sha256"] == control_meta["checkpoint_sha256"]
    assert probe["dataset_sha256"] == control_meta["dataset_sha256"]
    assert probe["arms"]["lehami"]["checkpoint_sha256"] == rollout_meta["checkpoint_sha256"]
    assert probe["dataset_sha256"] == rollout_meta["dataset_sha256"]
    assert rollout_meta["parameters_unchanged"] and rollout_meta["checkpoint_step"] == 10000
    for key in ("episode", "group_id", "energy_band", "time", "truth"):
        np.testing.assert_allclose(new[key], old[key], rtol=0, atol=0)
    selected = old["selected_indices"]
    assert np.array_equal(old["episode"][selected], reference["episode"])
    arrays = {key: reference[key] for key in
              ("true_rgb", "truth", "time", "energy_band", "group_id", "episode")}
    summary, full_outputs = {}, {}
    source_paths = [probe_path, rollout_meta_path, control_meta_path, HERE / "rollout/trajectories.npz",
                    PREVIOUS / "evaluation/trajectories.npz",
                    PREVIOUS / "evaluation/animation_inputs.npz"]
    for arm, trajectories in (("control", old), ("lehami", new)):
        feature_path = PREVIOUS / f"readout_diagnostic/{arm}_features.npz"
        source_paths.append(feature_path)
        fit = np.load(feature_path, allow_pickle=False)
        assert np.array_equal(fit["val_episodes"], old["episode"])
        assert np.array_equal(fit["val_group_ids"], old["group_id"])
        np.testing.assert_allclose(physics(fit["val_targets"]).reshape(64, 33, 2),
                                   np.stack((np.arctan2(np.sin(old["truth"][..., 0]),
                                                       np.cos(old["truth"][..., 0])),
                                             old["truth"][..., 1]), -1), atol=1e-12)
        mean, scale = fit["ridge_x_mean"], fit["ridge_x_scale"]
        np.testing.assert_allclose(mean, fit["fit_features"].mean(0))
        np.testing.assert_allclose(scale, np.maximum(fit["fit_features"].std(0), 1e-6))
        tree = cKDTree((fit["fit_features"] - mean) / scale)
        k = probe["arms"][arm]["knn"]["selected_k"]

        def decode(features, method):
            shape = features.shape[:-1]
            flat = features.reshape(-1, features.shape[-1])
            valid = np.isfinite(flat).all(1)
            output = np.full((len(flat), 3), np.nan)
            scaled = (flat[valid] - mean) / scale
            if method == "ridge":
                design = np.column_stack((scaled, np.ones(len(scaled))))
                # Five input terms only; explicit contraction avoids platform
                # BLAS floating-point status warnings while retaining FP64.
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    output[valid] = (np.einsum("ij,jk->ik", design, fit["ridge_weights"], optimize=False)
                                     * fit["ridge_y_scale"] + fit["ridge_y_mean"])
            else:
                _, indices = tree.query(scaled, k=k, workers=1)
                if k == 1:
                    indices = indices[:, None]
                output[valid] = fit["fit_targets"][indices].mean(1)
            if not np.isfinite(output[valid]).all():
                raise ValueError("Readout produced a nonfinite value for a valid latent state")
            return output.reshape(*shape, 3)

        for method in ("ridge", "knn"):
            teacher_y = decode(fit["val_features"], method)
            np.testing.assert_allclose(teacher_y, fit[f"{method}_prediction_targets"], atol=1e-12)
            teacher = physics(teacher_y).reshape(64, 33, 2)
            predicted = physics(decode(trajectories["features"], method))
            arrays[f"{arm}_teacher_{method}"] = teacher[selected]
            arrays[f"{arm}_rollout_{method}"] = predicted[selected]
            full_outputs[f"{arm}_teacher_{method}"] = teacher
            full_outputs[f"{arm}_rollout_{method}"] = predicted
            horizon_metrics = {}
            for horizon in (8, 32):
                complete = (trajectories["valid_steps"] >= horizon) & np.isfinite(predicted[:, :horizon + 1]).all((1, 2))
                horizon_metrics[str(horizon)] = dict(
                    n_total=64, n_complete=int(complete.sum()), seconds=horizon * .1,
                    metrics=metrics(predicted[complete, 1:horizon + 1], old["truth"][complete, 1:horizon + 1],
                                    float(fit["ridge_y_scale"][2])) if complete.any() else None)
            summary[f"{arm}_{method}"] = dict(
                observation_readout=probe["arms"][arm]["metrics"][method],
                autonomous_rollout=horizon_metrics)
        arrays[f"valid_{arm}"] = trajectories["valid_steps"][selected]
        full_outputs[f"valid_{arm}"] = trajectories["valid_steps"]
    np.savez_compressed(HERE / "comparison_inputs.npz", **arrays)
    np.savez_compressed(HERE / "all_readout_trajectories.npz", **full_outputs,
                        truth=old["truth"], episode=old["episode"], time=old["time"])
    meta = dict(checkpoint_step=10000, checkpoint_sha256={
        arm: probe["arms"][arm]["checkpoint_sha256"] for arm in ("control", "lehami")},
        dataset_sha256=probe["dataset_sha256"],
        selected_records=[{key: int(reference[key][i]) for key in ("energy_band", "group_id", "episode")}
                          for i in range(4)],
        selection="Same four fixed validation cases as prior figures; minimum group_id per initial energy band",
        source_sha256={str(path): sha(path) for path in source_paths},
        script_sha256=sha(__file__), metrics=summary,
        selected_knn_k={arm: probe["arms"][arm]["knn"]["selected_k"] for arm in ("control", "lehami")},
        readout_refitted=False, hyperparameters_changed=False, world_model_updates=0,
        scope="Observation readout sees a fresh true 21-frame history at each time. Autonomous rollout sees only initial21 frames. Both use the same frozen train-only readouts. Val diagnostics, not held-out test evaluation.",
        limitation="kNN maps a query to nearby training labels, even for off-manifold latent predictions. Better decoded images alone do not establish accurate latent dynamics.")
    (HERE / "comparison_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
