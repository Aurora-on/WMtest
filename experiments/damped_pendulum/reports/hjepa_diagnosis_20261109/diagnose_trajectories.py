"""Read saved validation arrays only; no training, refit, or model edits."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports/method_comparison_20261109/data/runs"
OUT = Path(__file__).with_name("trajectory_diagnostics.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x, dtype=float).ravel()
    if not len(x):
        return None
    assert np.isfinite(x).all()
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                p10=float(np.quantile(x, .1)), p90=float(np.quantile(x, .9)),
                max=float(x.max()))


def norm(x):
    return np.sqrt(np.sum(np.asarray(x)**2, axis=-1))


def variance_decomposition(x):
    trajectory_mean = x.mean(1)
    global_mean = x.mean((0, 1))
    within = float(np.mean((x - trajectory_mean[:, None])**2))
    between = float(np.mean((trajectory_mean - global_mean)**2))
    total = float(np.mean((x - global_mean)**2))
    assert np.isclose(within + between, total)
    return dict(within_trajectory_variance=within, between_trajectory_variance=between,
                total_variance=total, within_fraction=within/total,
                between_fraction=between/total)


def latent(pred, teacher):
    inc, tinc = norm(np.diff(pred, axis=1)), norm(np.diff(teacher, axis=1))
    pn, tn = norm(pred), norm(teacher)
    return dict(
        predicted_norm=stats(pn), observed_norm=stats(tn),
        final_predicted_to_observed_norm_ratio=stats(pn[:, -1]/np.maximum(tn[:, -1], 1e-12)),
        predicted_increment=stats(inc), observed_increment=stats(tinc),
        predicted_increment_excluding_first=stats(inc[:, 1:]),
        observed_increment_excluding_first=stats(tinc[:, 1:]),
        increment_median_by_step=np.median(inc, axis=0).tolist(),
        observed_increment_median_by_step=np.median(tinc, axis=0).tolist(),
        per_trajectory_increment_rms_ratio=stats(
            np.sqrt(np.mean(inc**2, axis=1)/np.maximum(np.mean(tinc**2, axis=1), 1e-24))),
        predicted_variance=variance_decomposition(pred), observed_variance=variance_decomposition(teacher),
        future_mse=float(np.mean((pred[:, 1:] - teacher[:, 1:])**2)),
        constant_anchor_future_mse=float(np.mean((teacher[:, :1] - teacher[:, 1:])**2)),
    )


def differences(state):
    dtheta = np.diff(state[..., 0], axis=1)
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    return dtheta, np.diff(state[..., 1], axis=1)


def longest_run(row):
    best = current = 0
    end = -1
    for i, flag in enumerate(row):
        current = current + 1 if flag else 0
        if current > best:
            best, end = current, i
    return best, end-best+1, end+1


def physical(states, truth, omega_scale):
    dtheta, domega = differences(states)
    repeated = (np.abs(dtheta) <= 1e-12) & (np.abs(domega) <= 1e-12)
    angle_static = np.abs(dtheta) <= 1e-12
    omega_mid = (states[:, 1:, 1] + states[:, :-1, 1])/2
    truth_active = np.mean(np.lib.stride_tricks.sliding_window_view(
        truth[:, 1:, 1]**2, 10, axis=1), axis=-1) > (.1*omega_scale)**2
    frozen_window = np.lib.stride_tricks.sliding_window_view(angle_static, 10, axis=1).all(-1)
    dynamic_frozen = truth_active & frozen_window
    runs = np.asarray([longest_run(row)[0] for row in angle_static])
    return dict(
        exact_state_repeat_transition_fraction=float(repeated.mean()),
        exact_angle_repeat_transition_fraction=float(angle_static.mean()),
        angle_static_but_decoded_speed_active_transition_fraction=float(
            (angle_static & (abs(omega_mid) > .1*omega_scale)).mean()),
        angle_static_at_least_1s_trajectory_count=int((runs >= 10).sum()),
        angle_static_at_least_1s_dynamic_trajectory_count=int(dynamic_frozen.any(1).sum()),
        angle_static_at_least_1s_dynamic_trajectory_fraction=float(dynamic_frozen.any(1).mean()),
        longest_exact_angle_plateau_seconds=float(runs.max()*.1),
        finite_difference_vs_decoded_omega_rmse=float(np.sqrt(np.mean((dtheta/.1-omega_mid)**2))),
        finite_difference_vs_decoded_omega_nrmse=float(np.sqrt(np.mean((dtheta/.1-omega_mid)**2))/omega_scale),
    )


def compact(z, mask, scale):
    f, t = z["features"][mask], z["teacher_features"][mask]
    pred, teacher, truth = (z[key][mask] for key in ("pred", "teacher", "truth"))
    pd, td = z["nearest_fit_distance"][mask], z["teacher_nearest_fit_distance"][mask]
    repeat = np.max(np.abs(np.stack(differences(pred), axis=-1)), axis=-1) <= 1e-12
    increments = norm(np.diff(f, axis=1))
    result = dict(
        trajectories=int(mask.sum()), feature_dimension=int(f.shape[-1]),
        concatenated_features=latent(f, t),
        nearest_fit_distance=dict(predicted_future=stats(pd[:, 1:]), observed_future=stats(td[:, 1:]),
                                 predicted_final=stats(pd[:, -1]), observed_final=stats(td[:, -1]),
                                 final_predicted_to_observed_ratio=stats(pd[:, -1]/np.maximum(td[:, -1], 1e-12))),
        predicted_physical=physical(pred, truth, scale),
        observed_physical=physical(teacher, truth, scale),
        truth_kinematic_discretization_control=physical(truth, truth, scale),
        latent_increment_while_physical_readout_repeats=stats(increments[repeat]),
    )
    return result


def examples(z, max_examples=3):
    dtheta, domega = differences(z["pred"])
    repeats = (abs(dtheta) <= 1e-12) & (abs(domega) <= 1e-12)
    runs = [longest_run(row) for row in repeats]
    chosen = sorted(range(len(runs)), key=lambda i: (-runs[i][0], i))[:max_examples]
    output = []
    for i in chosen:
        count, start, end = runs[i]
        if count < 10:
            continue
        state = z["features"][i, :, :32]
        window = z["pred"][i, start:end+1]
        output.append(dict(index=i, episode=int(z["episode"][i]), group_id=int(z["group_id"][i]),
                           damping=float(z["damping"][i]), plateau_start_seconds=start*.1,
                           plateau_end_seconds=end*.1, plateau_duration_seconds=count*.1,
                           fixed_decoded_theta_omega=window[0].tolist(),
                           latent_state_endpoint_displacement=float(norm(state[end]-state[start])),
                           latent_state_path_length=float(norm(np.diff(state[start:end+1], axis=0)).sum()),
                           nearest_fit_distance_range=[float(z["nearest_fit_distance"][i, start:end+1].min()),
                                                       float(z["nearest_fit_distance"][i, start:end+1].max())],
                           truth_omega_rms=float(np.sqrt(np.mean(z["truth"][i, start:end+1, 1]**2)))))
    return output


def main():
    result = dict(
        schema_version=1, methods={}, diagnostic_script_sha256=sha(Path(__file__)),
        scope="Saved seed0 val trajectories, 21-frame history, future32 at dt=.1; no model execution, no fitting or training",
        definitions={
            "raw_latent_norms": "Only compare predicted/observed within each model; coordinate scales differ across methods",
            "variance": "Population mean square per coordinate; total = within time per trajectory + between trajectory means; all33 time points equally weighted",
            "exact_plateau": "Adjacent wrapped angle and/or speed differences <=1e-12; >=1s means >=10 consecutive intervals",
            "dynamic_plateau": "Exact angle plateau overlaps 1s truth speed RMS >.1 times train-fit omega scale",
            "kinematics": "RMSE(wrapped(theta[t+1]-theta[t])/.1 - (omega[t+1]+omega[t])/2); truth value controls discretization",
            "hjepa_state": "First32 features are control state; last32 are model-specific causal state increment, NOT physical omega",
            "hjepa_observed_delta_caveat": "Observed-history pre-anchor repeats leftmost frame; each history is rebuilt. Delta differences must not be substituted for state-dynamics evidence",
            "distance": "Stored nearest-fit distance is Euclidean after the original train-only standardization",
            "readout_limit": "Readout.npz and neighbor IDs are absent locally; exact repeated decoded values support quantization but identical neighbor IDs cannot be directly verified",
        })
    for method in ("lehami", "lewm", "hjepa"):
        result["methods"][method] = {}
        for stage in ("a", "single", "multi"):
            directory = SOURCE / method / (method + "_seed0") / stage / "evaluation"
            summary = json.loads((directory/"summary.json").read_text())
            prov = json.loads((directory/"readout_provenance.json").read_text())
            with np.load(directory/"val/trajectories.npz", allow_pickle=False) as data:
                z = {key: data[key] for key in data.files}
            assert all(np.isfinite(z[key]).all() for key in ("features", "teacher_features", "pred", "teacher", "truth"))
            assert np.all(z["valid_steps"] == 32)
            scale = summary["omega_nrmse_denominator"]
            record = compact(z, np.ones(len(z["pred"]), dtype=bool), scale)
            record.update(selected_k=prov["calibration"]["selected_k"],
                          source_hashes={str((directory/name).relative_to(ROOT)): sha(directory/name)
                                         for name in ("summary.json", "readout_provenance.json", "val/trajectories.npz")},
                          checkpoint_sha256=summary["checkpoint_sha256"],
                          teacher_quality_gate_passed=summary["evaluation"]["val"]["rollout"]["32"]["teacher_quality_gate_passed"],
                          official_velocity_stop_metric=summary["evaluation"]["val"]["rollout"]["32"]["groups"]["all"]["metrics"])
            if method == "hjepa":
                f, t = z["features"], z["teacher_features"]
                record["state32"] = latent(f[..., :32], t[..., :32])
                record["delta32"] = latent(f[..., 32:], t[..., 32:])
                record["predicted_delta_matches_predicted_state_increment_max_error"] = float(
                    np.max(abs(f[:, 1:, 32:] - np.diff(f[..., :32], axis=1))))
                record["observed_delta_vs_adjacent_rebuilt_state_increment_rmse"] = float(
                    np.sqrt(np.mean((t[:, 1:, 32:] - np.diff(t[..., :32], axis=1))**2)))
                record["plateau_examples"] = examples(z)
                record["by_damping"] = {}
                for b in sorted(np.unique(z["damping"])):
                    mask = z["damping"] == b
                    part = compact(z, mask, scale)
                    part.pop("concatenated_features")
                    part["state32"] = latent(f[mask, :, :32], t[mask, :, :32])
                    part["delta32"] = latent(f[mask, :, 32:], t[mask, :, 32:])
                    record["by_damping"][str(b)] = part
            result["methods"][method][stage] = record
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
