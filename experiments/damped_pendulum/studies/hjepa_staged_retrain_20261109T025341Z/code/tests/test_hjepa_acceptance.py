"""Acceptance must catch decoded freezes and condition-specific failures without training."""
import json
from pathlib import Path

import numpy as np
import pytest

from dpbench.comparison_protocol import digest, json_digest
from dpbench.hjepa_acceptance import main, qualify_evaluation


def write(path, value):
    Path(path).write_text(json.dumps(value))


def seal(directory):
    summary = json.loads((directory / "summary.json").read_text())
    identity = {key: summary[key] for key in ("checkpoint_sha256", "dataset_sha256",
                                             "protocol_sha256", "evaluation_protocol_sha256")}
    files = ["readout.npz", "readout_provenance.json", "summary.json", "val/summary.json", "val/trajectories.npz"]
    write(directory / "complete.json", dict(identity, status="completed",
          artifacts={name: digest(directory / name) for name in files}))


def evaluation(tmp_path, conditions=(.2,)):
    directory = tmp_path / "evaluation"
    (directory / "val").mkdir(parents=True)
    n = len(conditions) * 64
    time = np.arange(33) * .1
    damping = np.repeat(conditions, 64)
    group_ids = np.tile(np.arange(64), len(conditions))
    theta = .8 * time[None, :] + .005 * group_ids[:, None]
    truth = np.stack([theta, np.full_like(theta, .8)], -1)
    state = np.sin(time[None, :, None] * np.linspace(.7, 2., 32)[None, None, :]
                   + .005 * group_ids[:, None, None])
    observed = np.concatenate([state, np.diff(state, axis=1, prepend=state[:, :1])], -1)
    predicted_state = .95 * state + .05 * state[:, :1]
    predicted = np.concatenate([predicted_state, np.diff(predicted_state, axis=1, prepend=predicted_state[:, :1])], -1)
    arrays = dict(pred=truth.copy(), teacher=truth.copy(), truth=truth, features=predicted,
                  teacher_features=observed, valid_steps=np.full(n, 32, dtype=np.int64),
                  time=time, damping=damping, group_id=group_ids)
    np.savez(directory / "val/trajectories.npz", **arrays)
    np.savez(directory / "readout.npz", y_scale=np.ones(3), selected_k=5)
    protocol = dict(history=21, dt=.1, train_frames=81, horizons=[8, 16, 32], split="val",
                    fit_groups=288, selection_groups=96, future_metrics_exclude_anchor=True,
                    training_conditions=list(conditions))
    identity = dict(checkpoint_sha256="a" * 64, dataset_sha256="b" * 64,
                    protocol_sha256="c" * 64, evaluation_protocol_sha256=json_digest(protocol))
    result = dict(split="val", horizon_steps=32, rollout={str(h): {} for h in (8, 16, 32)})
    calibration = dict(fit_groups=288, selection_groups=96, feature_dimension=64, weights="uniform",
                       fit_group_ids=list(range(1000, 1288)), selection_group_ids=list(range(2000, 2096)),
                       selected_damping=list(conditions))
    provenance = dict(identity, evaluation_protocol=protocol, checkpoint_step=4000,
                      readout_sha256=digest(directory / "readout.npz"), calibration=calibration,
                      validation_used_for_fitting_or_selection=False, readout_reused_from_other_model=False,
                      group_ids={"val": list(range(64))})
    write(directory / "readout_provenance.json", provenance)
    summary = dict(identity, status="completed", checkpoint_step=4000, evaluation_protocol=protocol,
                   evaluation={"val": result}, model_config=dict(kind="hjepa", rank=32, history=21, dt=.1),
                   parameters_unchanged=True, world_model_updates=0, test_split_accessed=False,
                   readout_sha256=digest(directory / "readout.npz"),
                   readout_provenance_sha256=digest(directory / "readout_provenance.json"),
                   omega_nrmse_denominator=1.)
    write(directory / "summary.json", summary)
    write(directory / "val/summary.json", result)
    seal(directory)
    return directory, arrays


def reseal_arrays(directory, arrays):
    np.savez(directory / "val/trajectories.npz", **arrays)
    seal(directory)


def test_qualified_report_is_read_only_and_uses_state_not_delta(tmp_path):
    directory, arrays = evaluation(tmp_path)
    arrays["features"][..., 32:] = 1e5  # Diagnostic must not count delta channels as state motion.
    reseal_arrays(directory, arrays)
    before = {str(p): digest(p) for p in directory.rglob("*") if p.is_file()}
    report = qualify_evaluation(directory)
    assert report["qualified"] and report["status"] == "qualified"
    row = report["per_damping"]["0.2"]
    assert row["metrics"]["state"]["mse_over_constant_anchor"] == pytest.approx(.05 ** 2)
    assert report["omega_nrmse_denominator"] == 1.
    assert report["state_feature_slice"] == [0, 32]
    assert {str(p): digest(p) for p in directory.rglob("*") if p.is_file()} == before


def test_angle_freeze_with_nonzero_decoded_velocity_fails(tmp_path):
    directory, arrays = evaluation(tmp_path)
    arrays["pred"][..., 0] = arrays["pred"][:, :1, 0]
    assert np.all(arrays["pred"][..., 1] == .8)
    reseal_arrays(directory, arrays)
    report = qualify_evaluation(directory)
    assert not report["qualified"]
    row = report["per_damping"]["0.2"]
    assert row["gates"]["teacher_readout"] and row["gates"]["better_than_constant_state"]
    assert not row["gates"]["dynamic_angle_freeze"]
    assert row["metrics"]["physical_freeze"]["dynamic_angle_freeze_trajectory_fraction"] == 1.
    assert row["metrics"]["physical_freeze"]["static_angle_but_nonzero_speed_transition_fraction"] == 1.


def test_per_condition_teacher_failure_cannot_be_hidden_by_pooling(tmp_path):
    conditions = (0., .05, .1, .2, .5, 1.)
    directory, arrays = evaluation(tmp_path, conditions)
    arrays["teacher"][arrays["damping"] == 0., :, 1] += .3
    # A pooled omega NRMSE would be below the .15 gate.
    assert np.sqrt(np.mean((arrays["teacher"][:, 1:, 1] - arrays["truth"][:, 1:, 1]) ** 2)) < .15
    reseal_arrays(directory, arrays)
    report = qualify_evaluation(directory)
    assert not report["qualified"]
    assert report["per_damping"]["0"]["metrics"]["teacher"]["omega_nrmse"] == pytest.approx(.3)
    assert report["failed_checks"] == ["b=0: teacher_readout"]


def test_nonfinite_rollout_fails_coverage_and_keeps_finite_conditional_metrics(tmp_path):
    directory, arrays = evaluation(tmp_path)
    arrays["valid_steps"][0] = 4
    arrays["features"][0, 5:] = np.nan
    arrays["pred"][0, 5:] = np.nan
    reseal_arrays(directory, arrays)
    report = qualify_evaluation(directory)
    assert not report["qualified"]
    assert report["per_damping"]["0.2"]["metrics"]["coverage"] == 63 / 64
    json.dumps(report, allow_nan=False)


def test_constant_latent_predictor_fails_even_with_perfect_physical_readout(tmp_path):
    directory, arrays = evaluation(tmp_path)
    arrays["features"][:, :, :32] = arrays["teacher_features"][:, :1, :32]
    reseal_arrays(directory, arrays)
    report = qualify_evaluation(directory)
    assert report["failed_checks"] == ["b=0.2: better_than_constant_state"]


def test_teacher_angle_error_is_wrapped(tmp_path):
    directory, arrays = evaluation(tmp_path)
    arrays["teacher"][..., 0] += 2 * np.pi
    reseal_arrays(directory, arrays)
    report = qualify_evaluation(directory)
    assert report["qualified"]
    assert report["per_damping"]["0.2"]["metrics"]["teacher"]["angle_mae_rad"] < 1e-14


@pytest.mark.parametrize("kind", ["hash", "missing", "scale", "conditions"])
def test_invalid_evaluation_fails_closed(tmp_path, kind):
    directory, arrays = evaluation(tmp_path)
    if kind == "hash":
        (directory / "val/trajectories.npz").write_bytes(b"corrupt")
    elif kind == "missing":
        (directory / "readout.npz").unlink()
    elif kind == "scale":
        summary = json.loads((directory / "summary.json").read_text())
        summary["omega_nrmse_denominator"] = 100.
        write(directory / "summary.json", summary)
        seal(directory)
    else:
        arrays["damping"][:] = 0.
        reseal_arrays(directory, arrays)
    report_file = tmp_path / "invalid.json"
    assert main(["--evaluation", str(directory), "--output", str(report_file)]) == 2
    report = json.loads(report_file.read_text())
    assert not report["qualified"] and report["status"] == "invalid_evaluation"


def test_cli_exit_codes_and_output_protection(tmp_path):
    directory, arrays = evaluation(tmp_path)
    output = tmp_path / "acceptance.json"
    assert main(["--evaluation", str(directory), "--output", str(output)]) == 0
    with pytest.raises(SystemExit) as exc:
        main(["--evaluation", str(directory), "--output", str(output)])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        main(["--evaluation", str(directory), "--output", str(directory / "extra.json")])
    arrays["pred"][..., 0] = arrays["pred"][:, :1, 0]
    reseal_arrays(directory, arrays)
    assert main(["--evaluation", str(directory)]) == 2
