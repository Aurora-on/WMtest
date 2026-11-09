"""Evaluation-only tests: causality, strict failure coverage, and matched support."""
import h5py
import numpy as np
import pytest
import torch

from dpbench.dissipative_core import SolverError
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.evaluate_passive import (
    EVALUATION_DIAGNOSTICS, autonomous_rollout, baseline_predictions,
    calibration_pairs, numerical_audit, summary_metrics,
)


@pytest.mark.parametrize('core_options', [{}, {'integrator':'frozen_dg', 'solver':'picard', 'substeps':3}])
def test_stepwise_rollout_matches_frozen_model_without_future_images(core_options):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    model = DissipativeWorldModel({"encoder_width": 8, "history": 5,
        "environment_context_dim": 2, "sigreg_projections": 4,
        "core": {"hidden_dim": 8, **core_options}}).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    history = torch.rand(2, 5, 3, 64, 64)
    past, actions = torch.zeros(2, 4, 1), torch.zeros(2, 4, 1)
    out = autonomous_rollout(model, history, past, torch.full_like(past, .1),
                             actions, torch.full_like(actions, .1), EVALUATION_DIAGNOSTICS)
    expected = model.rollout_features(history, actions, past).numpy()
    np.testing.assert_allclose(out["features"], expected, atol=1e-12)
    assert out["valid_steps"].tolist() == [4, 4]
    assert not out["failures"]
    assert np.isfinite(out["context"]).all()
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    audit, ledger = numerical_audit(out['latent_energy'], out['diagnostics'], out['valid_steps'])
    assert audit['passed_on_valid_prefixes'] and audit['audited_finite_transitions'] == 8
    assert np.max(np.abs(ledger)) < 1e-12
    assert audit['minimum_dissipation'] >= 0


class FailingCore:
    def energy(self, q, p):
        return (q.square() + p.square()).sum(-1)

    def __call__(self, q, p, actions, dt, context):
        # Identify source rows even after a failed row has been removed.
        assert torch.equal(p, context)
        if (q[:, 0] >= 2).any():
            raise SolverError("synthetic strict failure")
        info = {k:torch.zeros(len(q)) for k in
                ("solver_residual", "energy_residual", "dissipation", "solver_iterations")}
        return q + 1, p, info


class FailingModel:
    q_dim = 1
    core = FailingCore()

    def initialize(self, history, past_actions, past_dt):
        return torch.tensor([[2.], [0.]]), torch.tensor([[7.], [11.]]), None, None

    def infer_context(self, encoded, past_actions, past_dt):
        return torch.tensor([[7.], [11.]])


def test_failed_sample_keeps_prefix_and_other_sample_continues_same_context():
    out = autonomous_rollout(FailingModel(), None, None, None,
                             torch.zeros(2, 4, 1), torch.full((2, 4, 1), .1))
    assert out["valid_steps"].tolist() == [0, 2]
    assert np.isnan(out["features"][0, 1:]).all()
    assert np.isnan(out["features"][1, 3:]).all()
    assert np.isfinite(out["features"][1, :3]).all()
    assert [(x["batch_index"],x["failed_step"]) for x in out["failures"]] == [(0,1),(1,3)]


def test_calibration_has_no_test_or_future_prefix_leakage(tmp_path):
    path = tmp_path / 'sample.h5'
    with h5py.File(path, 'w') as f:
        f['ep_len'] = [81, 81, 341]
        f['splits/train'] = [0]
        f['splits/val'] = [1]
        f['splits/test'] = [2]
    pairs = calibration_pairs(path, 'train', history=21, cap=81, maximum=20000, seed=0)
    assert pairs == [(0,t) for t in range(20,81)]
    assert all(ep == 1 for ep,t in calibration_pairs(path,'val',21,81,10,0))
    with pytest.raises(ValueError, match='train or val'):
        calibration_pairs(path, 'test', 21, 81, 10, 0)


def test_metrics_report_failure_coverage_and_matched_support():
    truth = np.zeros((2,3,2))
    model = truth.copy(); model[0,:,0] = 1.; model[1,2:] = np.nan
    baseline = truth.copy(); baseline[1,:,0] = 2.
    records = [dict(family='seen', damping=.1,energy_band=0)] * 2
    rows = summary_metrics({'model':model,'persistence':baseline},truth,np.ones(2),records,
        [.1,.2],.1,dict(mass=1.,length=1.,gravity=9.81),np.array([2,1]))
    select = lambda method,support,h: next(r for r in rows if r['grouping']=='all'
        and r['method']==method and r['support']==support and r['horizon_s']==h)
    assert select('model','all_available',.1)['coverage']==1
    assert select('model','all_available',.2)['coverage']==.5
    assert select('persistence','all_available',.2)['n_valid']==2
    assert select('persistence','model_complete',.2)['n_valid']==1
    assert select('persistence','model_complete',.2)['angle_rmse_rad']==0
    assert select('model','all_available',.2)['angle_rmse_rad']==1


def test_fixed_frame_baseline_preserves_anchor_and_holds_angle_only_in_future():
    model = np.array([[[.5, 2.], [.4, 1.], [np.nan, np.nan]]])
    truth = np.zeros_like(model)
    predictions = baseline_predictions(model, truth)
    np.testing.assert_array_equal(predictions['fixed_frame'][0], [[.5, 2.], [.5, 0.], [.5, 0.]])
    np.testing.assert_array_equal(predictions['persistence'][0], [[.5, 2.]] * 3)
    assert np.isnan(model[0, 2]).all()  # Baseline construction never repairs model failures.


def test_numerical_audit_counts_only_valid_prefix_and_detects_corruption():
    storage = np.array([[1., .8, .5], [2., 2., np.nan]])
    valid_steps = np.array([2, 1])
    diagnostics = {key: np.array([[0., 0.], [0., np.nan]]) for key in EVALUATION_DIAGNOSTICS}
    diagnostics['dissipation'][0] = [.2, .3]
    audit, ledger = numerical_audit(storage, diagnostics, valid_steps)
    assert audit['passed_on_valid_prefixes']
    assert audit['declared_valid_transitions'] == audit['audited_finite_transitions'] == 3
    assert np.isnan(ledger[1, 1])
    diagnostics['work'][0, 0] = 1.
    audit, _ = numerical_audit(storage, diagnostics, valid_steps)
    assert not audit['passed_on_valid_prefixes']
    assert audit['ledger_identity_violation_count'] == 1
    assert audit['nonzero_passive_work_violation_count'] == 1
    diagnostics['work'][0, 0] = 0.
    diagnostics['solver_residual_ratio'][0, 0] = 2.
    diagnostics['dissipation'][0, 1] = -.3
    diagnostics['solver_iterations'][1, 0] = np.nan
    audit, _ = numerical_audit(storage, diagnostics, valid_steps)
    assert not audit['passed_on_valid_prefixes']
    assert audit['negative_dissipation_violation_count'] == 1
    assert audit['residual_ratio_violation_count'] == 1
    assert audit['invalid_prefix_diagnostic_count'] == 1
    empty, _ = numerical_audit(storage, diagnostics, np.zeros(2, dtype=int))
    assert not empty['passed_on_valid_prefixes'] and empty['max_abs_ledger_identity_error'] is None
