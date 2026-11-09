"""Readout and metadata checks only: no training, GPU use or dataset access."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('animation_evaluator', Path(__file__).with_name('evaluate_for_animation.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ridge_minimizes_mean_mse_with_unpenalized_intercept():
    rng = np.random.default_rng(5705)
    x = rng.normal(size=(24, 4)) + 3
    y = x @ rng.normal(size=(4, 3)) + np.array([10., -20., 30.])
    alpha = .3
    weights = module.ridge_coefficients(x, y, alpha)
    residual = x @ weights[:-1] + weights[-1] - y
    # Stationarity of the declared objective, independently differentiated.
    gradient_w = 2 * x.T @ residual / y.size + 2 * alpha * weights[:-1]
    gradient_b = 2 * residual.sum(0) / y.size
    np.testing.assert_allclose(gradient_w, 0, atol=1e-12)
    np.testing.assert_allclose(gradient_b, 0, atol=1e-12)
    shifted = module.ridge_coefficients(x, y + 100, alpha)
    np.testing.assert_allclose(shifted[:-1], weights[:-1], atol=1e-10)
    np.testing.assert_allclose(shifted[-1], weights[-1] + 100, atol=1e-10)


def test_partition_and_animation_selection_are_order_independent_and_metadata_only():
    records = [dict(episode=i, group_id=1000 + i, energy_band=i % 4) for i in range(384)]
    fit, held = module.split_training_records(records)
    fit_reverse, held_reverse = module.split_training_records(records[::-1])
    assert fit == fit_reverse and held == held_reverse
    assert len(fit) == 288 and len(held) == 96
    assert {r['group_id'] for r in fit}.isdisjoint(r['group_id'] for r in held)
    assert {r['group_id'] for r in fit + held} == {r['group_id'] for r in records}
    selected = module.select_animation_records(records)
    changed = [dict(row, prediction_error=1000 - row['episode']) for row in records[::-1]]
    assert [r['episode'] for r in selected] == [0, 1, 2, 3]
    assert [r['episode'] for r in module.select_animation_records(changed)] == [0, 1, 2, 3]


def test_ridge_standardization_uses_fit_only_and_failed_predictions_remain_nan():
    rng = np.random.default_rng(5707)
    x = rng.normal(size=(100, 4))
    y = rng.normal(size=(100, 3))
    xv, yv = rng.normal(size=(30, 4)) + 100, rng.normal(size=(30, 3)) - 100
    readout, alpha, candidates = module.choose_ridge(x, y, xv, yv)
    np.testing.assert_allclose(readout['x_mean'], x.mean(0))
    np.testing.assert_allclose(readout['y_mean'], y.mean(0))
    assert alpha == min(candidates, key=lambda row: row['selection_normalized_mse'])['alpha']
    features = np.stack([x[:3], np.full((3, 4), np.nan)])
    prediction = module.predict_readout(readout, features)
    assert np.isfinite(prediction[0]).all()
    assert np.isnan(prediction[1]).all()
    truth = np.array([[[0., 100.]]])
    pred = np.array([[[0., 102.]]])
    metrics = module.physical_metrics(pred, truth, omega_scale=4.)
    assert metrics['omega_nrmse'] == .5  # fixed fit scale, not val RMS100
