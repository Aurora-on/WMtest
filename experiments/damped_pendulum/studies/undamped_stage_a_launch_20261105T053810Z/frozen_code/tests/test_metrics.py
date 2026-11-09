import numpy as np
from dpbench.evaluate import metrics
from dpbench.generate import simulate


def test_reference_trajectory_and_wrapped_angle_metrics():
    s, w, d = simulate([[1.8, -.5]], np.ones((1, 80)), .1, 20, damping=.2)
    pred = s[0].copy()
    pred[:, 0] += 2*np.pi
    meta = dict(mass=1., length=1., gravity=9.81)
    out = metrics(pred, s[0], w[0], d[0], meta, passive=False)
    assert out["angle_rmse_rad"] < 1e-12
    assert out["omega_rmse_rad_s"] == 0
    assert out["energy_rmse_J"] < 1e-12
    assert out["reference_budget_rmse_J"] < 1e-6
    assert out["passive_energy_increase_fraction"] is None


def test_collapse_cannot_pass_trajectory_metrics():
    s, w, d = simulate([[1.8, -.5]], np.zeros((1, 80)), .1, 20, damping=.2)
    meta = dict(mass=1., length=1., gravity=9.81)
    out = metrics(np.zeros_like(s[0]), s[0], w[0], d[0], meta, passive=True)
    assert out["passive_energy_increase_fraction"] == 0
    assert out["angle_rmse_rad"] > .1
    assert out["omega_rmse_rad_s"] > .1
    assert out["energy_rmse_J"] > 1
