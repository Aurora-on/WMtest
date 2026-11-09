import json
from pathlib import Path

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from dpbench.data import WindowDataset
from dpbench.generate import derivative, energy, generate, make_plan, simulate


def test_independent_reference_and_energy():
    dt = .1
    initial = np.array([[1.7, -.6]])
    actions = np.r_[np.full(20, 1.5), np.zeros(20), np.full(20, -1.)][None]
    for damping in (0., .2):
        states, work, diss = simulate(initial, actions, dt, 20, damping=damping)
        e = energy(states, 1., 1., 9.81)
        assert np.max(np.abs(np.diff(e, axis=1)-work[:, :-1]+diss[:, :-1])) < 1e-6
        y = np.r_[initial[0], 0., 0.]
        for k, u in enumerate(actions[0]):
            y = solve_ivp(lambda t, z: derivative(z, u, 1., 1., 9.81, damping),
                          (0, dt), y, method="DOP853", rtol=1e-12, atol=1e-13).y[:, -1]
            np.testing.assert_allclose(states[0, k+1], y[:2], atol=2e-6)
        assert np.all(diss >= 0)
        if damping == 0:
            assert np.count_nonzero(diss) == 0


def test_passive_decay_and_actuation():
    s, w, d = simulate([[1., 0.]], np.zeros((1, 100)), .1, 20, damping=.2)
    e = energy(s, 1, 1, 9.81)
    assert np.max(np.diff(e)) < 1e-8
    assert e[0, -1] < e[0, 0]*.3
    assert not w.any()
    s, w, _ = simulate([[0., 0.]], np.array([[2., 0.]]), .1, 20, damping=.2)
    assert s[0, 1, 1] > 0 and w[0, 0] > 0 and w[0, 1] == 0


def test_splits_padding_and_training_input(tmp_path):
    c = json.loads((Path(__file__).parents[1]/"configs/smoke.json").read_text())
    c.update(episodes=16, frames=21, train_frames=17)
    manifest = generate(c, tmp_path)
    with h5py.File(tmp_path/manifest["datasets"][1]["file"], "r") as f:
        sets = [set(f["splits/"+k][:]) for k in ("train", "val", "test")]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        assert len(set.union(*sets)) == c["episodes"]
        for off, n in zip(f["ep_offset"][:], f["ep_len"][:]):
            assert not f["valid_transition"][off+n-1]
            assert f["action"][off+n-1, 0] == 0
            assert f["work"][off+n-1] == 0
            assert f["dissipation"][off+n-1] == 0
    ds = WindowDataset(tmp_path/manifest["datasets"][1]["file"], "train", 15)
    assert set(ds[0]) == {"pixels", "action", "dt_frame"}
    assert ds[0]["pixels"].shape == (15, 3, 64, 64)
    for ep, start in ds.index:
        assert start-ep*c["frames"]+15 <= c["train_frames"]
    p0, a0, _, split0 = make_plan(c)
    p1, a1, _, split1 = make_plan(c)
    np.testing.assert_array_equal(p0, p1)
    np.testing.assert_array_equal(a0, a1)
    assert split0 == split1
