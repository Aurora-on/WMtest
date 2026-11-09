"""Rendering regressions and data provenance; no models or optimizer updates.

Continuity claims apply to float coverage. Uint8 export necessarily retains
one-level quantization boundaries; it is not a differentiable observation map.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from dpbench.generate import render, render_smooth_float
from dpbench.passive_family import generate_family, plan, sample_groups


ROOT = Path(__file__).resolve().parents[1]


def test_default_and_explicit_legacy_preserve_historical_pixels():
    angles = [-2.5, -.1, -1e-12, 0., 1e-12, .1, 2.5]
    default = np.stack([render(theta) for theta in angles])
    explicit = np.stack([render(theta, renderer="legacy") for theta in angles])
    np.testing.assert_array_equal(default, explicit)
    # Captured from the unmodified legacy renderer before adding smooth_v1.
    assert hashlib.sha256(default.tobytes()).hexdigest() == (
        "36774553af95ccd08e75c0845626ddd81a4d7d23a445a7429b66cb0b808e30e4")


def test_zero_angle_no_longer_has_legacy_geometry_jump():
    legacy = [render(t) for t in (-1e-12, 1e-12)]
    assert np.any(legacy[0] != legacy[1], axis=-1).sum() == 30
    smooth = [render(t, renderer="smooth_v1") for t in (-1e-12, 0., 1e-12)]
    np.testing.assert_array_equal(smooth[0], smooth[1])
    np.testing.assert_array_equal(smooth[1], smooth[2])
    assert smooth[0].dtype == np.uint8 and smooth[0].shape == (64, 64, 3)


@pytest.mark.parametrize("theta", [-2.5, -.3, 0., .4, 2.5])
def test_float_coverage_varies_continuously_with_subpixel_motion(theta):
    # Bound changes in the actual observation, across circle/line edge pixels.
    # Do not assert global continuity for the quantized uint8 export.
    for epsilon in (1e-4, 1e-6, 1e-8, 1e-10):
        lo = render_smooth_float(theta - epsilon)
        hi = render_smooth_float(theta + epsilon)
        assert np.max(np.abs(hi - lo)) < 100. * epsilon
        assert np.isfinite(hi).all() and hi.min() >= 0. and hi.max() <= 1.


def test_smoothing_keeps_visible_motion_and_repeatable_static_frames():
    at_rest = render(0., renderer="smooth_v1")
    np.testing.assert_array_equal(at_rest, render(0., renderer="smooth_v1"))
    moving = render(.1, renderer="smooth_v1")
    assert np.any(at_rest != moving, axis=-1).sum() > 50
    assert np.sqrt(np.mean((at_rest.astype(float) - moving) ** 2)) > 5.
    float_frame = render_smooth_float(.1)
    # Quantization is bounded, not removed by antialiasing.
    assert np.max(np.abs(moving / 255. - float_frame)) <= .5 / 255. + 1e-15


def _tiny_config():
    cfg = json.loads((ROOT / "configs/passive_family_smoke.json").read_text())
    cfg.update(history=3, train_frames=6, test_frames=8)
    return cfg


def test_renderer_version_flows_through_plan_manifest_and_hdf5(tmp_path):
    legacy_config = _tiny_config()
    smooth_config = dict(legacy_config, renderer="smooth_v1")
    assert plan(legacy_config)["renderer"] == "legacy"
    assert plan(smooth_config)["renderer"] == "smooth_v1"
    old_groups, new_groups = sample_groups(legacy_config), sample_groups(smooth_config)
    for split in old_groups:
        for key in old_groups[split]:
            np.testing.assert_array_equal(old_groups[split][key], new_groups[split][key])
    legacy = generate_family(legacy_config, tmp_path / "legacy")
    smooth = generate_family(smooth_config, tmp_path / "smooth")
    assert legacy["renderer"] == "legacy"
    assert smooth["renderer"] == "smooth_v1"
    disk_manifest = json.loads((tmp_path / "smooth/manifest.json").read_text())
    assert disk_manifest["renderer"] == disk_manifest["plan"]["renderer"] == "smooth_v1"
    for entry in smooth["datasets"]:
        assert entry["renderer"] == "smooth_v1"
        with h5py.File(tmp_path / "smooth" / entry["file"]) as new, h5py.File(
                tmp_path / "legacy" / entry["file"]) as old:
            assert new.attrs["renderer"] == "smooth_v1" and old.attrs["renderer"] == "legacy"
            assert json.loads(new.attrs["config_json"])["renderer"] == "smooth_v1"
            # Renderer changes neither simulation nor group/split/clock contracts.
            for key in ("state", "energy", "work", "dissipation", "reference_residual",
                        "initial_state", "group_id", "damping", "energy_band", "regime_id",
                        "ep_offset", "ep_len", "action", "dt_frame", "valid_transition",
                        "splits/train", "splits/val", "splits/test"):
                np.testing.assert_array_equal(new[key][:], old[key][:])
            assert new["pixels"].dtype == np.uint8
            np.testing.assert_array_equal(new["pixels"][0], render(new["state"][0, 0], renderer="smooth_v1"))
            np.testing.assert_array_equal(old["pixels"][0], render(old["state"][0, 0]))
            assert not np.array_equal(new["pixels"][0], old["pixels"][0])
    with pytest.raises(FileExistsError):
        generate_family(smooth_config, tmp_path / "smooth")


def test_invalid_renderer_rejected_before_generation(tmp_path):
    with pytest.raises(ValueError, match="renderer"):
        generate_family(dict(_tiny_config(), renderer="smooth_typo"), tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
    with pytest.raises(ValueError, match="renderer"):
        render(0., renderer="smooth_typo")
