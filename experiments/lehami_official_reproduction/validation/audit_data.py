"""Read-only audit of the author's conservative multi-rod HDF5 and split.

Exit codes: 0 = semantic checks and historical byte hash match; 2 = failed or
incomplete semantic checks; 3 = semantic checks pass but historical bytes differ.
A byte mismatch never rewrites the manifest and does not establish equivalence
of the generated trajectory/image contents to the author's historical file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np


MANIFEST_SHA256 = "740482ae922bea77ebb1a0f39f15ebe115ccb6bf6fdd0048658b45236b1821b1"
DATA_SHA256 = "4786e18cf6463830bcf3095c5fee0098b16add3efe407381ec065305b40704d8"
SOURCE_COMMIT = "f76f65f8b99afb6b7e3063f573645d3e2f290ede"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(data, manifest):
    report = dict(schema_version=1, source_commit=SOURCE_COMMIT,
                  data=str(Path(data).resolve()), manifest=str(Path(manifest).resolve()),
                  checks={}, historical_byte_match=False, semantic_checks_passed=False)

    def check(name, value, detail=None):
        report["checks"][name] = dict(passed=bool(value))
        if detail is not None:
            report["checks"][name]["detail"] = detail

    def close():
        report["semantic_checks_passed"] = bool(report["checks"]) and all(
            c["passed"] for c in report["checks"].values())
        if not report["semantic_checks_passed"]:
            status, code = "SEMANTIC_CHECKS_FAILED", 2
        elif not report["historical_byte_match"]:
            status, code = "SEMANTIC_CHECKS_PASS_BYTE_MISMATCH", 3
        else:
            status, code = "SEMANTIC_AND_BYTE_CHECKS_PASS", 0
        report.update(status=status, exit_code=code)
        report["interpretation"] = (
            "All reads leave data and manifest untouched. Semantic checks cover schema, "
            "episode ordering, parameter grid, frame counts, fixed official split, finite "
            "states and initial-state bounds. If byte hashes differ, these checks do not "
            "prove trajectory or rendered-image equality to the historical author file. "
            "Image payloads are not decompressed by this structural audit.")
        return report, code

    try:
        m = json.loads(Path(manifest).read_text())
        report["manifest_sha256"] = digest(manifest)
        check("manifest_is_pinned_author_file", report["manifest_sha256"] == MANIFEST_SHA256)
        report["dataset_bytes"] = Path(data).stat().st_size
        report["dataset_sha256"] = digest(data)
        report["manifest_dataset_sha256"] = m.get("dataset_sha256")
        check("manifest_historical_sha_is_pinned", m.get("dataset_sha256") == DATA_SHA256)
        report["historical_byte_match"] = report["dataset_sha256"] == m.get("dataset_sha256")
        expected_recipe = dict(split_method="episode_srnn", train_frames=80,
                               num_steps=16, clip_stride=1, seed=42,
                               n_total_episodes=1200, n_train_episodes=210,
                               n_val_episodes=45, n_test_episodes=45)
        check("manifest_recipe", all(m.get(k) == v for k, v in expected_recipe.items()), expected_recipe)

        with h5py.File(data, "r") as f:
            expected_shapes = {"ep_len": (1200,), "ep_offset": (1200,),
                               "pixels": (99900, 64, 64, 3), "state": (99900, 2),
                               "length": (1200,), "dt": (1200,), "dt_mult": (1200,),
                               "is_train_dt": (1200,), "dt_frame": (99900, 1)}
            check("required_datasets", all(k in f and isinstance(f[k], h5py.Dataset)
                                           for k in expected_shapes), sorted(expected_shapes))
            if not report["checks"]["required_datasets"]["passed"]:
                return close()
            report["dataset_schema"] = {k: dict(shape=list(f[k].shape), dtype=str(f[k].dtype))
                                        for k in expected_shapes}
            check("dataset_shapes", all(f[k].shape == v for k, v in expected_shapes.items()))
            expected_types = {"ep_len": "int64", "ep_offset": "int64", "pixels": "uint8",
                              "state": "float32", "length": "float32", "dt": "float32",
                              "dt_mult": "int64", "is_train_dt": "bool", "dt_frame": "float32"}
            check("dataset_dtypes", all(f[k].dtype == np.dtype(v) for k, v in expected_types.items()))
            if not report["checks"]["dataset_shapes"]["passed"]:
                return close()

            lengths, rates = f["length"][:], f["dt_mult"][:]
            ep_len, offsets = f["ep_len"][:], f["ep_offset"][:]
            dt, dt_frames = f["dt"][:], f["dt_frame"][:, 0]
            expected_lengths = np.repeat(np.asarray([.7, 1., 1.4], dtype=np.float32), 400)
            expected_rates = np.tile(np.repeat(np.arange(1, 5), 100), 3)
            expected_ep_len = np.array([round(8 / (.05 * int(k))) for k in expected_rates], dtype=np.int64)
            expected_offsets = np.r_[0, np.cumsum(expected_ep_len[:-1])]
            check("parameter_grid_and_episode_order", np.array_equal(lengths, expected_lengths)
                  and np.array_equal(rates, expected_rates), "3 rod lengths x 4 dt rates x 100 episodes; rod-major, then dt-major")
            check("episode_lengths_and_offsets", np.array_equal(ep_len, expected_ep_len)
                  and np.array_equal(offsets, expected_offsets), "160/80/53/40 frames per dt cell; 99,900 total")
            check("sampling_intervals", np.allclose(dt, .05 * rates, rtol=1e-7, atol=1e-9)
                  and np.array_equal(dt_frames, np.repeat(dt, ep_len)))
            check("generation_rate_flags", np.array_equal(f["is_train_dt"][:], np.isin(rates, [2, 3, 4])))

            expected_attrs = dict(base_dt=.05, substep=.0125, window=8.,
                                  lengths=[.7, 1., 1.4], train_dt_mults=[2, 3, 4],
                                  test_dt_mults=[1], omega_scale=2 / np.sqrt(9.81),
                                  seed=42, rod_to_scale=True)
            attrs_ok = True
            report["generator_attributes"] = {}
            for key, wanted in expected_attrs.items():
                if key not in f.attrs:
                    attrs_ok = False
                    report["generator_attributes"][key] = None
                    continue
                value = np.asarray(f.attrs[key])
                report["generator_attributes"][key] = value.tolist()
                attrs_ok &= value.shape == np.asarray(wanted).shape and bool(np.allclose(value, wanted, rtol=1e-7, atol=1e-9))
            check("generator_attributes", attrs_ok)

            stats = m.get("dataset_stats", {})
            check("manifest_episode_metadata_matches_file",
                  np.array_equal(stats.get("ep_lengths"), ep_len)
                  and np.array_equal(stats.get("ep_offsets"), offsets)
                  and np.array_equal(stats.get("ep_dt_mult"), rates)
                  and np.allclose(stats.get("ep_cells"), np.c_[lengths, rates], rtol=0, atol=0))
            check("manifest_selects_only_dt2", stats.get("restrict_dt_mults") == [2]
                  and stats.get("held_out_dt") == [1])

            split_arrays = {name: np.asarray(m.get(name + "_episode_indices", []), dtype=np.int64)
                            for name in ["train", "val", "test"]}
            expected_counts = dict(train=210, val=45, test=45)
            valid_ids = all(len(ids) == expected_counts[name] and len(np.unique(ids)) == len(ids)
                            and np.all((ids >= 0) & (ids < 1200)) for name, ids in split_arrays.items())
            check("split_counts_unique_valid_ids", valid_ids, expected_counts)
            if valid_ids:
                combined = np.concatenate(list(split_arrays.values()))
                check("splits_disjoint_cover_dt2", len(np.unique(combined)) == 300
                      and np.array_equal(np.sort(combined), np.flatnonzero(rates == 2)))
                cells = {}
                cell_ok = True
                for name, ids in split_arrays.items():
                    cells[name] = [int(np.count_nonzero(lengths[ids] == v)) for v in np.asarray([.7, 1., 1.4], dtype=np.float32)]
                    cell_ok &= cells[name] == [expected_counts[name] // 3] * 3
                check("split_stratification_by_length", cell_ok, cells)

            states = f["state"][:]
            initial = states[offsets]
            check("states_finite", np.isfinite(states).all())
            check("initial_state_bounds", np.all(np.abs(initial[:, 0]) <= np.pi + 1e-6)
                  and np.all(np.abs(initial[:, 1]) <= 2 / np.sqrt(lengths) + 1e-6))
            report["expected_training_counts"] = {
                "a0m": dict(raw_window=14, train_windows=14070, val_windows=3015,
                             batch_size=32, drop_last=True, updates_per_epoch=439, epochs=100, total_updates=43900),
                "b0m": dict(raw_window=11, train_windows=14700, val_windows=3150,
                             batch_size=32, drop_last=True, updates_per_epoch=459, epochs=100, total_updates=45900),
            }
    except Exception as exc:
        check("audit_completed", False, f"{type(exc).__name__}: {exc}")
    return close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new report file; inputs are always read-only")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite report {args.output}")
    report, code = audit(args.data, args.manifest)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    sys.exit(code)


if __name__ == "__main__":
    main()
