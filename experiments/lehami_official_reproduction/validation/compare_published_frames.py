"""Compare a candidate multi-rod HDF5 with the author's published raw frame samples.

Read-only: HDF5 is opened with mode 'r', NPZ with allow_pickle=False; JSON goes to
stdout. Exit 0 means these published samples match exactly, never full-dataset
equivalence. Exit 2 means a mismatch, missing input, or incomplete comparison.

Source mapping: upstream/LeHami/lehami/experiments/src/export_paper_data.py:340-376
and make_datasets_strip_pendula.py:40-65. No author modules are imported or run.
The three rod snapshots lack saved episode/frame indices; their locations are
reconstructed from the candidate using the exact author selection procedure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401 -- register optional HDF5 compression filters
except ImportError:
    HDF5PLUGIN_AVAILABLE = False
else:
    HDF5PLUGIN_AVAILABLE = True


REFERENCE_HASHES = {
    "strip_frames.npz": "a296a25741aefeb149cd4cd3c36c0b26071a16a92dd7fad9bd95058e36f47f0f",
    "rod_length_frames.npz": "f16fb86854dae77b25ceb7de22c117ded0f66eb28edeb386e188f5ec0eb085d4",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_comparison(actual, expected):
    same_shape = actual.shape == expected.shape
    result = {
        "exact_match": bool(same_shape and actual.dtype == expected.dtype
                            and np.array_equal(actual, expected)),
        "candidate_shape": list(actual.shape),
        "reference_shape": list(expected.shape),
        "candidate_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
        "candidate_rgb_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "reference_rgb_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
    }
    if same_shape:
        difference = np.abs(actual.astype(np.int64) - expected.astype(np.int64))
        result.update(different_channel_values=int(np.count_nonzero(difference)),
                      max_abs_channel_difference=int(difference.max(initial=0)))
    return result


def pick_conservative(f, stride, n_col):
    """Literal author search order, including its break after an eligible e > 400."""
    need = stride * n_col
    offsets, lengths, dtm = f["ep_offset"][:], f["ep_len"][:], f["dt_mult"][:]
    best, best_span, best_err = None, None, np.inf
    for episode in range(len(lengths)):
        offset, length = int(offsets[episode]), int(lengths[episode])
        if length < need or int(dtm[episode]) != 2:
            continue
        theta = f["state"][offset:offset + need, 0]
        span = float(np.ptp(theta))
        error = abs(span - 3.2)
        if error < best_err:
            best, best_span, best_err = episode, span, error
        if episode > 400:
            break
    if best is None:
        raise ValueError("No conservative episode meets the author strip selection rules")
    return best, best_span


def compare(h5_path, reference_dir):
    report = {
        "schema_version": 1,
        "hdf5": str(h5_path.resolve()),
        "reference_dir": str(reference_dir.resolve()),
        "hdf5plugin_available": HDF5PLUGIN_AVAILABLE,
        "all_published_samples_match": False,
        "full_dataset_equivalence_established": False,
        "interpretation": (
            "Only 8 strip frames and 3 rod-length frames are published references. "
            "The strip has a stored episode index; rod indices are reconstructed "
            "from candidate states with the author selection rule. Exact sample "
            "matches do not establish equality of unobserved trajectories, images, "
            "metadata, or historical HDF5 bytes. This script never modifies data "
            "or manifests and does not explain a historical file hash mismatch."
        ),
        "checks": {},
    }

    def check(name, passed):
        report["checks"][name] = bool(passed)

    try:
        report["reference_files"] = {
            name: {"sha256": file_sha256(reference_dir / name), "expected_sha256": expected}
            for name, expected in REFERENCE_HASHES.items()
        }
        check("pinned_reference_files", all(
            item["sha256"] == item["expected_sha256"] for item in report["reference_files"].values()))
        with np.load(reference_dir / "strip_frames.npz", allow_pickle=False) as z:
            strip_frames = z["frames_conservative"]
            strip_episode, stride = int(z["episode_conservative"]), int(z["stride"])
        with np.load(reference_dir / "rod_length_frames.npz", allow_pickle=False) as z:
            rod_frames, rod_lengths, rod_theta = z["frames"], z["lengths"], z["theta"]
        check("published_reference_schema", (
            strip_frames.shape == (8, 64, 64, 3) and strip_frames.dtype == np.uint8
            and strip_episode == 154 and stride == 3
            and rod_frames.shape == (3, 64, 64, 3) and rod_frames.dtype == np.uint8
            and rod_lengths.shape == rod_theta.shape == (3,)
        ))
        if not all(report["checks"].values()):
            raise ValueError("Reference files do not match the pinned published assets")

        with h5py.File(h5_path, "r") as f:
            required = ("pixels", "state", "ep_len", "ep_offset", "length", "dt_mult")
            if not all(key in f for key in required) or "lengths" not in f.attrs:
                raise ValueError("HDF5 is missing required multi-rod arrays or lengths attribute")
            offsets, ep_lengths = f["ep_offset"][:], f["ep_len"][:]
            lengths, dt_mult = f["length"][:], f["dt_mult"][:]
            report["hdf5_size_bytes"] = h5_path.stat().st_size
            report["total_hdf5_frames"] = int(f["pixels"].shape[0])
            report["total_hdf5_episodes"] = int(len(ep_lengths))
            check("candidate_sample_schema", f["pixels"].shape[1:] == (64, 64, 3)
                  and f["pixels"].dtype == np.uint8 and f["state"].shape[1:] == (2,)
                  and f["state"].dtype == np.float32)
            if strip_episode >= len(ep_lengths) or (len(strip_frames) - 1) * stride >= ep_lengths[strip_episode]:
                raise ValueError("Stored reference strip episode/frame is outside the candidate HDF5")
            offset = int(offsets[strip_episode])
            check("stored_strip_episode_dt_mult", int(dt_mult[strip_episode]) == 2)
            selected_episode, selected_span = pick_conservative(f, stride, len(strip_frames))
            report["strip_selection"] = {
                "reference_episode": strip_episode,
                "candidate_recipe_selected_episode": selected_episode,
                "candidate_recipe_selected_span": selected_span,
                "stride": stride,
                "index_evidence": "episode and stride saved in published NPZ; frames start at t=0 in exporter",
            }
            check("strip_recipe_episode", selected_episode == strip_episode)
            report["strip_frames"] = []
            for index, reference in enumerate(strip_frames):
                local = index * stride
                row = {"reference_row": index, "episode": strip_episode,
                       "local_frame": local, "global_frame": offset + local,
                       "length": float(lengths[strip_episode]),
                       "dt_mult": int(dt_mult[strip_episode])}
                row.update(frame_comparison(f["pixels"][offset + local], reference))
                report["strip_frames"].append(row)
            check("strip_pixels_exact", all(row["exact_match"] for row in report["strip_frames"]))

            candidate_lengths = np.asarray(f.attrs["lengths"])
            check("rod_lengths_attribute", np.array_equal(candidate_lengths, rod_lengths))
            report["rod_frames"] = []
            for index, reference_length in enumerate(rod_lengths):
                if index >= len(candidate_lengths):
                    raise ValueError("Candidate lengths attribute is shorter than the published reference")
                # Keep float32 arithmetic and exact argmin/tie behavior of the author's exporter.
                length = candidate_lengths[index]
                episodes = np.where(np.isclose(lengths, length) & (dt_mult == 2))[0][:100]
                best = (np.inf, None, None, None, None)
                for episode in episodes:
                    start, count = int(offsets[episode]), int(ep_lengths[episode])
                    theta = f["state"][start:start + count, 0]
                    local = int(np.argmin(np.abs(theta - 0.6)))
                    error = abs(theta[local] - 0.6)
                    if error < best[0]:
                        best = (error, int(episode), local, start + local, float(theta[local]))
                if best[1] is None:
                    raise ValueError(f"No candidate episode for reference rod length {reference_length}")
                _, episode, local, global_frame, theta = best
                reference_theta = float(rod_theta[index])
                row = {"reference_row": index, "reference_length": float(reference_length),
                       "candidate_length": float(lengths[episode]), "episode": episode,
                       "local_frame": local, "global_frame": global_frame, "dt_mult": 2,
                       "candidate_theta": theta, "reference_theta": reference_theta,
                       "theta_exact_match": theta == reference_theta,
                       "theta_abs_difference": abs(theta - reference_theta),
                       "searched_episode_indices": episodes.astype(int).tolist(),
                       "index_evidence": "reconstructed on candidate; original indices absent from NPZ"}
                row.update(frame_comparison(f["pixels"][global_frame], rod_frames[index]))
                report["rod_frames"].append(row)
            check("rod_pixels_exact", all(row["exact_match"] for row in report["rod_frames"]))
            check("rod_theta_exact", all(row["theta_exact_match"] for row in report["rod_frames"]))
            check("rod_episode_lengths_exact", all(
                row["candidate_length"] == row["reference_length"] for row in report["rod_frames"]))
            all_rows = report["strip_frames"] + report["rod_frames"]
            report["published_frame_reference_count"] = len(all_rows)
            report["distinct_candidate_frames_compared"] = len({row["global_frame"] for row in all_rows})
            report["all_published_samples_match"] = all(report["checks"].values())
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    code = 0 if report["all_published_samples_match"] else 2
    report.update(status="PUBLISHED_SAMPLES_MATCH" if code == 0 else "SAMPLE_COMPARISON_FAILED_OR_INCOMPLETE",
                  exit_code=code)
    return report, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", "--data", dest="h5", type=Path, required=True,
                        help="Candidate pendulum_multi_v2.h5 (read-only)")
    parser.add_argument("--reference-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "upstream" / "LeHami"
                        / "paper_figures" / "data" / "datasets")
    args = parser.parse_args()
    report, code = compare(args.h5.expanduser(), args.reference_dir.expanduser())
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
