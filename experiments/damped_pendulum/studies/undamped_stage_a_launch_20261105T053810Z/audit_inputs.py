"""Audit generated stage-A provenance and split metadata, without training."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from dpbench.passive_family import sample_groups
from dpbench.train_dissipative import preflight, read_config, source_hashes


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    code = args.run_root / 'code'
    data_dir = args.run_root / 'data/undamped_v1'
    path = data_dir / 'undamped.h5'
    data_config = json.loads((code / 'configs/passive_family_smooth_v1.json').read_text())
    config = read_config(code / 'configs/dissipative_undamped_stage_a_v1.json')
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    if manifest['config'] != data_config or manifest['selected_damping'] != [0.0]:
        raise ValueError('Data manifest configuration or damping scope mismatch')
    digest = sha256(path)
    if len(manifest['datasets']) != 1 or manifest['datasets'][0]['sha256'] != digest:
        raise ValueError('Data file differs from generation manifest')
    for relative, expected in manifest['source_sha256'].items():
        if sha256(code / relative) != expected:
            raise ValueError('Generator source changed: ' + relative)
    report = preflight(path, config)
    groups = sample_groups(data_config)
    counts, split_groups = {}, {}
    with h5py.File(path, 'r') as handle:
        if handle.attrs['training_scope'] != 'undamped_stage_a' or handle.attrs['renderer'] != 'smooth_v1':
            raise ValueError('Incorrect data scope or renderer')
        if bool(handle.attrs['requires_environment_context']):
            raise ValueError('Stage A must not require environment context')
        if not np.isfinite(handle['damping'][:]).all() or np.any(handle['damping'][:] != 0):
            raise ValueError('Stage A data contains nonzero/nonfinite damping')
        for start in range(0, len(handle['action']), 8192):
            if np.any(handle['action'][start:start + 8192] != 0):
                raise ValueError('Stage A must be passive with zero torque')
        for split, expected_count in data_config['groups'].items():
            episodes = handle['splits/' + split][:]
            counts[split] = len(episodes)
            if len(episodes) != expected_count:
                raise ValueError('Wrong episode count for ' + split)
            ids = handle['group_id'][episodes]
            split_groups[split] = set(map(int, ids))
            expected_frames = data_config['test_frames' if split == 'test' else 'train_frames']
            if np.any(handle['ep_len'][episodes] != expected_frames):
                raise ValueError('Frame count changed for ' + split)
            expected_ids = groups[split]['group_id']
            if set(ids.tolist()) != set(expected_ids.tolist()):
                raise ValueError('Initial group identities changed for ' + split)
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if split_groups[a] & split_groups[b]:
                raise ValueError('Cross-split group leakage')
    print(json.dumps(dict(passed=True, dataset_sha256=digest, groups=counts,
                         manifest_sha256=sha256(data_dir / 'manifest.json'),
                         preflight=report, training_source_hashes=source_hashes(),
                         optimizer_created=False, optimizer_updates=0,
                         test_trajectories_evaluated=False), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
