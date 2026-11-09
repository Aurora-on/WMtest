"""Stage-B data wrappers preserving the complete parent passive protocol.

Single-condition data selects one positive seen damping; multi-condition data
retains the parent's seen/interpolation/extrapolation partitions. No model or
optimizer is constructed. Test trajectories are generated and numerically
audited as data, never used here for model training, selection or evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import h5py
import numpy as np

from dpbench.passive_family import (
    ROOT, SPLITS, _parent_config_sha256, _write_file, generate_family,
    plan as family_plan, sample_groups, sha256, validate_config,
)

SINGLE_SCOPE = 'damping_stage_b_single'
MULTI_SCOPE = 'damping_stage_b_multi'
HOLDOUT_SCOPE = 'damping_stage_b_multi_holdout'
PARENT_SOURCES = ('dpbench/passive_family.py', 'dpbench/generate.py')


def _config(value):
    config = copy.deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text())
    validate_config(config)
    if config.get('renderer', 'legacy') != 'smooth_v1':
        raise ValueError('Stage-B pairing requires the smooth_v1 parent renderer')
    if 0.0 not in config['seen_damping']:
        raise ValueError('Parent seen_damping must contain stage-A b=0')
    return config


def _selection(config, damping, multi):
    if not isinstance(multi, bool):
        raise ValueError('multi must be boolean')
    if multi:
        return list(config['seen_damping'])
    if (isinstance(damping, bool) or not isinstance(damping, (int, float))
            or not math.isfinite(damping) or damping <= 0):
        raise ValueError('Single stage B requires exactly one finite positive damping')
    if damping not in config['seen_damping']:
        raise ValueError('Single damping must belong to the parent seen partition, not a held-out condition')
    return [float(damping)]


def plan(config, damping=.2, *, multi=False):
    config = _config(config)
    selected = _selection(config, damping, multi)
    result = family_plan(config, undamped_only=not multi)
    result.update(mode='plan_only', writes_files=False, training_started=False,
                  selection_scope='parent_multi_damping' if multi else 'positive_single_subset',
                  training_scope=MULTI_SCOPE if multi else SINGLE_SCOPE,
                  selected_damping=selected, requires_environment_context=multi,
                  parent_config_sha256=_parent_config_sha256(config),
                  output_files=['seen.h5', 'unseen.h5'] if multi else ['damping.h5'],
                  training_file='seen.h5' if multi else 'damping.h5',
                  holdout_file='unseen.h5' if multi else None,
                  pairing_contract='Reuse all parent initial groups/split identities; select the declared condition set before simulation, with no resampling',
                  holdout_policy='Held-out condition trajectories are generated, never trained on or used for model selection',
                  metadata_boundary='Only RGB/action/dt are model inputs; group/regime metadata serve pairing/sampling only')
    return result


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def _decorate(path, config, selected, scope, context):
    with h5py.File(path, 'r+') as handle:
        handle.attrs.update(training_scope=scope, requires_environment_context=context,
                            parent_config_sha256=_parent_config_sha256(config),
                            parent_config_json=json.dumps(config, sort_keys=True),
                            config_json=json.dumps(config, sort_keys=True),
                            selected_damping_json=json.dumps(selected),
                            selection_scope='parent_multi_damping' if context else 'positive_single_subset')


def generate(config, output, damping=.2, *, multi=False):
    """Generate into a new/empty directory; preserve initial groups and clocks.

    Single generation passes a one-condition internal simulation view to the
    existing writer, after fully validating the unchanged parent protocol.
    Parent config/provenance and original regime IDs are restored explicitly;
    this internal view is never represented as a standalone valid protocol.
    """
    config = _config(config)
    specification = plan(config, damping, multi=multi)
    selected = specification['selected_damping']
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Choose a new or empty stage-B output directory')
    if multi:
        # The underlying generator reserves a nonexistent directory itself.
        # rmdir succeeds only while this caller-provided directory is empty.
        if output.exists():
            output.rmdir()
        manifest = generate_family(config, output)
        _decorate(output / 'seen.h5', config, selected, MULTI_SCOPE, True)
        holdout = list(config['interpolation_damping']) + list(config['extrapolation_damping'])
        _decorate(output / 'unseen.h5', config, holdout, HOLDOUT_SCOPE, True)
    else:
        output.mkdir(parents=True, exist_ok=True)
        groups = sample_groups(config)
        simulation_view = copy.deepcopy(config)
        simulation_view['seen_damping'] = selected
        record, contexts = _write_file(output / 'damping.h5', simulation_view, groups, 'seen')
        _decorate(output / 'damping.h5', config, selected, SINGLE_SCOPE, False)
        parent_id = config['seen_damping'].index(selected[0])
        with h5py.File(output / 'damping.h5', 'r+') as handle:
            handle['damping_id'][:] = parent_id
            handle['regime_id'][:] = parent_id
        development = set(contexts['train']) | set(contexts['val'])
        manifest = dict(status='generated_untrained', training_started=False,
                        config=config, renderer=config['renderer'], datasets=[record],
                        initial_context_overlap=dict(
                            val_matches_train=sorted(ep for key, ids in contexts['val'].items()
                                                     if key in contexts['train'] for ep in ids),
                            test_matches_development=sorted(ep for key, ids in contexts['test'].items()
                                                            if key in development for ep in ids)))
    sources = {name: sha256(ROOT / name) for name in PARENT_SOURCES}
    sources['dpbench/damping_stage_data.py'] = sha256(__file__)
    manifest.update(config=config, plan=specification, training_scope=specification['training_scope'],
                    selected_damping=selected, requires_environment_context=multi,
                    training_file=specification['training_file'], holdout_file=specification['holdout_file'],
                    selection_scope=specification['selection_scope'],
                    parent_config_sha256=_parent_config_sha256(config), source_sha256=sources,
                    split_contract='Parent group IDs, split membership, seeds, initial states and all physical clocks preserved; no resampling',
                    metadata_boundary=specification['metadata_boundary'],
                    test_policy='Test trajectories are prepared and numerically audited only; no model evaluation, fitting or selection',
                    model_training_started=False, optimizer_created=False)
    for record in manifest['datasets']:
        path = output / record['file']
        record.update(sha256=sha256(path), bytes=path.stat().st_size,
                      parent_config_sha256=manifest['parent_config_sha256'],
                      training_scope=HOLDOUT_SCOPE if path.name == 'unseen.h5' else specification['training_scope'])
    _write_json(output / 'generation_config.json', config)
    _write_json(output / 'manifest.json', manifest)
    return manifest


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else value


def validate_pair(data_a, data_b, *, multi=False):
    """Validate paired metadata/clocks, without reading any image or state trajectory.

    Initial states and grouping are metadata. Whole-file hashes bind the bytes,
    but no pixels dataset is read, including test pixels. Multi B repeats each A
    group exactly once for every parent seen damping; unseen.h5 is rejected.
    """
    data_a, data_b = Path(data_a), Path(data_b)
    if not isinstance(multi, bool):
        raise ValueError('multi must be boolean')
    manifests = [json.loads((path.parent / 'manifest.json').read_text()) for path in (data_a, data_b)]
    for source in PARENT_SOURCES:
        if not manifests[0].get('source_sha256', {}).get(source) or (
                manifests[0]['source_sha256'][source] != manifests[1].get('source_sha256', {}).get(source)):
            raise ValueError(f'A/B parent generation source hash mismatch: {source}')
    with h5py.File(data_a, 'r') as a, h5py.File(data_b, 'r') as b:
        if _text(a.attrs.get('training_scope')) != 'undamped_stage_a':
            raise ValueError('A must be the undamped_stage_a dataset')
        expected_scope = MULTI_SCOPE if multi else SINGLE_SCOPE
        if _text(b.attrs.get('training_scope')) != expected_scope:
            raise ValueError(f'B training_scope must be {expected_scope}; holdout files cannot train B')
        if bool(a.attrs.get('requires_environment_context', True)) or bool(b.attrs.get('requires_environment_context', False)) != multi:
            raise ValueError('A/B environment-context contract mismatch')
        parent_a = json.loads(_text(a.attrs['parent_config_json']))
        parent_b = json.loads(_text(b.attrs['parent_config_json']))
        if parent_a != parent_b:
            raise ValueError('A/B parent configs differ')
        parent_hash = _parent_config_sha256(parent_a)
        if any(_text(f.attrs.get('parent_config_sha256')) != parent_hash for f in (a, b)):
            raise ValueError('A/B parent config hash metadata mismatch')
        for manifest in manifests:
            if manifest.get('parent_config_sha256') != parent_hash:
                raise ValueError('Manifest parent config does not match data')
        for key in ('renderer', 'history_required', 'train_frames', 'dt', 'mass', 'length', 'gravity'):
            if _text(a.attrs.get(key)) != _text(b.attrs.get(key)):
                raise ValueError(f'A/B physical or observation setting differs: {key}')
        if _text(b.attrs['renderer']) != 'smooth_v1':
            raise ValueError('Requires smooth_v1 renderer')
        b_values = sorted(np.unique(b['damping'][:]).tolist())
        if not b_values:
            raise ValueError('B contains no damping episodes')
        _selection(parent_a, b_values[0] if len(b_values) == 1 else None, multi)
        selected = list(parent_a['seen_damping']) if multi else b_values
        if b_values != sorted(selected) or (not multi and len(b_values) != 1):
            raise ValueError('B condition partition differs from the selected protocol')
        if np.count_nonzero(a['damping'][:]):
            raise ValueError('A contains nonzero damping')
        if json.loads(_text(b.attrs['selected_damping_json'])) != selected:
            raise ValueError('B selected_damping metadata disagrees with episode conditions')
        comparisons, split_counts = 0, {}
        group_split = {}
        for split in SPLITS:
            a_ids, b_ids = a['splits/' + split][:], b['splits/' + split][:]
            a_by_group = {int(a['group_id'][ep]): int(ep) for ep in a_ids}
            if len(a_by_group) != len(a_ids):
                raise ValueError('A groups repeat within a split')
            for group in a_by_group:
                if group in group_split:
                    raise ValueError('Parent group appears in multiple splits')
                group_split[group] = split
            expected = {(group, coefficient) for group in a_by_group for coefficient in selected}
            observed = set()
            for ep in b_ids:
                ep = int(ep)
                group, coefficient = int(b['group_id'][ep]), float(b['damping'][ep])
                key = (group, coefficient)
                if key not in expected or key in observed:
                    raise ValueError('B groups/conditions duplicate or cross parent split membership')
                observed.add(key)
                ae = a_by_group[group]
                for field in ('initial_state', 'energy_band', 'ep_len'):
                    if not np.array_equal(a[field][ae], b[field][ep]):
                        raise ValueError(f'A/B paired {field} differs for group {group}')
                ao, bo = int(a['ep_offset'][ae]), int(b['ep_offset'][ep])
                n = int(a['ep_len'][ae])
                for field in ('dt_frame', 'valid_transition'):
                    if not np.array_equal(a[field][ao:ao + n], b[field][bo:bo + n]):
                        raise ValueError(f'A/B paired {field} differs for group {group}')
                if np.count_nonzero(a['action'][ao:ao + n]) or np.count_nonzero(b['action'][bo:bo + n]):
                    raise ValueError('Passive paired data must have zero actions')
                comparisons += 1
            if observed != expected:
                raise ValueError(f'Missing paired groups/conditions in B {split}')
            split_counts[split] = dict(a_groups=len(a_ids), b_episodes=len(b_ids))
    actual_hashes = [sha256(path) for path in (data_a, data_b)]
    for path, manifest, digest in zip((data_a, data_b), manifests, actual_hashes):
        entry = next((row for row in manifest.get('datasets', []) if row['file'] == path.name), None)
        if entry is None or entry.get('sha256') != digest:
            raise ValueError('Dataset bytes differ from the generation manifest')
    return dict(status='matched', training_scope=expected_scope, multi=multi,
                selected_damping=selected, paired_episode_comparisons=comparisons,
                split_counts=split_counts, parent_config_sha256=parent_hash,
                data_a_sha256=actual_hashes[0], data_b_sha256=actual_hashes[1],
                parent_source_hashes={name: manifests[0]['source_sha256'][name] for name in PARENT_SOURCES},
                pixels_dataset_read=False, state_trajectories_read=False, test_pixels_read=False,
                scope='Metadata, initial states, clocks, actions and whole-file hashes only; no model training or test performance evaluation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--damping', type=float, default=.2)
    parser.add_argument('--multi', action='store_true')
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args(argv)
    if args.generate and args.output is None:
        parser.error('--generate requires --output')
    result = generate(args.config, args.output, args.damping, multi=args.multi) if args.generate else plan(args.config, args.damping, multi=args.multi)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
