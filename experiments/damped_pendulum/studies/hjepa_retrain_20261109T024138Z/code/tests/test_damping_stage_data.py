"""Tiny data preparation/pairing only; no model fitting or formal generation."""
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from dpbench import damping_stage_data as stage
from dpbench import passive_family as family
from dpbench.dissipative_data import DissipativeWindowDataset, MODEL_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def tiny_config():
    config = json.loads((ROOT / 'configs/passive_family_smoke.json').read_text())
    config.update(history=3, train_frames=6, test_frames=8, renderer='smooth_v1')
    return config


def test_formal_plans_preserve_protocol_and_default_single_scope(tmp_path, capsys):
    path = ROOT / 'configs/passive_family_smooth_v1.json'
    single, multi = stage.plan(path), stage.plan(path, multi=True)
    assert single['selected_damping'] == [.2] and single['training_file'] == 'damping.h5'
    assert single['seen_episodes'] == dict(train=384, val=64, test=128)
    assert single['seen_frames'] == 79936 and single['unseen_frames'] == 0
    assert not single['requires_environment_context']
    assert multi['selected_damping'] == [0, .05, .1, .2, .5, 1]
    assert multi['training_file'] == 'seen.h5' and multi['holdout_file'] == 'unseen.h5'
    assert multi['seen_episodes'] == dict(train=2304, val=384, test=768)
    assert multi['requires_environment_context']
    for result in (single, multi):
        assert result['history_seconds'] == 2 and result['test_rollout_seconds'] == 32
        assert not result['writes_files'] and not result['training_started']
    nonexistent = tmp_path / 'must_not_exist'
    stage.main(['--config', str(path), '--output', str(nonexistent)])
    assert not nonexistent.exists()
    assert json.loads(capsys.readouterr().out)['mode'] == 'plan_only'


@pytest.mark.parametrize('damping', [0, -.2, [.2], None, True, float('nan'), .1])
def test_single_rejects_nonpositive_multiple_or_heldout_condition(damping):
    # Smoke parent holds b=.1 out for interpolation.
    with pytest.raises(ValueError):
        stage.plan(tiny_config(), damping)


@pytest.fixture
def generated(tmp_path):
    config = tiny_config()
    a, single, multi = tmp_path / 'a', tmp_path / 'single', tmp_path / 'multi'
    family.generate_undamped(config, a)
    single.mkdir()  # Public API accepts an existing empty output directory.
    sm = stage.generate(config, single)
    multi.mkdir()
    mm = stage.generate(config, multi, multi=True)
    return config, a / 'undamped.h5', single / 'damping.h5', multi / 'seen.h5', sm, mm


def test_single_is_exact_parent_cohort_and_keeps_original_regime_id(generated):
    config, a_path, single_path, multi_path, manifest, _ = generated
    report = stage.validate_pair(a_path, single_path)
    assert report['status'] == 'matched' and report['selected_damping'] == [.2]
    assert report['paired_episode_comparisons'] == 12
    assert manifest['training_scope'] == stage.SINGLE_SCOPE
    assert not manifest['requires_environment_context'] and manifest['holdout_file'] is None
    with h5py.File(single_path, 'r') as single, h5py.File(multi_path, 'r') as parent:
        assert not single.attrs['requires_environment_context']
        assert np.unique(single['damping'][:]).tolist() == [.2]
        assert np.unique(single['regime_id'][:]).tolist() == [config['seen_damping'].index(.2)]
        assert json.loads(single.attrs['parent_config_json']) == config
        # Compare only train observations/trajectories; no test-pixel scoring.
        parent_ids = {(int(parent['group_id'][ep]), float(parent['damping'][ep])): int(ep)
                      for ep in parent['splits/train'][:]}
        for ep in single['splits/train'][:]:
            pe = parent_ids[(int(single['group_id'][ep]), .2)]
            start, pstart = int(single['ep_offset'][ep]), int(parent['ep_offset'][pe])
            n = int(single['ep_len'][ep])
            for name in ('pixels', 'state', 'energy', 'dissipation', 'dt_frame'):
                np.testing.assert_array_equal(single[name][start:start + n], parent[name][pstart:pstart + n])


def test_multi_pair_allows_repeated_groups_across_conditions_but_never_holdout(generated):
    config, a_path, _, multi_path, _, manifest = generated
    report = stage.validate_pair(a_path, multi_path, multi=True)
    assert report['selected_damping'] == config['seen_damping']
    assert report['paired_episode_comparisons'] == 24
    assert report['split_counts']['train'] == dict(a_groups=4, b_episodes=8)
    assert manifest['training_file'] == 'seen.h5' and manifest['holdout_file'] == 'unseen.h5'
    with h5py.File(multi_path.parent / 'unseen.h5', 'r') as f:
        assert f.attrs['training_scope'] == stage.HOLDOUT_SCOPE
        assert len(f['splits/train']) == len(f['splits/val']) == 0
        assert len(f['splits/test']) > 0 and f.attrs['train_frames'] == 0
    with pytest.raises(ValueError, match='training_scope'):
        stage.validate_pair(a_path, multi_path.parent / 'unseen.h5', multi=True)


def test_pair_validation_never_reads_pixels_or_state_trajectories(generated, monkeypatch):
    _, a_path, single_path, multi_path, _, _ = generated
    original = h5py.Dataset.__getitem__
    def guarded(self, key):
        if self.name in ('/pixels', '/state', '/energy', '/work', '/dissipation'):
            pytest.fail('Pair validation must read metadata/clocks only: ' + self.name)
        return original(self, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    for path, multi in ((single_path, False), (multi_path, True)):
        result = stage.validate_pair(a_path, path, multi=multi)
        assert not result['pixels_dataset_read'] and not result['state_trajectories_read']
        assert not result['test_pixels_read']


@pytest.mark.parametrize('field', ['initial_state', 'dt_frame', 'parent_source', 'group_id'])
def test_pair_rejects_mismatched_initial_clock_source_or_group(generated, field):
    _, a_path, single_path, _, _, _ = generated
    if field == 'parent_source':
        path = single_path.parent / 'manifest.json'
        manifest = json.loads(path.read_text())
        manifest['source_sha256']['dpbench/generate.py'] = 'wrong'
        path.write_text(json.dumps(manifest))
    else:
        with h5py.File(single_path, 'r+') as f:
            if field == 'initial_state':
                f[field][0] += .01
            elif field == 'dt_frame':
                f[field][0] = .2
            else:
                f[field][0] = f[field][-1]  # crosses the parent split boundary
    with pytest.raises(ValueError):
        stage.validate_pair(a_path, single_path)


def test_model_batches_have_only_rgb_action_and_clock_and_outputs_cannot_overwrite(generated):
    config, _, single_path, multi_path, _, _ = generated
    for path in (single_path, multi_path):
        data = DissipativeWindowDataset(path, 'train', history=3, max_horizon=2, train_frames=6)
        try:
            assert set(data.batch([0, 1])) == set(MODEL_FIELDS)
        finally:
            data.close()
    with pytest.raises(FileExistsError):
        stage.generate(config, single_path.parent)
    with pytest.raises(FileExistsError):
        stage.generate(config, multi_path.parent, multi=True)
