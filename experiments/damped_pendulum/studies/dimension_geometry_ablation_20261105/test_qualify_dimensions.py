"""Qualifier provenance, CLI and sampler checks; no CUDA, fitting or optimizer."""
import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = importlib.util.spec_from_file_location('dimension_qualifier', HERE / 'qualify_dimensions.py')
qualifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualifier)
OLD = ROOT / 'studies/lehami_loss_pair_launch_20261105T064216Z'


@pytest.fixture
def records():
    run = json.loads((OLD / 'progress_20261105T080925Z/runs/lehami_4000_10000/run.json').read_text())
    qualification = json.loads((OLD / 'received/lehami_cuda_qualification.json').read_text())
    config = json.loads((HERE / 'configs/lehami_d8_s0.json').read_text())
    checked = dict(resolved_model_config=qualifier.resolve_model_config(config['model']),
                   dataset_sha256=run['dataset_sha256'])
    return config, checked, run, qualification


@pytest.mark.parametrize('dimension', [2, 8, 32])
def test_actual_d2_continuation_provenance_accepts_only_dimension_change(records, dimension):
    config, checked, run, qualification = records
    config['model']['q_dim'] = dimension
    checked['resolved_model_config'] = qualifier.resolve_model_config(config['model'])
    reference_config, reference_model = qualifier.verify_reference_contract(config, checked, run, qualification)
    assert reference_config == run['train_config'] and reference_model['q_dim'] == 2
    assert run['stop_after_step'] == 10000


@pytest.mark.parametrize('problem', ['training', 'model', 'data', 'source', 'qualification', 'run_hash'])
def test_reference_mismatch_fails_before_any_sample_read(records, monkeypatch, problem):
    config, checked, run, qualification = records
    if problem == 'training':
        config['training']['seed'] += 1
    elif problem == 'model':
        config['model']['momentum_reference_time'] = .2
    elif problem == 'data':
        checked['dataset_sha256'] = 'different'
    elif problem == 'source':
        monkeypatch.setattr(qualifier, 'source_hashes', lambda: {'different.py': 'changed'})
    elif problem == 'qualification':
        qualification['config_sha256'] = 'different'
    else:
        run['config_sha256'] = 'different'
    with pytest.raises(ValueError):
        qualifier.verify_reference_contract(config, checked, run, qualification)


@pytest.fixture
def paired_metadata(records, monkeypatch):
    config, checked, run, qualification = records
    created = []

    class MetadataOnlyDataset:
        def __init__(self, path, split, **kwargs):
            self.split, self.kwargs, self.closed = split, kwargs, False
            self.index = [(tuple(sorted(kwargs.items())), index) for index in range(10)]
            self.strata = {'test': self.index}
            created.append(self)

        def __len__(self):
            return len(self.index)

        def close(self):
            self.closed = True

        def batch(self, *args, **kwargs):
            pytest.fail('Pairing metadata must never load pixel samples')

    class Sampler:
        def __init__(self, dataset, batch_size, seed):
            assert dataset.split == 'train' and batch_size == 32 and seed == 0

        def indices(self, step):
            return [index + step for index in qualification['fixed_train_window_indices']]

    def anchors(dataset, count, seed):
        assert dataset.split == 'val' and count == 288 and seed == 1009
        return list(run['validation_indices'])

    monkeypatch.setattr(qualifier, 'DissipativeWindowDataset', MetadataOnlyDataset)
    monkeypatch.setattr(qualifier, 'StepBatchSampler', Sampler)
    monkeypatch.setattr(qualifier, 'fixed_validation_indices', anchors)
    return records, created


def test_pairing_checks_archived_first_batch_boundary_steps_and_validation(paired_metadata):
    records, created = paired_metadata
    config, checked, run, qualification = records
    report = qualifier.verify_pairing('unused.h5', config, checked, run, qualification)
    assert report['validation_indices'] == run['validation_indices']
    assert report['paired_train_window_indices']['0'] == qualification['fixed_train_window_indices']
    assert tuple(map(int, report['paired_train_window_indices'])) == qualifier.SAMPLER_STEPS
    assert report['validation_indices_match'] and not report['samples_read']
    assert all(dataset.closed for dataset in created)
    assert all(dataset.kwargs['target_tail_frames'] == 2 for dataset in created)


@pytest.mark.parametrize('mismatch', ['first_batch', 'validation'])
def test_sampler_or_validation_mismatch_is_rejected_and_datasets_close(paired_metadata, monkeypatch, mismatch):
    records, created = paired_metadata
    config, checked, run, qualification = records
    if mismatch == 'first_batch':
        original = qualifier.StepBatchSampler

        class ChangedSampler(original):
            def indices(self, step):
                return [value + 1 for value in super().indices(step)]

        monkeypatch.setattr(qualifier, 'StepBatchSampler', ChangedSampler)
    else:
        monkeypatch.setattr(qualifier, 'fixed_validation_indices', lambda *args: [0])
    with pytest.raises(ValueError, match='batch|Validation'):
        qualifier.verify_pairing('unused.h5', config, checked, run, qualification)
    assert all(dataset.closed for dataset in created)


def test_cli_forwards_reference_paths_without_cuda_or_optimizer(monkeypatch, tmp_path):
    sentinel = object()
    calls = []

    def intercept(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(qualifier, 'execute', intercept)
    values = ['c.json', 'd.h5', 'o.json', 'run.json', 'qualification.json']
    args = []
    for flag, value in zip(['--config', '--data', '--output', '--reference-run', '--reference-qualification'], values):
        args.extend([flag, value])
    assert qualifier.main(args) is sentinel
    assert calls == [tuple(map(Path, values))]
    assert not list(tmp_path.iterdir())


def test_horizon_schedule_has_k16_backward_and_no_optimizer_call():
    assert qualifier.HORIZONS == (4, 8, 16, 32)
    tree = ast.parse((HERE / 'qualify_dimensions.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_checks')
    backward = next(node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == 'backward' for target in node.targets))
    assert [eval(compile(ast.Expression(backward), '<horizon>', 'eval'), {'horizon': horizon})
            for horizon in qualifier.HORIZONS] == [True, True, True, False]
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {'Adam', 'AdamW', 'SGD', 'step'} for node in ast.walk(tree))
