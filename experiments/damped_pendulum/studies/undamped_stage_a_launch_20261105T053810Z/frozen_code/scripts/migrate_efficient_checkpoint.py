#!/usr/bin/env python3
"""Explicit DG/backtracking-2 to predicted-D/Picard-3 experiment migration.

This changes the discrete dynamics and backward definition. It is not an exact
continuation of the old experiment. Preserve model/Adam/RNG/sample position,
discard old model-selection scores, and require validation before the next update.
Default mode only verifies and reports; --write creates a fresh checkpoint copy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpbench.dissipative_model import DissipativeWorldModel, resolve_model_config
from dpbench.train_dissipative import (
    atomic_json, atomic_torch_save, config_digest, preflight, read_config,
    runtime_signature, source_hashes, validate_checkpoint,
)

ALGORITHM_KEYS = {
    'integrator', 'solver', 'substeps', 'picard_iterations', 'picard_check_every',
    'fd_relative_step', 'implicit_rcond_min', 'adjoint_rtol',
}
BASE_SOURCES = {
    'dpbench/train_dissipative.py', 'dpbench/dissipative_model.py',
    'dpbench/dissipative_core.py', 'dpbench/dissipative_data.py',
    'third_party/le-wm/module.py',
}
CHANGED_SOURCES = {'dpbench/train_dissipative.py', 'dpbench/dissipative_model.py'}
ADDED_SOURCES = {
    'dpbench/efficient_dissipative_core.py', 'dpbench/automatic_discrete_gradient.py',
}


def sha256(filename):
    digest = hashlib.sha256()
    with Path(filename).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_config_change(saved, new_config):
    old = saved['train_config']
    if old['training'] != new_config['training']:
        raise ValueError('Migration cannot change training schedule, data or optimizer settings')
    before = resolve_model_config(old['model'])
    if before != saved['config']:
        raise ValueError('Original resolved model differs from its saved train_config')
    after = resolve_model_config(new_config['model'])
    if (before['core']['integrator'], before['core']['solver'], before['core']['substeps']) != (
            'discrete_gradient', 'backtracking', 2):
        raise ValueError('Source must use strict discrete_gradient/backtracking with 2 substeps')
    if (after['core']['integrator'], after['core']['solver'], after['core']['substeps']) != (
            'frozen_dg', 'picard', 3):
        raise ValueError('Target must use strict frozen_dg/picard with 3 substeps')
    # resolve_model_config rejects non-strict models. Keep all legacy solver
    # controls and root/chain tolerances, not only the learned architecture.
    stripped = []
    for configuration in (before, after):
        stripped_config = copy.deepcopy(configuration)
        stripped_config['core'] = {key: value for key, value in stripped_config['core'].items()
                                   if key not in ALGORITHM_KEYS}
        stripped.append(stripped_config)
    if stripped[0] != stripped[1]:
        raise ValueError('Architecture, losses, physics, tolerances and legacy solver controls must match')
    changes = {key: {'before': before['core'].get(key), 'after': after['core'].get(key)}
               for key in sorted(ALGORITHM_KEYS)
               if before['core'].get(key) != after['core'].get(key)}
    return after, changes


def verify_source_change(saved_hashes, original_source_root, current):
    original = Path(original_source_root).resolve()
    # Validate paths before any reads, including rejected extra manifest entries.
    for name in saved_hashes:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not (original / relative).resolve().is_relative_to(original):
            raise ValueError('Original source manifest contains an unsafe relative path')
    if not BASE_SOURCES <= set(saved_hashes) or set(saved_hashes) - BASE_SOURCES - ADDED_SOURCES:
        raise ValueError('Original source manifest has missing or unexpected entries')
    if set(current) != BASE_SOURCES | ADDED_SOURCES:
        raise ValueError('Current source manifest has missing or unexpected entries')
    old_hashes = {name: sha256(original / name) for name in saved_hashes}
    if old_hashes != saved_hashes:
        raise ValueError('Original source snapshot does not match checkpoint')
    changes = {}
    for name in sorted(set(old_hashes) | set(current)):
        before, after = old_hashes.get(name), current.get(name)
        if before == after:
            continue
        if before is None:
            allowed = name in ADDED_SOURCES and after is not None
        else:
            allowed = name in CHANGED_SOURCES and after is not None
        if not allowed:
            raise ValueError(f'Unapproved source change: {name}')
        changes[name] = {'before': before, 'after': after,
                         'kind': 'added' if before is None else 'modified'}
    return old_hashes, changes


def _validate_states(saved, model, training):
    model.load_state_dict(saved['state_dict'], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Nonfinite checkpoint model state')
    if not saved['optimizer']['state']:
        raise ValueError('Migration requires nonempty inherited Adam state')
    optimizer = torch.optim.AdamW([
        {'params': list(model.encoder_parameters()), 'lr': training['encoder_lr'],
         'base_lr': training['encoder_lr'], 'name': 'encoder'},
        {'params': list(model.dynamics_parameters()), 'lr': training['dynamics_lr'],
         'base_lr': training['dynamics_lr'], 'name': 'dynamics'},
    ], weight_decay=training['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite inherited Adam state')


def migrate(checkpoint, config_path, data, original_source_root, output,
            device='cuda', write=False):
    original_hash = sha256(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = read_config(config_path)
    resolved, changes = check_config_change(saved, config)
    current = source_hashes()
    old_hashes, source_changes = verify_source_change(saved['source_hashes'], original_source_root, current)
    report = preflight(data, config)
    runtime = runtime_signature(device)
    validate_checkpoint(saved, saved['train_config'], report['dataset_sha256'], old_hashes,
                        torch.device(device), runtime)
    if not 0 < saved['step'] < config['training']['steps']:
        raise ValueError('Migration requires a nonzero checkpoint with remaining updates')
    if saved['step'] % config['training']['val_every'] or saved.get('validation_status') != 'completed':
        raise ValueError('Migration requires a completed scheduled validation checkpoint')
    model = DissipativeWorldModel(config['model'])
    _validate_states(saved, model, config['training'])
    if sha256(checkpoint) != original_hash:
        raise ValueError('Original checkpoint changed while migration was being verified')
    provenance = {
        'kind': 'explicit_integrator_and_backward_migration',
        'original_checkpoint': str(Path(checkpoint).resolve()),
        'original_checkpoint_sha256': original_hash,
        'original_source_root': str(Path(original_source_root).resolve()),
        'original_source_hashes': old_hashes, 'new_source_hashes': current,
        'source_changes': source_changes, 'algorithm_changes': changes,
        'original_config_sha256': saved['config_sha256'],
        'new_config_sha256': config_digest(config), 'step': saved['step'],
        'preserved': ['state_dict', 'optimizer', 'rng', 'step', 'sampler_next_step',
                      'validation_indices', 'dataset_sha256'],
        'reset': ['validation', 'validation_status', 'best_score', 'best_step', 'best_checkpoint_path'],
        'previous_migration': saved.get('solver_migration'),
        'scope': 'New discrete map and implicit-root backward; unchanged architecture/data/schedule. '
                 'This is a new experiment initialized from the old weights and Adam state, '
                 'not an unchanged-experiment resume. Revalidate before the first update.',
        'migration_script_sha256': sha256(__file__),
    }
    if not write:
        return provenance
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new migration directory')
    migrated = copy.deepcopy(saved)
    migrated.update(config=resolved, train_config=config, config_sha256=config_digest(config),
                    source_hashes=current, best_score=None, best_step=None, best_checkpoint_path=None,
                    validation=None, validation_status='not_run', solver_migration=provenance)
    validate_checkpoint(migrated, config, report['dataset_sha256'], current, torch.device(device), runtime)
    output.mkdir(parents=True)
    atomic_torch_save(output / 'last.pt', migrated)
    atomic_json(output / 'migration.json', provenance)
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('checkpoint', 'config', 'data', 'original-source-root', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    print(json.dumps(migrate(args.checkpoint, args.config, args.data, args.original_source_root,
                             args.output, args.device, args.write), indent=2, allow_nan=False))
