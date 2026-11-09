#!/usr/bin/env python3
"""Explicit solver-only migration, preserving optimizer/RNG and invalidating old selection.

This does not bypass normal resume checks: it validates the old source snapshot,
restricts changes, and creates a new checkpoint with an auditable migration record.
Default mode validates and prints the plan; --write saves into a fresh directory.
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from dpbench.dissipative_model import DissipativeWorldModel, resolve_model_config
from dpbench.train_dissipative import (atomic_json, atomic_torch_save, config_digest,
    preflight, read_config, runtime_signature, source_hashes, validate_checkpoint)

SOLVER_KEYS = {'solver','newton_iterations','line_search_max_backtracks',
               'line_search_shrink','line_search_c1','max_bisections'}
SOURCE_ALLOWLIST = {'dpbench/dissipative_core.py','dpbench/dissipative_model.py',
                    'dpbench/train_dissipative.py'}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def check_config_change(saved, new_config):
    old = saved['train_config']
    if old['training'] != new_config['training']:
        raise ValueError('Solver migration cannot change training schedule or data/optimizer settings')
    before, after = resolve_model_config(saved['config']), resolve_model_config(new_config['model'])
    stripped=[]
    for cfg in (before, after):
        c=copy.deepcopy(cfg)
        c['core']={k:v for k,v in c['core'].items() if k not in SOLVER_KEYS}
        stripped.append(c)
    if stripped[0] != stripped[1]:
        raise ValueError('Only declared solver controls may change; architecture, physics and tolerances must match')
    if after['core']['solver'] != 'backtracking' or not after['core']['strict']:
        raise ValueError('Migration target must use strict backtracking')
    changes={k:{'before':before['core'].get(k),'after':after['core'].get(k)}
             for k in SOLVER_KEYS if before['core'].get(k)!=after['core'].get(k)}
    return after, changes

def migrate(checkpoint, config_path, data, original_source_root, output, device='cuda', write=False):
    config=read_config(config_path)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    old_hashes={name:sha256(Path(original_source_root)/name) for name in saved['source_hashes']}
    if old_hashes != saved['source_hashes']:
        raise ValueError('Old source snapshot does not match checkpoint')
    report=preflight(data,config)
    validate_checkpoint(saved,saved['train_config'],report['dataset_sha256'],old_hashes,
                        torch.device(device),runtime_signature(device))
    resolved, changes=check_config_change(saved,config)
    if saved['step'] % config['training']['val_every']:
        raise ValueError('Migration must start at a validation checkpoint for mandatory revalidation')
    current=source_hashes()
    source_changes={n:{'before':v,'after':current.get(n)} for n,v in old_hashes.items() if current.get(n)!=v}
    if set(source_changes)-SOURCE_ALLOWLIST:
        raise ValueError('Unapproved non-solver source change')
    model=DissipativeWorldModel(config['model'])
    model.load_state_dict(saved['state_dict'],strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite checkpoint parameters')
    provenance={'kind':'explicit_solver_only_migration','original_checkpoint':str(Path(checkpoint).resolve()),
        'original_checkpoint_sha256':sha256(checkpoint),'original_source_root':str(Path(original_source_root).resolve()),
        'original_source_hashes':old_hashes,'new_source_hashes':current,'source_changes':source_changes,
        'solver_changes':changes,'original_config_sha256':saved['config_sha256'],
        'new_config_sha256':config_digest(config),'step':saved['step'],
        'preserved':['state_dict','optimizer','rng','sampler_next_step','validation_indices','dataset_sha256'],
        'reset':['validation','best_score','best_step','best_checkpoint_path'],
        'scope':'Changed numerical solver; same architecture/data/schedule. This is not an unchanged-experiment resume.',
        'migration_script_sha256':sha256(__file__)}
    if not write: return provenance
    output=Path(output)
    if output.exists(): raise FileExistsError('Choose a new migration directory')
    migrated=copy.deepcopy(saved)
    migrated.update(config=resolved,train_config=config,config_sha256=config_digest(config),
        source_hashes=current,best_score=None,best_step=None,best_checkpoint_path=None,
        validation=None,validation_status='not_run',solver_migration=provenance)
    validate_checkpoint(migrated,config,report['dataset_sha256'],current,torch.device(device),runtime_signature(device))
    output.mkdir(parents=True)
    atomic_torch_save(output/'last.pt',migrated)
    atomic_json(output/'migration.json',provenance)
    return provenance

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','config','data','original-source-root','output'): p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--write',action='store_true')
    a=p.parse_args()
    print(json.dumps(migrate(a.checkpoint,a.config,a.data,a.original_source_root,a.output,a.device,a.write),indent=2))
