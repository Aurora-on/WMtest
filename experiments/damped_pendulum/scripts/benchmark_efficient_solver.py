#!/usr/bin/env python3
"""Bounded old/new solver benchmark from the same saved step-3750 model.

Default is plan-only. Explicit --execute performs at most 42 optimizer updates
(default 24), in isolated model/Adam/RNG resets for each variant and horizon.
No resumable weights are saved, no source checkpoint is modified, and no formal
training, validation selection, physical readout, or test-set evaluation occurs.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpbench.dissipative_data import DissipativeWindowDataset, StepBatchSampler
from dpbench.dissipative_model import DissipativeWorldModel, resolve_model_config
from dpbench.train_dissipative import (atomic_json, config_digest, lr_factor, move_batch,
    preflight, read_config, restore_rng, runtime_signature, scalar_metrics,
    source_hashes, validate_checkpoint)
from scripts.profile_dissipative import _optimizer, _state_hash

HORIZONS = (8, 16, 32)
VARIANTS = ('reference', 'efficient')
CHANGED_CORE_OPTIONS = {'integrator', 'solver', 'picard_iterations', 'picard_check_every',
                       'fd_relative_step', 'implicit_rcond_min', 'adjoint_rtol'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def benchmark_sources():
    return {**source_hashes(),
            'scripts/profile_dissipative.py': sha256(ROOT / 'scripts/profile_dissipative.py'),
            'scripts/benchmark_efficient_solver.py': sha256(__file__)}


def plan(args):
    if not 1 <= args.warmup <= 2 or not 1 <= args.timed <= 5:
        raise ValueError('Bounded benchmark requires 1<=warmup<=2 and 1<=timed<=5')
    horizons = tuple(args.horizons)
    if not horizons or len(set(horizons)) != len(horizons) or set(horizons) - set(HORIZONS):
        raise ValueError('Choose a unique nonempty subset of horizons8,16,32')
    return dict(status='plan_only', optimizer_updates_enabled=False,
                checkpoint=str(args.checkpoint), expected_checkpoint_step=3750,
                variants=list(VARIANTS), horizons=list(horizons),
                warmup_updates_per_case=args.warmup, timed_updates_per_case=args.timed,
                total_optimizer_update_budget=2 * len(horizons) * (args.warmup + args.timed),
                read_only_rollout_batches=2 * len(horizons), batch_size=32, device=args.device,
                output=str(args.output), writes_files=False, saves_training_weights=False,
                formal_training_resumed=False, test_samples_read=False)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def verified_inputs(args, efficient):
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved['step'] != 3750 or saved['validation_status'] != 'completed':
        raise ValueError('This comparison requires the validated step-3750 checkpoint')
    if not saved['optimizer']['state']:
        raise ValueError('A nonempty inherited Adam state is required')
    original = Path(args.original_source_root).resolve()
    old_hashes = {}
    for name in saved['source_hashes']:
        path = (original / name).resolve()
        if not path.is_relative_to(original):
            raise ValueError('Original-source manifest contains an unsafe relative path')
        old_hashes[name] = sha256(path)
    if old_hashes != saved['source_hashes']:
        raise ValueError('Original source snapshot does not match the checkpoint')
    reference = copy.deepcopy(saved['train_config'])
    checked = preflight(args.data, reference)
    # Validate against ORIGINAL sources/config, not the new algorithm's hashes.
    validate_checkpoint(saved, reference, checked['dataset_sha256'], old_hashes,
                        torch.device(args.device), runtime_signature(args.device))
    if reference['training'] != efficient['training']:
        raise ValueError('Data, optimizer and full-budget training settings must match')
    before = resolve_model_config(reference['model'])
    after = resolve_model_config(efficient['model'])
    if before != saved['config']:
        raise ValueError('Reference model resolution changed from its saved configuration')
    if before['core']['integrator'] != 'discrete_gradient' or before['core']['solver'] != 'backtracking':
        raise ValueError('Reference must be the paused robust backtracking solver')
    if after['core']['integrator'] != 'frozen_dg' or after['core']['solver'] != 'picard':
        raise ValueError('Candidate must select frozen_dg/picard')
    allowed_changes = set(CHANGED_CORE_OPTIONS)
    if args.allow_substeps_change:
        if before['core']['substeps'] != 2 or after['core']['substeps'] != 3:
            raise ValueError('Explicit substep comparison requires reference2 versus candidate3')
        allowed_changes.add('substeps')
    stripped = []
    for configuration in (before, after):
        c = copy.deepcopy(configuration)
        c['core'] = {k: v for k, v in c['core'].items() if k not in allowed_changes}
        stripped.append(c)
    if stripped[0] != stripped[1] or reference['training']['batch_size'] != 32:
        raise ValueError('Architecture, loss, tolerances, substeps and global batch32 must remain unchanged')
    # The candidate model dispatcher changes, but the old numerical core and
    # sampler must remain byte-identical for a meaningful reference arm.
    for name in ('dpbench/dissipative_core.py', 'dpbench/dissipative_data.py', 'third_party/le-wm/module.py'):
        if sha256(ROOT / name) != old_hashes[name]:
            raise ValueError(f'Reference numerical/data code changed: {name}')
    changes = {k: {'before': before['core'].get(k), 'after': after['core'].get(k)}
               for k in sorted(allowed_changes) if before['core'].get(k) != after['core'].get(k)}
    return saved, reference, checked, changes


def readonly_rollout(model, batch, horizon, output, tag):
    """Small same-initial-state numerical check; no physical accuracy claim."""
    anchor = model.history - 1
    model.eval()
    with torch.no_grad():
        prediction = model.rollout(batch['pixels'][:, :model.history],
            batch['action'][:, anchor:anchor + horizon], batch['action'][:, :anchor],
            batch['dt_frame'][:, :anchor], batch['dt_frame'][:, anchor:anchor + horizon])
        q, p, diag = prediction['q'], prediction['p'], prediction['diagnostics']
        energy = model.core.energy(q, p)
        defect = energy[:, 1:] - energy[:, :-1] - diag['work'] + diag['dissipation']
        identity_error = defect - diag['residual_work'] - diag['chain_defect']
        scale = 1 + energy[:, 1:].abs() + energy[:, :-1].abs() + diag['dissipation'].abs() + diag['work'].abs()
        if not all(torch.isfinite(x).all() for x in (q, p, energy, *diag.values())):
            raise FloatingPointError('Nonfinite read-only rollout')
        if (diag['solver_residual_ratio'] > 1 + 1e-10).any():
            raise RuntimeError('Read-only residual exceeds original tolerance')
        if (diag['dissipation'] < -1e-12).any() or (diag['work'].abs() > 1e-12).any():
            raise RuntimeError('Invalid passive energy accounting')
        if (identity_error.abs() > 1e-10 * scale).any():
            raise RuntimeError('Read-only rollout energy ledger identity failed')
        arrays = {'q': q.cpu().numpy(), 'p': p.cpu().numpy(), 'energy': energy.cpu().numpy(),
                  **{k: v.cpu().numpy() for k, v in diag.items()}}
        np.savez_compressed(output / f'{tag}_readonly.npz', **arrays)
        result = dict(trajectories=q.shape[0], transitions=q.shape[0] * horizon,
                      max_residual_ratio=float(diag['solver_residual_ratio'].max()),
                      max_split_depth=float(diag['solver_split_depth'].max()),
                      max_ledger_identity_error=float(identity_error.abs().max()),
                      max_numerical_energy_defect=float(defect.abs().max()),
                      raw_arrays=f'{tag}_readonly.npz')
    model.train()
    return result


def execute(args):
    report = plan(args)
    if args.output.exists():
        raise FileExistsError('Choose a new benchmark directory; existing evidence is never overwritten')
    efficient = read_config(args.efficient_config)
    saved, reference, checked, changes = verified_inputs(args, efficient)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    cfg = reference['training']
    torch.set_num_threads(cfg['threads'])
    dataset = DissipativeWindowDataset(args.data, 'train', history=saved['config']['history'],
        max_horizon=32, train_frames=cfg['train_frames'])
    sampler = StepBatchSampler(dataset, cfg['batch_size'], cfg['seed'])
    indices = [sampler.indices(saved['step'] + i) for i in range(args.warmup + args.timed)]
    args.output.mkdir(parents=True)
    initial_sources, checkpoint_hash = benchmark_sources(), sha256(args.checkpoint)
    report.update(status='running', optimizer_updates_enabled=True, writes_files=True,
        kind='non_resumable_solver_benchmark', checkpoint_sha256=checkpoint_hash,
        checkpoint_step=saved['step'], original_source_hashes=saved['source_hashes'],
        current_source_hashes=initial_sources, script_sha256=sha256(__file__),
        dataset_sha256=checked['dataset_sha256'], runtime=runtime_signature(device),
        algorithm_changes=changes, reference_config=reference, efficient_config=efficient,
        efficient_config_sha256=config_digest(efficient),
        efficient_config_file_sha256=sha256(args.efficient_config), sampler_window_indices=indices,
        learning_rate_scope='Original full-20000-step LR at3751 onward; forced horizon; reset Adam/RNG for every case',
        timing_scope='Synchronized whole update includes data read/transfer, loss, scalar checks, backward, gradient clipping and AdamW. Parameter-finiteness/hash checks and read-only rollout are outside the timed update.',
        phase_timing_scope='Wall segments are host observations, not independently synchronized. CUDA-event forward/backward timings are queried after the final synchronization; do not add event timings to wall segments.',
        optimizer_updates_completed=0, measurements={}, numerical_check_only=True,
        prediction_accuracy_evaluated=False)
    atomic_json(args.output / 'manifest.json', report)
    started, current_case, current_phase = time.perf_counter(), None, 'setup'
    try:
        with (args.output / 'history.jsonl').open('x') as log:
            # Alternate which method is first to reduce a simple ordering bias.
            for h_index, horizon in enumerate(args.horizons):
                for variant in (VARIANTS if h_index % 2 == 0 else tuple(reversed(VARIANTS))):
                    current_case, current_phase = f'{variant}_k{horizon}', 'restore'
                    case_config = reference if variant == 'reference' else efficient
                    model = DissipativeWorldModel(case_config['model']).to(device)
                    model.load_state_dict(saved['state_dict'], strict=True)
                    optimizer = _optimizer(model, case_config)
                    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
                    restore_rng(saved['rng'])
                    start_hash = _state_hash(model)
                    current_phase = 'read_only_rollout'
                    batch = move_batch(dataset.batch(indices[0], horizon=horizon), device)
                    read_only = readonly_rollout(model, batch, horizon, args.output, current_case)
                    if _state_hash(model) != start_hash:
                        raise RuntimeError('Read-only rollout changed model state')
                    restore_rng(saved['rng'])
                    case_rows = []
                    for index in range(args.warmup + args.timed):
                        current_phase = 'optimizer_update'
                        synchronize(device)
                        if device.type == 'cuda':
                            torch.cuda.reset_peak_memory_stats(device)
                            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                        else:
                            events = None
                        tick = time.perf_counter()
                        batch = move_batch(dataset.batch(indices[index], horizon=horizon), device)
                        factor = lr_factor(reference, saved['step'] + index + 1)
                        for group in optimizer.param_groups:
                            group['lr'] = group['base_lr'] * factor
                        optimizer.zero_grad(set_to_none=True)
                        data_end = time.perf_counter()
                        if events: events[0].record()
                        losses = model.loss(batch, horizon)
                        if events: events[1].record()
                        forward_end = time.perf_counter()
                        metrics = scalar_metrics(losses)
                        scalar_end = time.perf_counter()
                        if events: events[2].record()
                        losses['loss'].backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
                        optimizer.step()
                        report['optimizer_updates_completed'] += 1
                        if events: events[3].record()
                        synchronize(device)
                        end = time.perf_counter()
                        grad = float(norm)
                        if not math.isfinite(grad) or not all(torch.isfinite(p).all() for p in model.parameters()):
                            raise FloatingPointError('Nonfinite gradient or updated parameter')
                        if metrics['solver_residual_ratio'] > 1 + 1e-10 or metrics['solver_split_depth'] > model.core.max_bisections:
                            raise RuntimeError('Accepted optimizer rollout violates numerical acceptance')
                        row = dict(case=current_case, variant=variant, horizon=horizon, case_update=index + 1,
                            timed=index >= args.warmup, sampler_absolute_step=saved['step'] + index + 1,
                            lr_factor=factor, total_seconds=end - tick, data_setup_wall_seconds=data_end - tick,
                            forward_wall_seconds=forward_end - data_end, scalar_wall_seconds=scalar_end - forward_end,
                            backward_optimizer_wall_seconds=end - scalar_end,
                            forward_cuda_seconds=events[0].elapsed_time(events[1]) / 1000 if events else None,
                            backward_optimizer_cuda_seconds=events[2].elapsed_time(events[3]) / 1000 if events else None,
                            gradient_norm=grad, losses=metrics,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if events else None,
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if events else None)
                        case_rows.append(row)
                        log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                    current_phase = 'case_audit'
                    changed = sum(not torch.equal(p.detach().cpu(), saved['state_dict'][name])
                                  for name, p in model.named_parameters())
                    if changed == 0 or not optimizer.state:
                        raise RuntimeError('Benchmark did not actually update parameters/Adam')
                    timed = [r for r in case_rows if r['timed']]
                    report['measurements'][current_case] = dict(
                        variant=variant, horizon=horizon, timed_updates=len(timed),
                        total_updates=len(case_rows), parameters_changed=changed,
                        initial_state_sha256=start_hash, final_state_sha256=_state_hash(model),
                        optimizer_state_entries=len(optimizer.state), read_only_rollout=read_only,
                        mean_seconds=statistics.mean(r['total_seconds'] for r in timed),
                        median_seconds=statistics.median(r['total_seconds'] for r in timed),
                        minimum_seconds=min(r['total_seconds'] for r in timed),
                        maximum_seconds=max(r['total_seconds'] for r in timed),
                        max_solver_residual_ratio=max(r['losses']['solver_residual_ratio'] for r in case_rows),
                        max_split_depth=max(r['losses']['solver_split_depth'] for r in case_rows),
                        phase_mean_seconds={k: statistics.mean(r[k] for r in timed)
                            for k in ('data_setup_wall_seconds', 'forward_wall_seconds', 'scalar_wall_seconds', 'backward_optimizer_wall_seconds')})
                    atomic_json(args.output / 'manifest.json', report)
                    print(json.dumps({'case': current_case, **report['measurements'][current_case]}, allow_nan=False), flush=True)
                    del model, optimizer, losses, batch, case_rows
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()
        if len({m['initial_state_sha256'] for m in report['measurements'].values()}) != 1:
            raise RuntimeError('Cases did not all start from the same model state')
        if (benchmark_sources() != initial_sources or sha256(args.checkpoint) != checkpoint_hash
                or sha256(args.efficient_config) != report['efficient_config_file_sha256']):
            raise RuntimeError('Source snapshot or source checkpoint changed during benchmark')
        report['speedup_reference_over_efficient'] = {str(h):
            report['measurements'][f'reference_k{h}']['mean_seconds'] /
            report['measurements'][f'efficient_k{h}']['mean_seconds'] for h in args.horizons}
        report.update(status='completed', source_checkpoint_unchanged=True,
                      source_snapshot_unchanged=True, same_initial_weights_all_cases=True)
    except BaseException as error:
        report.update(status='failed', failed_case=current_case, failed_phase=current_phase,
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        dataset.close()
        report['elapsed_seconds'] = time.perf_counter() - started
        atomic_json(args.output / 'manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'data', 'efficient-config', 'original-source-root', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timed', type=int, default=3)
    parser.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    parser.add_argument('--allow-substeps-change', action='store_true',
                        help='Explicitly compare reference2 versus candidate3 substeps')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    try:
        proposal = plan(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.execute:
        print(json.dumps(proposal, indent=2))
    else:
        execute(args)


if __name__ == '__main__':
    main()
