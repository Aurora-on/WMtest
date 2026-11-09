#!/usr/bin/env python3
"""Qualify fresh stage-A CUDA forward/backward paths without any optimizer.

CLI requires an allocated Slurm GPU. It reads one fixed batch of32 training
windows, preserves their full configured history, checks K4/K8 backward and
K32 no-grad prediction, and writes a JSON audit. It neither trains nor saves a
checkpoint. The CPU-callable run_checks helper exists for nontraining tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpbench.dissipative_data import DissipativeWindowDataset, StepBatchSampler
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.train_dissipative import (
    atomic_json, build_optimizer_parameter_groups, config_digest,
    configure_trainable_modules, maximum_horizon, move_batch, preflight,
    read_config, runtime_signature, scalar_metrics, source_hashes,
)

HORIZONS = (4, 8, 32)
SELECTED_MODULES = {'encoder', 'initializer', 'potential'}


def require_cuda_allocation():
    job_id = os.environ.get('SLURM_JOB_ID', '')
    gpu_ids = os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')
    gpu_count = os.environ.get('SLURM_GPUS_ON_NODE')
    gpu_allocation = gpu_ids or gpu_count
    if not job_id.isdigit() or not gpu_allocation or (not gpu_ids and gpu_count == '0'):
        raise RuntimeError('Qualification requires a Slurm GPU allocation, not a login-node process')
    if not torch.cuda.is_available():
        raise RuntimeError('Qualification requires an available CUDA device')
    return torch.device('cuda'), dict(job_id=job_id, allocated_gpus=gpu_allocation,
                                     node=socket.gethostname(),
                                     partition=os.environ.get('SLURM_JOB_PARTITION'),
                                     cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                     device_uuid=str(torch.cuda.get_device_properties(0).uuid))


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _gradient_audit(model):
    modules = dict(encoder=model.encoder, initializer=model.velocity_correction,
                   potential=model.core.potential_net)
    rows = {}
    for name, module in modules.items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError(f'{name}: missing or nonfinite gradients')
        norm = float(torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt())
        if norm <= 0:
            raise RuntimeError(f'{name}: qualification requires a nonzero module gradient')
        rows[name] = dict(norm=norm, finite=True, nonzero=True,
                          tensors_with_gradient=len(gradients))
    missing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise RuntimeError(f'Frozen parameter received a gradient: {name}')
        elif parameter.grad is None:
            # A raw potential's final additive constant does not affect forces.
            if model.config['core']['potential_mode'] != 'raw' or name != 'core.potential_net.0.4.bias':
                raise RuntimeError(f'Selected parameter has no gradient: {name}')
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Nonfinite gradient: {name}')
    return dict(modules=rows, permitted_gauge_parameters_without_gradient=missing,
                frozen_parameters_have_no_gradient=True)


def run_checks(model, batch, config, device):
    """No-update checks; same batch/history at K4, K8, K32, no optimizer.

    Forward timing includes synchronized per-step numerical auditing. Backward
    timing includes synchronized module-gradient checks. These are qualification
    timings, not optimizer-update speed, training ETA or physical accuracy.
    """
    device = torch.device(device)
    if (model.core.damping_mode != 'none' or model.environment_context_dim != 0
            or set(config['training'].get('trainable_modules', ())) != SELECTED_MODULES):
        raise ValueError('Require stage-A hard-zero damping, no environment, and encoder/initializer/potential only')
    if model.environment_encoder is not None or model.core.damping_net is not None or model.core.constant_factor is not None:
        raise ValueError('Disabled damping/environment modules must be absent')
    trainability = configure_trainable_modules(model, config['training'])
    groups = build_optimizer_parameter_groups(model, config['training'])
    if set(trainability['trainable_modules']) != SELECTED_MODULES:
        raise ValueError('Unexpected trainable module partition')
    if batch['pixels'].shape[1] != model.history + max(HORIZONS) + model.config.get('target_tail_frames', 0):
        raise ValueError('Qualification needs exactly the full history plus K32 frames')
    if batch['action'].shape[1] != model.history - 1 + max(HORIZONS) + model.config.get('target_tail_frames', 0):
        raise ValueError('Action coverage differs from the fixed K32 window')
    if not torch.isfinite(batch['action']).all() or torch.count_nonzero(batch['action']) != 0:
        raise ValueError('Undamped passive qualification requires exactly zero actions')
    if any(p.device.type != device.type or (device.index is not None and p.device.index != device.index)
           for p in model.parameters()):
        raise ValueError('Model/device mismatch')
    if any(value.device.type != device.type for value in batch.values()):
        raise ValueError('Batch/device mismatch')
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_hash = _state_hash(before)
    rows = []
    previous_mode = model.training
    try:
        for horizon in HORIZONS:
            backward = horizon in (4, 8)
            model.train(backward)
            model.zero_grad(set_to_none=True)
            _sync(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            audit = dict(core_steps=0, maximum_abs_damping=0., maximum_abs_work=0.,
                         maximum_abs_dissipation=0., ledger_identity_error=0., max_diagnostics={})

            def check_core(module, args, kwargs, output):
                q, p, action, _ = args
                q_next, p_next, diagnostics = output
                if kwargs.get('context') is not None or torch.count_nonzero(action) != 0:
                    raise RuntimeError('Unexpected environment or input during passive rollout')
                with torch.no_grad():
                    for value in (q, p, q_next, p_next, *diagnostics.values()):
                        if not torch.isfinite(value).all():
                            raise FloatingPointError('Nonfinite latent state or numerical diagnostic')
                    damping = module.damping(q, p, context=None)
                    damping_next = module.damping(q_next, p_next, context=None)
                    if torch.count_nonzero(damping) or torch.count_nonzero(damping_next):
                        raise RuntimeError('Hard-zero damping contract failed')
                    for key in ('work', 'dissipation'):
                        if key not in diagnostics or torch.count_nonzero(diagnostics[key]):
                            raise RuntimeError(f'Expected exactly zero {key} in every sample/step')
                    energy0, energy1 = module.energy(q, p), module.energy(q_next, p_next)
                    defect = diagnostics['numerical_balance_defect']
                    error = (energy1 - energy0 - defect).abs()
                    scale = 1 + energy0.abs() + energy1.abs()
                    if (error > 1e-10 * scale).any():
                        raise RuntimeError('Learned-energy change disagrees with numerical ledger')
                    if (diagnostics['solver_residual_ratio'] > 1 + 1e-10).any():
                        raise RuntimeError('Accepted residual exceeds its tolerance')
                    audit['ledger_identity_error'] = max(audit['ledger_identity_error'], float(error.max()))
                    for key, value in diagnostics.items():
                        audit['max_diagnostics'][key] = max(audit['max_diagnostics'].get(key, 0.), float(value.abs().max()))
                    audit['core_steps'] += 1

            hook = model.core.register_forward_hook(check_core, with_kwargs=True)
            started = time.perf_counter()
            try:
                with torch.set_grad_enabled(backward):
                    values = model.loss(batch, horizon)
                metrics = scalar_metrics(values)
                _sync(device)
                forward_seconds = time.perf_counter() - started
                backward_seconds, gradient_report = None, None
                if backward:
                    tick = time.perf_counter()
                    values['loss'].backward()
                    gradient_report = _gradient_audit(model)
                    _sync(device)
                    backward_seconds = time.perf_counter() - tick
                elif any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('No-grad K32 path produced parameter gradients')
            finally:
                hook.remove()
            if audit['core_steps'] != horizon:
                raise RuntimeError('Qualification did not execute the complete requested rollout')
            unchanged = all(torch.equal(before[name], value.detach().cpu())
                            for name, value in model.state_dict().items())
            if not unchanged:
                raise RuntimeError('Qualification unexpectedly changed model parameters or buffers')
            rows.append(dict(horizon=horizon, backward=backward, losses=metrics,
                             forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                             gradients=gradient_report, numerical_audit=audit, parameters_unchanged=unchanged,
                             cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                             cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None))
            del values
    finally:
        model.zero_grad(set_to_none=True)
        model.train(previous_mode)
    return dict(status='completed', optimizer_created=False, optimizer_updates=0,
                checkpoint_saved=False, parameters_unchanged=True, initial_state_sha256=initial_hash,
                final_state_sha256=_state_hash(model.state_dict()), trainability=trainability,
                optimizer_group_names=[group['name'] for group in groups],
                batch_size=int(batch['pixels'].shape[0]), history_frames=model.history,
                rows=rows, physical_accuracy_evaluated=False,
                timing_scope='Forward/backward with synchronized numerical/gradient auditing; no optimizer. Not training-step timing or ETA.')


def execute(config_path, data, output):
    device, allocation = require_cuda_allocation()
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new qualification JSON output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(dict(status='starting', optimizer_created=False, optimizer_updates=0), handle)
    started, dataset = time.perf_counter(), None
    report = dict(status='running', optimizer_created=False, optimizer_updates=0,
                  allocation=allocation, test_samples_read=False, formal_training_started=False,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        config = read_config(config_path)
        cfg = config['training']
        if cfg['batch_size'] != 32 or maximum_horizon(config) < 32:
            raise ValueError('Qualification requires the actual batch32 configuration and K32 coverage')
        if cfg.get('required_data_scope') != 'undamped_stage_a':
            raise ValueError('Qualification requires the explicit undamped_stage_a data scope')
        checked = preflight(data, config)
        torch.set_num_threads(cfg['threads'])
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        model = DissipativeWorldModel(config['model']).to(device)
        dataset = DissipativeWindowDataset(data, 'train', history=model.history,
                                            max_horizon=32, train_frames=cfg['train_frames'],
                                            target_tail_frames=model.config.get('target_tail_frames', 0))
        indices = StepBatchSampler(dataset, 32, cfg['seed']).indices(0)
        batch = move_batch(dataset.batch(indices, horizon=32), device)
        report.update(config=config, config_sha256=config_digest(config),
                      dataset_sha256=checked['dataset_sha256'], source_hashes=source_hashes(),
                      runtime=runtime_signature(device), data_path=str(Path(data).resolve()),
                      fixed_train_window_indices=[int(i) for i in indices], sampler_step=0,
                      preflight=checked)
        report['checks'] = run_checks(model, batch, config, device)
        report.update(status='completed', elapsed_seconds=time.perf_counter() - started)
        atomic_json(output, report)
        print(json.dumps(report, allow_nan=False), flush=True)
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.perf_counter() - started)
        atomic_json(output, report)
        raise
    finally:
        if dataset is not None:
            dataset.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON audit file; never overwritten')
    args = parser.parse_args(argv)
    return execute(args.config, args.data, args.output)


if __name__ == '__main__':
    main()
