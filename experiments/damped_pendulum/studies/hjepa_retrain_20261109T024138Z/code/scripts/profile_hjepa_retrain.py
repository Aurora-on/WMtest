#!/usr/bin/env python3
"""Profile a disposable H-JEPA model before formal training; no checkpoint reuse."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from dpbench.comparison_models import build_comparison_model
from dpbench.comparison_protocol import preflight, json_digest
from dpbench.dissipative_data import DissipativeWindowDataset, StepBatchSampler, fixed_validation_indices
from dpbench.train_comparison import _seed, scalar_losses, validation
from dpbench.train_dissipative import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text())
    report = preflight(args.data, cfg)
    if cfg['model']['kind'] != 'hjepa' or not torch.cuda.is_available():
        raise RuntimeError('Requires H-JEPA and an allocated CUDA GPU')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = dict(status='running', protocol_sha256=json_digest(cfg),
                  dataset_sha256=report['dataset_sha256'], formal_training_started=False,
                  scope='Disposable randomly initialized model: 2 warmup +5 measured optimizer updates, full latent validation; no training checkpoint',
                  device=torch.cuda.get_device_name(0), rows=[])
    atomic_json(args.output, record)
    train = val = None
    try:
        torch.set_num_threads(cfg['training']['threads'])
        _seed(cfg['training']['seed'])
        model = build_comparison_model(cfg['model'], device='cuda')
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
        train = DissipativeWindowDataset(args.data, 'train', 21, 32, 81, 2)
        val = DissipativeWindowDataset(args.data, 'val', 21, 32, 81, 2)
        sampler = StepBatchSampler(train, cfg['training']['batch_size'], cfg['training']['seed'])
        torch.cuda.reset_peak_memory_stats()
        model.train()
        for i in range(7):
            torch.cuda.synchronize()
            start = time.perf_counter()
            batch = {k: v.to('cuda') for k, v in train.batch(sampler.indices(i), 5).items()}
            optimizer.zero_grad(set_to_none=True)
            losses = model.loss(batch, 5)
            values = scalar_losses(losses)
            losses['loss'].backward()
            damping_parameters = list(model.predictor.r.parameters())
            if not all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in damping_parameters):
                raise RuntimeError('Dissipation network is not fully trainable with finite gradients')
            damping_grad = math.sqrt(sum(float(p.grad.square().sum()) for p in damping_parameters))
            if damping_grad <= 0:
                raise RuntimeError('Dissipation gradient is zero')
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            record['rows'].append(dict(warmup=i < 2, seconds=time.perf_counter()-start,
                                       losses=values, damping_gradient_norm=damping_grad,
                                       gradient_norm=float(gradient)))
            atomic_json(args.output, record)
        record['train_peak_bytes'] = torch.cuda.max_memory_allocated()
        del batch, losses
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        indices = fixed_validation_indices(val, cfg['training']['val_windows'], cfg['training']['val_seed'])
        record['latent_validation'] = validation(model, val, indices, cfg, torch.device('cuda'))
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - start
        mean_step = statistics.mean(row['seconds'] for row in record['rows'] if not row['warmup'])
        record.update(status='passed', validation_peak_bytes=torch.cuda.max_memory_allocated(),
                      mean_train_step_seconds=mean_step, validation_seconds=validation_seconds,
                      estimated_20k_train_and_latent_validation_hours=(20000*mean_step+80*validation_seconds)/3600,
                      estimate_excludes=['queue', 'checkpoint writes', 'physical evaluation', 'later model/runtime changes'])
        atomic_json(args.output, record)
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        atomic_json(args.output, record)
        raise
    finally:
        if train is not None: train.close()
        if val is not None: val.close()
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
