#!/usr/bin/env python3
"""Strict-batch autonomous numerical audit of a normal-training checkpoint.

Calls model.rollout directly: no exception-driven batch shrinking, sample skips,
optimizer or physical readout. All initial histories are from seen validation;
only the 21 initial frames are loaded. Any failed batch terminates the audit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dpbench.dissipative_data import DissipativeWindowDataset
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.passive_family import sha256
from dpbench.train_dissipative import (atomic_json, preflight, read_config,
    runtime_signature, source_hashes, validate_checkpoint)


def validate_trajectory(model, result):
    """Independently recompute frame and sequence ledgers from returned states."""
    tensors = [result['q'], result['p'], result['history_encoding'], result['correction']]
    if result['context'] is not None:
        tensors.append(result['context'])
    tensors += list(result['diagnostics'].values())
    if not all(torch.isfinite(x).all() for x in tensors):
        raise FloatingPointError('Nonfinite state, initialization or solver diagnostic')
    q, p = result['q'], result['p']
    with torch.no_grad():
        energy = model.core.energy(q, p)
    if not torch.isfinite(energy).all():
        raise FloatingPointError('Nonfinite recomputed latent energy')
    diagnostics = result['diagnostics']
    needed = {'solver_residual_ratio','solver_split_depth','solver_backtracks','solver_iterations',
              'solver_replay_relative_error','solver_failed_attempts','solver_leaf_steps',
              'work','dissipation','energy_residual','numerical_balance_defect','residual_work','chain_defect'}
    if needed-set(diagnostics):
        raise ValueError(f'Missing solver diagnostics: {sorted(needed-set(diagnostics))}')
    if (diagnostics['solver_residual_ratio'] > 1+1e-10).any():
        raise RuntimeError('An accepted leaf exceeds the original residual tolerance')
    if (diagnostics['solver_split_depth'] > model.core.max_bisections).any():
        raise RuntimeError('Bisection budget exceeded')
    w, d = diagnostics['work'], diagnostics['dissipation']
    if (d < -1e-10).any() or (w.abs() > 1e-12).any():
        raise RuntimeError('Invalid passive work/dissipation')
    actual = energy[:,1:]-energy[:,:-1]-w+d
    defect = diagnostics['numerical_balance_defect']
    scale = 1+energy[:,1:].abs()+energy[:,:-1].abs()+w.abs()+d.abs()
    checks = {
        'frame_identity': (actual-defect).abs(),
        'frame_reported_residual': (actual-diagnostics['energy_residual']).abs(),
        'frame_components': (defect-diagnostics['residual_work']-diagnostics['chain_defect']).abs(),
    }
    for name,error in checks.items():
        if (error > 1e-10*scale).any():
            raise RuntimeError(f'Energy ledger disagreement: {name}')
    sequence = energy[:,-1]-energy[:,0]-w.sum(1)+d.sum(1)
    sequence_error = (sequence-defect.sum(1)).abs()
    sequence_scale = 1+energy[:,0].abs()+energy[:,-1].abs()+w.abs().sum(1)+d.abs().sum(1)
    if (sequence_error > 1e-10*sequence_scale).any():
        raise RuntimeError('Full-sequence accumulated energy ledger disagreement')
    stats = {f'{name}_max':float(error.max()) for name,error in checks.items()}
    stats.update(sequence_identity_max=float(sequence_error.max()),
                 energy_balance_defect_max=float(actual.abs().max()),
                 sequence_energy_balance_defect_max=float(sequence.abs().max()),
                 q_norm_max=float(q.norm(dim=-1).max()),p_norm_max=float(p.norm(dim=-1).max()),
                 energy_abs_max=float(energy.abs().max()))
    return energy, stats


def execute(args):
    config = read_config(args.config)
    device = torch.device(args.device)
    torch.set_num_threads(config['training']['threads'])
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    checked = preflight(args.data,config)
    validate_checkpoint(checkpoint,config,checked['dataset_sha256'],source_hashes(),device,runtime_signature(device))
    model = DissipativeWorldModel(config['model']).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    if model.config != checkpoint['config']:
        raise ValueError('Resolved model configuration differs from checkpoint')
    if model.core.solver != 'backtracking' or not model.core.strict:
        raise ValueError('Expected strict robust solver')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    val = DissipativeWindowDataset(args.data,'val',history=model.history,max_horizon=32,
                                  train_frames=config['training']['train_frames'])
    initial = [ref for ref in val.index if ref.start_local==0]
    if len(initial) != len(val.split_ids) or len(initial) != args.expected_episodes:
        val.close()
        raise ValueError('Validation initial-history coverage differs from the declared expected count')
    if len({ref.episode for ref in initial}) != len(initial):
        val.close()
        raise ValueError('Duplicate validation trajectories')
    if args.output.exists():
        val.close()
        raise FileExistsError('Choose a new strict audit directory')
    args.output.mkdir(parents=True)
    manifest = dict(status='running',checkpoint=str(args.checkpoint.resolve()),
        checkpoint_step=checkpoint['step'],checkpoint_sha256=sha256(args.checkpoint),
        config=config,source_hashes=source_hashes(),script_sha256=sha256(__file__),
        data_sha256=checked['dataset_sha256'],runtime=runtime_signature(device),
        batch_size=args.batch_size,expected_episodes=args.expected_episodes,rollout_steps=args.steps,
        history=model.history,dt=model.dt,optimizer_updates=0,physical_readout_fitted=False,
        test_samples_read=False,future_pixels_read=False,
        execution='Direct model.rollout under torch.no_grad; zero exception-driven batch retries or sample skips',
        parameter_updates_enabled=False,strict_batch=True,completed_episodes=0,
        limits='Numerical stability on this checkpoint and validation initial histories; not physical accuracy or universal convergence')
    atomic_json(args.output/'manifest.json',manifest)
    start=time.perf_counter();complete=0;batch_offset=None;episodes=[];maxima={};all_counts={};ledger_maxima={}
    try:
        with h5py.File(args.data,'r') as f:
            metadata = [{'episode':ref.episode,'group_id':int(f['group_id'][ref.episode]),
                         'damping':float(f['damping'][ref.episode]),'energy_band':int(f['energy_band'][ref.episode])}
                        for ref in initial]
            atomic_json(args.output/'initial_histories.json',metadata)
            for batch_offset in range(0,len(initial),args.batch_size):
                refs=initial[batch_offset:batch_offset+args.batch_size]
                episodes=[ref.episode for ref in refs]
                pixels=np.stack([f['pixels'][ref.start:ref.start+model.history] for ref in refs])
                pixels=np.ascontiguousarray(pixels.transpose(0,1,4,2,3))
                images=torch.as_tensor(pixels,device=device,dtype=torch.float32)/255.
                past_actions=torch.as_tensor(np.stack([f['action'][ref.start:ref.start+model.history-1] for ref in refs]),device=device,dtype=torch.float32)
                past_dt=torch.as_tensor(np.stack([f['dt_frame'][ref.start:ref.start+model.history-1] for ref in refs]),device=device,dtype=torch.float32)
                if torch.count_nonzero(past_actions):
                    raise ValueError('Strict passive audit requires zero past actions')
                actions=torch.zeros(len(refs),args.steps,1,device=device,dtype=torch.float32)
                intervals=torch.full_like(actions,model.dt)
                if device.type=='cuda':
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                tick=time.perf_counter()
                with torch.no_grad():
                    # Deliberately no try/retry around this call.
                    result=model.rollout(images,actions,past_actions,past_dt,intervals)
                    energy,stats=validate_trajectory(model,result)
                if device.type=='cuda':torch.cuda.synchronize()
                seconds=time.perf_counter()-tick
                diag={k:v.detach().cpu().numpy() for k,v in result['diagnostics'].items()}
                for key,value in diag.items():
                    maxima[key]=max(maxima.get(key,0.),float(np.max(abs(value))))
                    if key.startswith('solver_'):
                        all_counts.setdefault(key,[]).append(value.ravel())
                for key,value in stats.items():ledger_maxima[key]=max(ledger_maxima.get(key,0.),value)
                np.savez_compressed(args.output/f'batch_{batch_offset:04d}.npz',
                    q=result['q'].cpu().numpy(),p=result['p'].cpu().numpy(),latent_energy=energy.cpu().numpy(),
                    context=result['context'].cpu().numpy() if result['context'] is not None else np.empty((len(refs),0)),
                    episodes=np.asarray(episodes),**diag)
                complete+=len(refs)
                atomic_json(args.output/'progress.json',{'completed_episodes':complete,'expected_episodes':len(initial),
                    'last_batch_offset':batch_offset,'last_batch_seconds':seconds,
                    'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                    'diagnostic_maxima':maxima,'ledger_and_state_maxima':ledger_maxima})
                print(json.dumps({'phase':'strict_rollout','completed_episodes':complete,'batch_seconds':seconds}),flush=True)
        unchanged=all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items())
        if not unchanged:raise RuntimeError('Model state changed during read-only audit')
        manifest.update(status='completed',ready_numerically=True,completed_episodes=complete,
            autonomous_transitions=complete*args.steps,diagnostic_maxima=maxima,
            ledger_and_state_maxima=ledger_maxima,
            solver_quantiles={k:np.quantile(np.concatenate(v),[.5,.95,.99,1.]).tolist() for k,v in all_counts.items()},
            model_state_unchanged=unchanged,elapsed_seconds=time.perf_counter()-start,
            batch_retries=0,samples_skipped=0)
    except BaseException as exc:
        manifest.update(status='failed',ready_numerically=False,completed_episodes=complete,
            failed_batch_offset=batch_offset,failed_batch_episode_ids=episodes,error_type=type(exc).__name__,
            error=str(exc),elapsed_seconds=time.perf_counter()-start,batch_retries=0,samples_skipped=0)
        raise
    finally:
        atomic_json(args.output/'manifest.json',manifest);val.close()
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','config','data','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--steps',type=int,default=320)
    parser.add_argument('--expected-episodes',type=int,default=384)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if min(args.batch_size,args.steps,args.expected_episodes)<1:parser.error('Positive counts required')
    if not args.execute:
        print(json.dumps({'mode':'plan_only','parameter_updates_enabled':False,'strict_batch':True,
            'batch_size':args.batch_size,'steps':args.steps,'expected_episodes':args.expected_episodes}))
    else:
        execute(args)
