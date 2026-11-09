#!/usr/bin/env python3
"""Bounded K32 optimizer stress and 320-step validation-initialized numerical audit.

Only train/val histories are read. No physical readout, test labels, or scientific
prediction scores. The stress uses base LR and a forced horizon, so its weight
artifact is deliberately NOT a resumable normal-training checkpoint.
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.dissipative_data import DissipativeWindowDataset,StepBatchSampler,fixed_validation_indices
from dpbench.train_dissipative import (atomic_json,atomic_torch_save,preflight,read_config,source_hashes,
    validate_checkpoint,runtime_signature,restore_rng,move_batch,scalar_metrics,validation)
from dpbench.evaluate_passive import autonomous_rollout
from scripts.profile_dissipative import _optimizer
from dpbench.passive_family import sha256


def execute(args):
    cfg=read_config(args.config); settings=cfg['training']; device=torch.device(args.device)
    if args.output.exists(): raise FileExistsError('Choose a new qualification output directory')
    report=preflight(args.data,cfg)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    validate_checkpoint(checkpoint,cfg,report['dataset_sha256'],source_hashes(),device,runtime_signature(device))
    torch.set_num_threads(settings['threads'])
    model=DissipativeWorldModel(cfg['model']).to(device)
    model.load_state_dict(checkpoint['state_dict']); optimizer=_optimizer(model,cfg)
    optimizer.load_state_dict(checkpoint['optimizer']);restore_rng(checkpoint['rng'])
    for group in optimizer.param_groups: group['lr']=group['base_lr']
    kwargs=dict(history=model.history,max_horizon=32,train_frames=settings['train_frames'])
    train=DissipativeWindowDataset(args.data,'train',**kwargs);val=DissipativeWindowDataset(args.data,'val',**kwargs)
    sampler=StepBatchSampler(train,settings['batch_size'],settings['seed'])
    initial_ids=[i for i,ref in enumerate(val.index) if ref.start_local==0]
    if len(initial_ids)!=len(val.split_ids):raise ValueError('Require exactly one initial history per validation episode')
    anchors=fixed_validation_indices(val,settings['val_windows'],settings['val_seed'])
    args.output.mkdir(parents=True)
    manifest={'status':'running','scope':'bounded numerical readiness stress, not normal-schedule training or physical test evaluation',
        'checkpoint_sha256':sha256(args.checkpoint),'checkpoint_step':checkpoint['step'],'config':cfg,
        'source_hashes':source_hashes(),'script_sha256':sha256(__file__),'data_sha256':report['dataset_sha256'],
        'updates_budget':args.updates,'horizon':args.horizon,'learning_rate_scope':'full configured base LR with inherited Adam state',
        'runtime':runtime_signature(device),'test_samples_read':False,'physical_readout_fitted':False}
    atomic_json(args.output/'manifest.json',manifest)
    start=time.perf_counter(); model.train(); rows=[]
    try:
        with (args.output/'history.jsonl').open('x') as log:
            for index in range(args.updates):
                if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                tick=time.perf_counter()
                batch=move_batch(train.batch(sampler.indices(checkpoint['step']+index),horizon=args.horizon),device)
                optimizer.zero_grad(set_to_none=True)
                loss=model.loss(batch,args.horizon);metrics=scalar_metrics(loss)
                loss['loss'].backward()
                grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'],error_if_nonfinite=True))
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite parameter after update')
                if device.type=='cuda':torch.cuda.synchronize()
                row={'stress_update':index+1,'sampler_absolute_step':checkpoint['step']+index+1,
                     'horizon':args.horizon,'seconds':time.perf_counter()-tick,'gradient_norm':grad,'losses':metrics,
                     'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None}
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if (index+1)%25==0:print(json.dumps(row),flush=True)
        manifest['optimizer_updates_completed']=len(rows)
        manifest['validation']=validation(model,val,anchors,cfg,device)
        atomic_torch_save(args.output/'stress_weights.pt',{'kind':'robust_solver_stress_weights',
            'config':model.config,'state_dict':model.state_dict(),'updates':len(rows),'base_checkpoint_sha256':sha256(args.checkpoint)})
        # Exactly one initial history per seen-validation trajectory, all six damping values.
        ids=initial_ids
        manifest['long_rollout_initial_histories']=len(ids)
        if not ids:raise ValueError('No validation initial histories')
        failed=[]; valid=[];diagnostics={};energy_increases=0;transitions=0;balance_defect=0.;ledger_error=0.
        names=('solver_residual','solver_residual_ratio','energy_residual','dissipation','work',
               'solver_iterations','solver_replay_iterations','solver_replay_relative_error','solver_backtracks','solver_line_search_evals',
               'solver_failed_attempts','solver_leaf_steps','solver_split_depth','numerical_balance_defect',
               'residual_work','chain_defect','chain_defect_max')
        model.eval()
        for offset in range(0,len(ids),settings['batch_size']):
            batch=move_batch(val.batch(ids[offset:offset+settings['batch_size']],horizon=32),device)
            b=len(batch['pixels']);history=model.history
            past_actions=batch['action'][:,:history-1];past_dt=batch['dt_frame'][:,:history-1]
            actions=torch.zeros(b,args.rollout_steps,1,device=device)
            intervals=torch.full((b,args.rollout_steps,1),model.dt,device=device)
            result=autonomous_rollout(model,batch['pixels'][:,:history],past_actions,past_dt,actions,intervals,diagnostic_names=names)
            valid.extend(result['valid_steps'].tolist())
            for fail in result['failures']:
                failed.append({**fail,'dataset_window':ids[offset+fail['batch_index']]})
            for key,value in result['diagnostics'].items():
                finite_values=value[np.isfinite(value)]
                if len(finite_values):diagnostics[key]=max(diagnostics.get(key,0.),float(np.max(np.abs(finite_values))))
            delta=np.diff(result['latent_energy'],axis=1);diss=result['diagnostics']['dissipation']
            measured=delta+diss
            for i,completed in enumerate(result['valid_steps']):
                if not np.isfinite(result['features'][i,:completed+1]).all():raise FloatingPointError('Nonfinite accepted latent state')
                if not np.isfinite(result['latent_energy'][i,:completed+1]).all():raise FloatingPointError('Nonfinite accepted energy')
                if not completed:continue
                dd={k:v[i,:completed] for k,v in result['diagnostics'].items()}
                if not all(np.isfinite(v).all() for v in dd.values()):raise FloatingPointError('Nonfinite accepted diagnostic')
                if np.max(dd['solver_residual_ratio'])>1+1e-10:raise RuntimeError('Accepted residual ratio exceeds one')
                if np.max(dd['solver_split_depth'])>model.core.max_bisections:raise RuntimeError('Bisection budget exceeded')
                if np.min(dd['dissipation']) < -1e-10 or np.max(abs(dd['work']))>1e-12:raise RuntimeError('Invalid passive energy ledger')
                observed=measured[i,:completed]; defect=dd['numerical_balance_defect']
                scale=1+abs(result['latent_energy'][i,:completed])+abs(result['latent_energy'][i,1:completed+1])+abs(diss[i,:completed])
                error=abs(observed-defect)
                if np.any(error>1e-10*scale):raise RuntimeError('Energy change differs from accumulated numerical ledger')
                if np.any(abs(defect-dd['residual_work']-dd['chain_defect'])>1e-10*scale):raise RuntimeError('Ledger components disagree')
                ledger_error=max(ledger_error,float(error.max()))
                balance_defect=max(balance_defect,float(abs(observed).max()))
            energy_increases+=int(np.sum(delta>1e-6));transitions+=int(np.isfinite(delta).sum())
            np.savez_compressed(args.output/f'rollout_{offset:04d}.npz',features=result['features'],
                valid_steps=result['valid_steps'],latent_energy=result['latent_energy'],**result['diagnostics'])
            atomic_json(args.output/'long_rollout_progress.json',{'completed':len(valid),'failed':failed,
                'steps':args.rollout_steps,'max_diagnostics':diagnostics})
            print(json.dumps({'phase':'long_rollout','completed':len(valid),'failures':len(failed)}),flush=True)
        manifest.update(status='completed',long_rollout_steps=args.rollout_steps,long_rollout_failures=failed,
            long_rollout_completed=sum(s==args.rollout_steps for s in valid),long_rollout_total=len(valid),
            long_rollout_max_diagnostics=diagnostics,long_rollout_max_energy_balance_defect=balance_defect,
            long_rollout_max_ledger_identity_error=ledger_error,
            long_rollout_energy_increases_above_1e_6=energy_increases,long_rollout_transitions=transitions,
            mean_update_seconds=float(np.mean([r['seconds'] for r in rows])),elapsed_seconds=time.perf_counter()-start,
            ready_numerically=not failed and len(rows)==args.updates)
        if failed:raise RuntimeError(f'{len(failed)} long-rollout failures; readiness rejected')
    except BaseException as exc:
        manifest.update(status='failed',optimizer_updates_completed=len(rows),error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.perf_counter()-start,ready_numerically=False)
        raise
    finally:
        atomic_json(args.output/'manifest.json',manifest);train.close();val.close()
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','config','data','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--updates',type=int,default=500)
    p.add_argument('--horizon',type=int,choices=(4,8,16,32),default=32);p.add_argument('--rollout-steps',type=int,default=320)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.updates<1 or a.rollout_steps<1:p.error('Positive budgets required')
    if not a.execute:print(json.dumps({'mode':'plan_only','updates':a.updates,'horizon':a.horizon,'rollout_steps':a.rollout_steps}))
    else:execute(a)
