#!/usr/bin/env python3
"""Independent time-refinement audit of the frozen learned ODE (not physical truth).

No parameters are optimized. DOP853 is an independent time discretization of
qdot=p, pdot=-grad V(q)-D(q,p,c)p, using the same frozen learned force functions.
The deployed solver tolerances remain unchanged across the DG step-size sweep.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from dpbench.dissipative_core import DissipativeCore


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(core,q,p,context,duration,rtol,atol,max_step):
    """Independent explicit adaptive integrator; append continuous dissipated work."""
    d=q.shape[-1]
    def rhs(t,y):
        del t
        with torch.enable_grad():
            qt=torch.tensor(y[:d],dtype=torch.float64).reshape(1,d).requires_grad_(True)
            vt=torch.tensor(y[d:2*d],dtype=torch.float64).reshape(1,d)
            gradient=torch.autograd.grad(core.potential(qt).sum(),qt)[0]
        with torch.no_grad():
            force=(core.damping(qt.detach(),vt,context)@vt.unsqueeze(-1)).squeeze(-1)
            dy=torch.cat((vt,-gradient-force),-1)[0].numpy()
            dissipative_power=(force*vt).sum().item()
        return np.r_[dy,dissipative_power]
    initial=np.r_[q.detach().numpy().ravel(),p.detach().numpy().ravel(),0.]
    start=time.perf_counter()
    result=solve_ivp(rhs,(0.,duration),initial,method='DOP853',rtol=rtol,atol=atol,
                     max_step=max_step,dense_output=True)
    if not result.success or not np.isfinite(result.y).all():
        raise RuntimeError(f'DOP853 reference failed: {result.message}')
    energy0=float(core.energy(q,p).detach())
    end=torch.as_tensor(result.y[:2*d,-1]).reshape(1,2*d)
    end_energy=float(core.energy(end[:,:d],end[:,d:]).detach())
    return result,dict(rtol=rtol,atol=atol,max_step=max_step,nfev=result.nfev,
        elapsed_seconds=time.perf_counter()-start,energy_balance_defect=end_energy-energy0+result.y[-1,-1],
        endpoint=result.y[:2*d,-1].tolist(),integrated_dissipation=result.y[-1,-1])


@torch.no_grad()
def discrete_trajectory(core,q,p,context,duration,dt):
    steps=round(duration/dt)
    if abs(steps*dt-duration)>1e-12:raise ValueError('Duration must fit all grids exactly')
    energy0=core.energy(q,p).item();states=[torch.cat((q,p),-1).numpy()[0]]
    total_diss=total_work=total_defect=0.;max_ratio=0.;max_depth=0.;max_ledger_error=0.
    totals={k:0. for k in ('solver_iterations','solver_backtracks','solver_failed_attempts','solver_leaf_steps')}
    start=time.perf_counter()
    for _ in range(steps):
        before=core.energy(q,p).item()
        q,p,info=core(q,p,torch.zeros((1,core.action_dim),dtype=torch.float64),dt,context=context)
        values={k:float(v.item()) for k,v in info.items()}
        if not all(np.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite DG ledger')
        after=core.energy(q,p).item()
        difference=after-before-values['work']+values['dissipation']
        max_ledger_error=max(max_ledger_error,abs(difference-values['numerical_balance_defect']))
        if values['solver_residual_ratio']>1+1e-10 or values['dissipation'] < -1e-10:
            raise RuntimeError('DG tolerance or passive dissipation check failed')
        total_diss+=values['dissipation'];total_work+=values['work'];total_defect+=values['numerical_balance_defect']
        max_ratio=max(max_ratio,values['solver_residual_ratio']);max_depth=max(max_depth,values['solver_split_depth'])
        for k in totals:totals[k]+=values[k]
        states.append(torch.cat((q,p),-1).numpy()[0])
    macro_defect=core.energy(q,p).item()-energy0-total_work+total_diss
    if abs(macro_defect-total_defect)>1e-10*(1+abs(energy0)+abs(total_diss)):
        raise RuntimeError('Accumulated energy ledger is inconsistent')
    return np.asarray(states),dict(frame_dt=dt,base_leaf_h=dt/core.substeps,frames=steps,
        max_accepted_residual_ratio=max_ratio,max_split_depth=max_depth,**totals,
        integrated_dissipation=total_diss,accumulated_energy_defect=macro_defect,
        max_ledger_identity_error=max_ledger_error,elapsed_seconds=time.perf_counter()-start)


def add_errors(rows):
    for i,row in enumerate(rows):
        row['endpoint_observed_order_from_previous']=None
        row['trajectory_observed_order_from_previous']=None
        if i:
            previous=rows[i-1]
            factor=previous['frame_dt']/row['frame_dt']
            for field,key in [('endpoint_error_l2','endpoint_observed_order_from_previous'),
                              ('common_grid_rms_state_error','trajectory_observed_order_from_previous')]:
                if row[field]>0 and previous[field]>0:
                    row[key]=float(np.log(previous[field]/row[field])/np.log(factor))
    return rows


class Harmonic(nn.Module):
    def __init__(self,omega):super().__init__();self.omega=omega
    def forward(self,q):return 0.5*self.omega**2*q.square().sum(-1,keepdim=True)


def run(capture_path,output):
    torch.set_num_threads(1)
    started=time.perf_counter()
    capture=torch.load(capture_path,map_location='cpu',weights_only=True)
    opts=dict(capture['core_config'],solver='backtracking',newton_iterations=12,max_bisections=4,
              line_search_max_backtracks=12,line_search_shrink=.5,line_search_c1=1e-4)
    core=DissipativeCore(**opts);core.load_state_dict(capture['core_state_dict']);core.eval()
    for par in core.parameters():par.requires_grad_(False)
    index=20
    q,p,c=(capture['input'][k][index:index+1].clone() for k in ('q','p','context'))
    duration=0.8;dts=(.1,.05,.025,.0125);common=np.linspace(0,duration,9)
    ref,ref_info=reference(core,q,p,c,duration,1e-11,1e-12,.025)
    tight,tight_info=reference(core,q,p,c,duration,1e-12,1e-13,.00625)
    reference_disagreement=float(np.max(np.linalg.norm(ref.sol(common)[:4].T-tight.sol(common)[:4].T,axis=1)))
    rows=[]
    for dt in dts:
        states,info=discrete_trajectory(core,q.clone(),p.clone(),c,duration,dt)
        indices=np.rint(common/dt).astype(int)
        error=states[indices]-tight.sol(common)[:4].T
        info.update(endpoint_error_l2=float(np.linalg.norm(error[-1])),common_grid_rms_state_error=float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
                    common_grid_max_state_error=float(np.max(np.linalg.norm(error,axis=1))),endpoint=states[-1].tolist())
        rows.append(info)
    add_errors(rows)
    harmonic=DissipativeCore(q_dim=1,damping_mode='none',substeps=2,solver='backtracking',newton_iterations=12,max_bisections=4)
    omega=1.7;harmonic.potential_net=Harmonic(omega)
    for par in harmonic.parameters():par.requires_grad_(False)
    hq=torch.tensor([[.7]],dtype=torch.float64);hp=torch.tensor([[-.2]],dtype=torch.float64)
    hduration=1.6;hcommon=np.linspace(0,hduration,17)
    exact=np.stack((.7*np.cos(omega*hcommon)-.2/omega*np.sin(omega*hcommon),
                    -.7*omega*np.sin(omega*hcommon)-.2*np.cos(omega*hcommon)),-1)
    hrows=[]
    for dt in dts:
        states,info=discrete_trajectory(harmonic,hq.clone(),hp.clone(),None,hduration,dt)
        err=states[np.rint(hcommon/dt).astype(int)]-exact
        info.update(endpoint_error_l2=float(np.linalg.norm(err[-1])),common_grid_rms_state_error=float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),common_grid_max_state_error=float(np.max(np.linalg.norm(err,axis=1))))
        hrows.append(info)
    add_errors(hrows)
    result={'status':'completed','scope':'Independent time-discretization convergence of one frozen learned ODE, not real physical prediction accuracy or global solver proof',
        'capture_sha256':sha(capture_path),'core_source_sha256':sha(ROOT/'dpbench/dissipative_core.py'),'script_sha256':sha(__file__),
        'torch':str(torch.__version__),'scipy':__import__('scipy').__version__,'device':'cpu','parameters_optimized':False,
        'learned_case':{'capture_sample_index':index,'training_step':capture['training_step'],'duration_seconds':duration,'core_config':core.config,
            'q0':q.numpy().tolist(),'p0':p.numpy().tolist(),'context':c.numpy().tolist(),'reference':ref_info,'tighter_reference':tight_info,
            'max_common_grid_reference_disagreement':reference_disagreement,'sweep':rows,
            'endpoint_errors_strictly_decrease':all(rows[i]['endpoint_error_l2']<rows[i-1]['endpoint_error_l2'] for i in range(1,len(rows))),
            'trajectory_errors_strictly_decrease':all(rows[i]['common_grid_rms_state_error']<rows[i-1]['common_grid_rms_state_error'] for i in range(1,len(rows)))},
        'harmonic_case':{'omega':omega,'duration_seconds':hduration,'reference':'Analytic q,p of conservative harmonic oscillator','core_config':harmonic.config,'sweep':hrows},
        'interpretation_limits':['Observed orders on four grids are empirical, not a universal convergence theorem.','Adaptive subdivision can make effective leaf grids differ; leaf count/depth is reported.','Original production residual/chain tolerances are unchanged; tolerance error can dominate on sufficiently fine grids.','Learned reference uses the same frozen V,D but an independent explicit adaptive integrator; its trajectory is not simulator ground truth.'],
        'elapsed_seconds':time.perf_counter()-started}
    output=Path(output)
    if output.exists():raise FileExistsError('Do not overwrite existing refinement evidence')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'learned_errors':[r['endpoint_error_l2'] for r in rows],'learned_orders':[r['endpoint_observed_order_from_previous'] for r in rows],
                      'harmonic_orders':[r['endpoint_observed_order_from_previous'] for r in hrows],'reference_disagreement':reference_disagreement,'seconds':result['elapsed_seconds']}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,default=ROOT/'studies/passive_failure_20261104/replay_47690/failure_capture.pt')
    p.add_argument('--output',type=Path,default=ROOT/'studies/robust_solver_20261104/time_refinement.json')
    a=p.parse_args();run(a.capture,a.output)
