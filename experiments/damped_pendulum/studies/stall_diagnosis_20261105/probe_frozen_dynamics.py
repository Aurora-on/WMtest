"""Frozen-checkpoint forces and integrator sensitivity, with no optimizer."""
import copy, hashlib, json, sys, time, types
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from dpbench.efficient_dissipative_core import EfficientDissipativeCore
from dpbench.dissipative_core import DissipativeCore
from scripts.diagnose_teacher_forced_readout import load_readout, metrics
from dpbench.evaluate import decode_state

torch.set_num_threads(2)
torch.manual_seed(0)
EVAL = ROOT/'studies/efficient3_evaluation_20261105T023318Z'
checkpoint = HERE/'frozen_training/best.pt'
sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
assert sha == '5163a6a293a7c773a90e9ef59a62353a4debdba64c0dc24814bcbdb3b47f9042'
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
config = saved['config']
state = {k.removeprefix('core.'):v for k,v in saved['state_dict'].items() if k.startswith('core.')}
def make_core(substeps, kind='dg', scale=1.):
    cfg = copy.deepcopy(config['core'])
    cfg['substeps'] = substeps
    if kind == 'rk4':
        for k in ('picard_iterations','picard_check_every','fd_relative_step','implicit_rcond_min','adjoint_rtol'):
            cfg.pop(k,None)
        cfg.update(integrator='rk4',solver='full',max_bisections=0)
        core = DissipativeCore(q_dim=config['q_dim'], context_dim=config['environment_context_dim'],**cfg)
    else:
        core = EfficientDissipativeCore(q_dim=config['q_dim'], context_dim=config['environment_context_dim'],**cfg)
    core.load_state_dict(state, strict=True)
    core.eval()
    for par in core.parameters(): par.requires_grad_(False)
    if scale != 1.:
        original_damping = core.damping
        core.damping = types.MethodType(lambda self,q,p,context=None: scale*original_damping(q,p,context),core)
    return core

a=np.load(EVAL/'received/trajectories.npz')
records=json.loads((EVAL/'received/episodes.json').read_text())
teacher=json.loads((EVAL/'teacher_received/diagnostic.json').read_text())
teacher_a=np.load(EVAL/'teacher_received/diagnostic_arrays.npz')
core=make_core(3)
tt=[0,1,2,5,10,20,40,80,160,320]
z=torch.tensor(a['features'][:,tt].reshape(-1,4))
c=torch.tensor(np.repeat(a['context'][:,None,:],len(tt),axis=1).reshape(-1,4))
q,p=z[:,:2],z[:,2:]
with torch.no_grad():
    V=core.potential(q); raw=core.potential_net[0](q).squeeze(-1)
    grad=core.potential_gradient(q)
    D=core.damping(q,p,c); eig=torch.linalg.eigvalsh(D)
    damping_force=(D@p[...,None]).squeeze(-1)
    out={
      'V':V,'V_raw':raw,'restoring_force_norm':grad.norm(dim=-1),
      'damping_eigen_min':eig[:,0],'damping_eigen_max':eig[:,1],
      'damping_force_norm':damping_force.norm(dim=-1),
      'p_norm':p.norm(dim=-1),'kinetic_energy':.5*p.square().sum(-1),
      'effective_damping_rate':(p*damping_force).sum(-1)/p.square().sum(-1).clamp_min(1e-30),
      'softplus_multiplier':torch.sigmoid(raw),
    }
    out={k:v.numpy().reshape(len(records),len(tt)) for k,v in out.items()}

def stats(values):
    v=np.asarray(values)
    return dict(mean=float(np.mean(v)),median=float(np.median(v)),p05=float(np.quantile(v,.05)),p95=float(np.quantile(v,.95)))
force_rows=[]
for b in sorted({e['damping'] for e in records}):
    mask=np.array([e['damping']==b for e in records])
    for j,t in enumerate(tt):
        force_rows.append(dict(damping=b,time_s=t*.1,n=int(mask.sum()),**{k:stats(v[mask,j]) for k,v in out.items()}))
# Evaluate the potential on independently re-encoded true trajectories as well.
tz=torch.tensor(teacher_a['teacher_features'].reshape(-1,4))
with torch.no_grad():
    true_force=core.potential_gradient(tz[:,:2]).norm(dim=-1).numpy()
    true_p=tz[:,2:].norm(dim=-1).numpy()
    true_V=core.potential(tz[:,:2]).numpy()
force_result=dict(status='completed',checkpoint_sha256=sha,scope='Frozen learned functions, not physical coefficient identification',
    rows=force_rows,true_encoded_sample_summary=dict(restoring_force_norm=stats(true_force),p_norm=stats(true_p),V=stats(true_V)))
(HERE/'frozen_force_audit.json').write_text(json.dumps(force_result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'phase':'force_completed','b0_anchor':force_rows[0],'b0_2s':force_rows[5]},allow_nan=False),flush=True)

chosen=[r for r in teacher['selected_records'] if r['damping'] in (0.,.2)]
ids=np.array([r['evaluation_index'] for r in chosen])
init=torch.tensor(a['features'][ids,0]); cc=torch.tensor(a['context'][ids]); action=torch.zeros(len(ids),1,dtype=torch.float64)
probe=load_readout(HERE/'frozen_training/readouts.pt','mlp')
steps=80
reference=a['features'][ids,:steps+1]
truth=a['truth'][ids,:steps+1]
specs=[('dg3',3,'dg',1.),('dg6',6,'dg',1.),('dg12',12,'dg',1.),('dg24',24,'dg',1.),
       ('rk4_64',64,'rk4',1.),('dg12_Dzero',12,'dg',0.),('dg12_D01',12,'dg',.1)]
results=[]; arrays={}
for name,substeps,kind,scale in specs:
    model=make_core(substeps,kind,scale)
    q,p=init[:,:2].clone(),init[:,2:].clone(); zs=[init.numpy()]
    start=time.perf_counter(); max_residual=0.; max_split=0.
    with torch.no_grad():
        for k in range(steps):
            q,p,info=model(q,p,action,.1,context=cc)
            zs.append(torch.cat((q,p),-1).numpy())
            max_residual=max(max_residual,float(info.get('solver_residual_ratio',torch.zeros(1)).max()))
            max_split=max(max_split,float(info.get('solver_split_depth',torch.zeros(1)).max()))
    zz=np.stack(zs,axis=1); arrays[name]=zz
    pred=decode_state(probe.predict(zz)); arrays[name+'_physical']=pred
    rows=[]
    for b in (0.,.2):
        mask=np.array([r['damping']==b for r in chosen])
        for t in (1,2,4,8):
            n=round(t/.1)
            rows.append(dict(damping=b,time_s=t,n=int(mask.sum()),
                endpoint=metrics(pred[mask,n],truth[mask,n]),
                cumulative=metrics(pred[mask,1:n+1],truth[mask,1:n+1]),
                q_displacement_rms=float(np.sqrt(np.mean((zz[mask,n,:2]-zz[mask,n-1,:2])**2))),
                p_rms=float(np.sqrt(np.mean(zz[mask,n,2:]**2))),
                feature_difference_from_dg3_rmse=float(np.sqrt(np.mean((zz[mask,:n+1]-reference[mask,:n+1])**2)))))
    unchanged=all(torch.equal(v,model.state_dict()[k]) for k,v in state.items())
    assert unchanged
    result=dict(name=name,substeps=substeps,integrator=kind,damping_multiplier=scale,elapsed_seconds=time.perf_counter()-start,
                weights_unchanged=unchanged,max_residual_ratio=max_residual,max_split_depth=max_split,
                original_feature_max_difference=float(np.max(abs(zz-reference))),metrics=rows)
    results.append(result)
    (HERE/'step_sensitivity_progress.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'phase':'variant_completed',**result}),flush=True)

for result in results:
    z=arrays[result['name']]
    result['against_rk4_64']=dict(q_rms=float(np.sqrt(np.mean((z[:,:,:2]-arrays['rk4_64'][:,:,:2])**2))),
                                p_rms=float(np.sqrt(np.mean((z[:,:,2:]-arrays['rk4_64'][:,:,2:])**2))))
final=dict(status='completed',checkpoint_sha256=sha,selection='One metadata-selected episode per damping0/.2 x4 energy bands; same as teacher audit',
           selected=chosen,step_size_scope='Same weights, latent initialization and context; strict tolerances unchanged. D multipliers are deliberate counterfactual dynamics, not test scores.',
           variants=results)
(HERE/'step_sensitivity.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
np.savez_compressed(HERE/'step_sensitivity_arrays.npz',**arrays,truth=truth,original=reference)
print(json.dumps({'phase':'completed','variants':len(results)}),flush=True)
