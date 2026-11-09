"""Read-only final-checkpoint gradient audit. No optimizer and no model edits."""
from pathlib import Path
import sys, json, hashlib, time
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dpbench.efficient_dissipative_core import EfficientDissipativeCore
from dpbench.automatic_discrete_gradient import potential_value_and_gradient

torch.set_num_threads(2)
START=time.monotonic()
study=Path(__file__).resolve().parent
checkpoint=study/'frozen_training/best.pt'
blob=checkpoint.read_bytes()
assert hashlib.sha256(blob).hexdigest()=='5163a6a293a7c773a90e9ef59a62353a4debdba64c0dc24814bcbdb3b47f9042'
ck=torch.load(checkpoint, map_location='cpu', weights_only=False)
cfg=ck['config']
core=EfficientDissipativeCore(q_dim=cfg['q_dim'], context_dim=cfg['environment_context_dim'], **cfg['core'])
core.load_state_dict({k.removeprefix('core.'):v for k,v in ck['state_dict'].items() if k.startswith('core.')})
original={k:v.clone() for k,v in core.state_dict().items()}
arrays=np.load(ROOT/'studies/efficient3_evaluation_20261105T023318Z/received/trajectories.npz')
ep=json.loads((ROOT/'studies/efficient3_evaluation_20261105T023318Z/received/episodes.json').read_text())
ids=[]
for b,band in [(0.,0),(0.,2),(.2,0),(.2,2)]:
    candidates=[i for i,e in enumerate(ep) if e['damping']==b and e['energy_band']==band]
    ids.append(min(candidates, key=lambda i:(ep[i]['group_id'],ep[i]['episode'])))

output={'scope':'Finite differences of frozen 3-substep core, no optimizer/model/readout fitting',
 'checkpoint_sha256':hashlib.sha256(blob).hexdigest(), 'checkpoint_step':ck['step'],
 'selected_episodes':[ep[i] for i in ids], 'selection':'Minimum group_id in b0/b.2 x energy bands0/2, not errors',
 'existing_test_result':'36 passed in 2.25s; tests/test_efficient_dissipative_core.py and test_automatic_discrete_gradient.py',
 'tolerance_scope':'Forward tolerance tightened only for derivative diagnostics to separate root error from FD error',
 'runs':[], 'explicit_potential_gradient':[]}

for at in (0,20):
    features=torch.from_numpy(arrays['features'][ids,at]).double()
    q,p=features[:,:2].clone().requires_grad_(),features[:,2:].clone().requires_grad_()
    c=torch.from_numpy(arrays['context'][ids]).double().requires_grad_()
    dt=torch.full((len(ids),1), .1,dtype=torch.float64,requires_grad=True)
    g=potential_value_and_gradient(core.potential_net,q)[1]
    reference=torch.autograd.grad(core.potential(q).sum(), q, create_graph=True)[0]
    hess1=torch.autograd.grad(g.sum(),q,retain_graph=True)[0]
    hess2=torch.autograd.grad(reference.sum(),q)[0]
    output['explicit_potential_gradient'].append({'at_s':at*.1, 'gradient_max_abs_error':float((g-reference).abs().max()),
       'hessian_contraction_max_abs_error':float((hess1-hess2).abs().max())})

    core.residual_atol=core.residual_rtol=1e-11
    targets={'q':q,'p':p,'context':c,'dt':dt,
             'damping_first_weight':core.damping_net[0].weight,
             'damping_last_weight':core.damping_net[-1].weight,
             'potential_first_weight':core.potential_net[0][0].weight,
             'potential_last_weight':core.potential_net[0][-1].weight}
    action=torch.zeros(len(ids),1,dtype=torch.float64)
    def objective(steps):
        qn,pn=q,p
        value=q.new_zeros(())
        max_split=0
        for k in range(steps):
            qn,pn,info=core(qn,pn,action,dt,context=c)
            # Smooth q-only objective exercises the loss path used in training.
            value=value+((qn*torch.tensor([.3,-.7])).sum(-1)+.1*qn.square().sum(-1)).mean()/steps
            max_split=max(max_split,float(info['solver_split_depth'].max()))
        return value,max_split
    for steps in (1,5,20):
        val,split=objective(steps)
        gradients=torch.autograd.grad(val,tuple(targets.values()),allow_unused=True)
        run={'at_s':at*.1,'steps':steps,'objective':float(val.detach()),'max_split_depth':split,'targets':[]}
        for (name,target),grad in zip(targets.items(),gradients):
            assert grad is not None and torch.isfinite(grad).all(),name
            direction=torch.linspace(.19,.91,target.numel(),dtype=torch.float64).reshape(target.shape)
            direction/=direction.norm()
            wanted=float((grad*direction).sum())
            initial=target.detach().clone()
            target_result={'name':name,'gradient_norm':float(grad.norm()),'directional_derivative':wanted,'finite_differences':[]}
            for eps in (2e-5,5e-6):
                values=[]
                splits=[]
                try:
                    with torch.no_grad():
                        for sign in (1,-1):
                            target.copy_(initial+sign*eps*direction)
                            perturbed,branch=objective(steps)
                            values.append(float(perturbed));splits.append(branch)
                finally:
                    with torch.no_grad(): target.copy_(initial)
                got=(values[0]-values[1])/(2*eps)
                error=abs(got-wanted)
                relative=error/max(abs(got),abs(wanted),1e-9)
                target_result['finite_differences'].append({'epsilon':eps,'numerical':got,'absolute_error':error,
                    'relative_error':relative,'same_subdivision':all(s==split for s in splits),
                    'passed':error <= 1e-6+1e-3*max(abs(got),abs(wanted)) and all(s==split for s in splits)})
            run['targets'].append(target_result)
        output['runs'].append(run)
        print(json.dumps({'at_s':at*.1,'steps':steps,'all_passed':all(fd['passed'] for t in run['targets'] for fd in t['finite_differences'])}),flush=True)

output['all_finite_difference_checks_passed']=all(fd['passed'] for run in output['runs'] for t in run['targets'] for fd in t['finite_differences'])
output['max_absolute_fd_error']=max(fd['absolute_error'] for run in output['runs'] for t in run['targets'] for fd in t['finite_differences'])
output['max_relative_fd_error']=max(fd['relative_error'] for run in output['runs'] for t in run['targets'] for fd in t['finite_differences'])
output['parameters_unchanged']=all(torch.equal(original[k],v) for k,v in core.state_dict().items())
assert output['parameters_unchanged']
output['elapsed_seconds']=time.monotonic()-START
(study/'final_core_gradient_audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k not in ('runs','selected_episodes','explicit_potential_gradient')}))
