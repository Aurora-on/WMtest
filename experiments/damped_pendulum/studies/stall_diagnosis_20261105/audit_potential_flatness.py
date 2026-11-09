"""Decompose learned potential slopes on true-image and autonomous states."""
from pathlib import Path
import sys,json,hashlib
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from dpbench.efficient_dissipative_core import EfficientDissipativeCore
torch.set_num_threads(2)
S=Path(__file__).resolve().parent; E=ROOT/'studies/efficient3_evaluation_20261105T023318Z'
checkpoint=S/'frozen_training/best.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
c=ck['config'];core=EfficientDissipativeCore(q_dim=c['q_dim'],context_dim=c['environment_context_dim'],**c['core'])
core.load_state_dict({k.removeprefix('core.'):v for k,v in ck['state_dict'].items() if k.startswith('core.')})
rawnet=core.potential_net[0]
a=np.load(E/'received/trajectories.npz');episodes=json.loads((E/'received/episodes.json').read_text())
t=np.load(E/'teacher_received/diagnostic_arrays.npz');td=json.loads((E/'teacher_received/diagnostic.json').read_text())
teacher_records=td['selected_records']

def summary(x):
    x=x.detach().cpu().numpy().ravel();return dict(mean=float(x.mean()),median=float(np.median(x)),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),minimum=float(x.min()),maximum=float(x.max()))
def analyze(data):
    q=torch.tensor(data,dtype=torch.float64,requires_grad=True)
    z1=rawnet[0](q);a1=z1.tanh();z2=rawnet[2](a1);a2=z2.tanh();r=rawnet[4](a2).squeeze(-1)
    v=torch.nn.functional.softplus(r)
    gr=torch.autograd.grad(r.sum(),q,retain_graph=True)[0]
    gv=torch.autograd.grad(v.sum(),q)[0]
    sigmoid=r.sigmoid()
    return dict(n=len(q),raw_output=summary(r),softplus_derivative=summary(sigmoid),
      raw_force_norm=summary(gr.norm(dim=-1)),final_force_norm=summary(gv.norm(dim=-1)),
      raw_to_final_identity_error=float((gv-sigmoid[:,None]*gr).abs().max().detach()),
      q=summary(q), layer1_tanh_derivative=summary(1-a1.square()),layer2_tanh_derivative=summary(1-a2.square()),
      layer1_fraction_slope_below_1e3=float(((1-a1.square())<1e-3).double().mean().detach()),
      layer2_fraction_slope_below_1e3=float(((1-a2.square())<1e-3).double().mean().detach()),
      softplus_fraction_slope_below_1e3=float((sigmoid<1e-3).double().mean().detach()))

result={'scope':'Frozen potential diagnostic; no parameter changes, fitting, or training',
        'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'last_linear_bias':float(rawnet[4].bias.detach()),
        'last_linear_weight_l1':float(rawnet[4].weight.abs().sum().detach()),'rows':[]}
for b in (0.,.2):
    ids=[i for i,e in enumerate(episodes) if e['damping']==b]
    tids=[i for i,e in enumerate(teacher_records) if e['damping']==b]
    for sec in (0.,1.,2.,8.,32.):
        r=dict(source='autonomous_all',damping=b,time_s=sec,**analyze(a['features'][ids,int(round(sec*10)),:2]));result['rows'].append(r)
        ti=td['relative_times_seconds'].index(sec)
        result['rows'].append(dict(source='true_image_selected',damping=b,time_s=sec,**analyze(t['teacher_features'][tids,ti,:2])))
qmin=a['features'][..., :2].reshape(-1,2).min(0);qmax=a['features'][..., :2].reshape(-1,2).max(0)
xx,yy=np.meshgrid(np.linspace(qmin[0],qmax[0],100),np.linspace(qmin[1],qmax[1],100));grid=np.stack((xx.ravel(),yy.ravel()),1)
result['observed_latent_bbox']={'min':qmin.tolist(),'max':qmax.tolist(),'uniform_grid_n':len(grid),'diagnostics':analyze(grid)}
(S/'potential_flatness_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'bias':result['last_linear_bias'],'weight_l1':result['last_linear_weight_l1'],
  'b0_8sec':next(r for r in result['rows'] if r['source']=='autonomous_all' and r['damping']==0 and r['time_s']==8)},indent=2))
