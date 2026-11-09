"""Check the fine RK4 reference without repeating other diagnostic variants."""
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from dpbench.dissipative_core import DissipativeCore
from dpbench.train_dissipative import source_hashes
torch.set_num_threads(2)
saved=torch.load(HERE/'frozen_training/best.pt',map_location='cpu',weights_only=True)
assert source_hashes()==saved['source_hashes']
cfg=saved['config']['core'].copy()
for k in ('picard_iterations','picard_check_every','fd_relative_step','implicit_rcond_min','adjoint_rtol'):cfg.pop(k,None)
cfg.update(integrator='rk4',solver='full',max_bisections=0,substeps=128)
core=DissipativeCore(q_dim=2,context_dim=4,**cfg)
state={k[5:]:v for k,v in saved['state_dict'].items() if k.startswith('core.')};core.load_state_dict(state)
for p in core.parameters():p.requires_grad_(False)
sa=np.load(HERE/'step_sensitivity_arrays.npz');diag=json.loads((HERE/'step_sensitivity.json').read_text())
indices=[r['evaluation_index'] for r in diag['selected']]
base=np.load(ROOT/'studies/efficient3_evaluation_20261105T023318Z/received/trajectories.npz')
context=torch.tensor(base['context'][indices]);z=torch.tensor(sa['original'][:,0]);q,p=z[:,:2],z[:,2:]
zz=[z.numpy()];action=torch.zeros(len(q),1,dtype=torch.float64);start=time.perf_counter()
with torch.no_grad():
 for i in range(80):
  q,p,_=core(q,p,action,.1,context=context);zz.append(torch.cat((q,p),-1).numpy())
out=np.stack(zz,1);ref=sa['rk4_64'];delta=out-ref
result={'status':'completed','weights_unchanged':all(torch.equal(v,core.state_dict()[k]) for k,v in state.items()),
        'frozen_training_source_hashes_match':True,'episodes':len(indices),'seconds':time.perf_counter()-start,
        'comparison':'RK4 64 vs128substeps per0.1s frame, fixed learned functions,8s',
        'max_abs_feature_difference':float(abs(delta).max()),'feature_rmse':float(np.sqrt(np.mean(delta**2))),
        'final_p_rms_64':float(np.sqrt(np.mean(ref[:,-1,2:]**2))),'final_p_rms_128':float(np.sqrt(np.mean(out[:,-1,2:]**2)))}
(HERE/'rk4_reference_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
