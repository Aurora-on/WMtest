"""Reconstruct physical training targets to audit sampling; no images or fitting."""
import json
from pathlib import Path
import numpy as np
from dpbench.passive_family import sample_groups
from dpbench.generate import simulate, energy

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
cfg = json.loads((root/'configs/passive_family.json').read_text())
groups = sample_groups(cfg)
records=[]
saved={}
bins=np.array_split(np.arange(29),3)
def weighted_stats(x,initial_energy):
    e=energy(x,cfg['mass'],cfg['length'],cfg['gravity'])
    return {'angle_rms':float(np.sqrt(np.mean(x[...,0]**2))),
            'omega_rms':float(np.sqrt(np.mean(x[...,1]**2))),
            'rest_fraction_theta005_omega01':float(np.mean((np.abs(x[...,0])<.05)&(np.abs(x[...,1])<.1))),
            'low_energy_fraction_E001':float(np.mean(e<.01*initial_energy)),
            'mean_energy_fraction':float(np.mean(e/initial_energy))}
for b in cfg['seen_damping']:
    initial=groups['train']['initial']
    states,_,_=simulate(initial,np.zeros((len(initial),80)),cfg['dt'],cfg['substeps'],cfg['mass'],cfg['length'],cfg['gravity'],b)
    saved[f'state_b{b:g}']=states
    initial_energy=energy(initial,cfg['mass'],cfg['length'],cfg['gravity'])
    for h in (4,8,16,32):
        br=[]
        for time_bin,starts in enumerate(bins):
            anchor=states[:,starts+20]
            future=states[:,starts[:,None]+20+np.arange(1,h+1)]
            br.append({'time_bin':time_bin,'anchor_times_seconds':[(20+starts.min())*.1,(20+starts.max())*.1],
                       'anchor':weighted_stats(anchor,initial_energy[:,None]),
                       'target':weighted_stats(future,initial_energy[:,None,None])})
        records.append({'damping':b,'horizon':h,'bins':br,
                        'target_balanced':{k:float(np.mean([r['target'][k] for r in br])) for k in br[0]['target']}})

# Verify reconstruction against existing test data, never use it for fit/selection.
study=root/'studies/efficient3_evaluation_20261105T023318Z/received'
ep=json.loads((study/'episodes.json').read_text())
a=np.load(study/'trajectories.npz')
initial=groups['test']['initial']
test,_,_=simulate(initial,np.zeros((len(initial),340)),cfg['dt'],cfg['substeps'],cfg['mass'],cfg['length'],cfg['gravity'],0.)
ii=[i for i,e in enumerate(ep) if e['damping']==0]
diff=float(np.max(np.abs(test[:,20:]-a['truth'][ii])))
assert diff<1e-9,diff
result={'scope':'Posthoc physical sampling audit. Reconstructed exact generator states; no training or labels supplied to model.',
        'test_reconstruction_max_difference':diff,'train_initial_groups':len(groups['train']['initial']),
        'anchor_times_seconds':[2.,4.8],'future_times_seconds_max_horizon':[2.1,8.],
        'sampling':'Equal regime and time-bin; uniform initial group and start inside each bin; average metrics across bins.',
        'records':records}
(out/'training_distribution_audit.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(out/'reconstructed_training_states.npz',**saved,initial=groups['train']['initial'],energy_band=groups['train']['energy_band'])
for row in records:
    if row['horizon']==32:print(row['damping'],row['target_balanced'])
print('reconstruction matches existing test truth within 1e-9:',diff)
