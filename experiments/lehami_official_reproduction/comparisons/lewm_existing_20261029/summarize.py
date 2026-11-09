"""Compare frozen transfer results; do not combine them with in-domain training claims."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
truth_file = np.load(ROOT/'truth.npz')
truth = truth_file['state'][:,4:75].astype(float)
lengths = truth_file['length']


def error_curve(pred, gt=truth):
    assert pred.shape == gt.shape and np.isfinite(pred).all()
    delta = pred - gt
    angle = np.abs(np.arctan2(np.sin(delta[...,0]),np.cos(delta[...,0])))
    omega = np.abs(delta[...,1])
    return np.stack([angle.mean(0)*180/np.pi, omega.mean(0)],-1)


def at_steps(curve):
    return [{'seconds':k/10,'angle_mae_deg':float(curve[k,0]),'omega_mae_rad_s':float(curve[k,1])}
            for k in [0,10,30,70]]


lewm = {mode:np.stack([np.load(ROOT/f'lewm_seed{s}.npz')[mode] for s in range(3)])
        for mode in ['mlp','ridge']}
lehami = dict(np.load(ROOT/'lehami_anchor4.npz'))
curves = {mode:np.stack([error_curve(p) for p in preds]) for mode,preds in lewm.items()}
hcurves = {mode:error_curve(p) for mode,p in lehami.items()}

# Re-run at the original anchor to validate the new inference path against the
# already-completed original evaluator; then compare both models at frame4.
check = error_curve(np.load(ROOT/'lehami_anchor3.npz')['knn_k1'], truth_file['state'][:,3:74])
original_path = PROJECT/'received/completed_47141/results/exp067_a0m_lehami_w1_multi/probe_metrics.csv'
original_rows = sorted([r for r in csv.DictReader(original_path.open()) if r['split']=='test'],key=lambda r:int(r['k']))
original = np.array([[float(r['mae_theta__knn_k1'])*180/np.pi,float(r['mae_omega__knn_k1'])] for r in original_rows])
diff = np.abs(check-original).max(0)
assert diff[0]<.1 and diff[1]<.01, diff

summary = dict(status='PASS', protocol=json.loads((ROOT/'protocol.json').read_text()),
               anchor3_validation_max_diff=dict(angle_mae_deg=float(diff[0]),omega_mae_rad_s=float(diff[1])),
               lewm={},lehami={k:at_steps(v) for k,v in hcurves.items()}, per_length={})
summary['execution_validation'] = {
    'all_model_weights_unchanged': all(json.loads((ROOT/f'lewm_seed{s}.json').read_text())['weights_unchanged'] for s in range(3)) and json.loads((ROOT/'lehami.json').read_text())['weights_unchanged'],
    'readouts_refitted': False,
    'lewm_batch_size': 1,
    'lewm_batch_note': 'Batch8 failed the decoded-trajectory equivalence gate; final results use original episode-by-episode inference.',
    'lehami_encoder_batch': 50,
    'lehami_encoder_padding': 'Only observed frames are repeated to match the original encoder batch shape. Changed padding was verified to leave observed latents bitwise unchanged.',
    'numerical_sensitivity': 'Encoding only 4/5 frames changes floating-point arithmetic; small initial differences amplified in long rollouts. Matching the original batch50 shape reduced the validation discrepancy to the recorded limits.',
}
for mode,values in curves.items():
    summary['lewm'][mode] = dict(seed_mean=at_steps(values.mean(0)),seed_sample_std=at_steps(values.std(0,ddof=1)),
                               each_seed={str(s):at_steps(v) for s,v in enumerate(values)})
for length in np.unique(lengths):
    mask = lengths==length
    vals = np.stack([error_curve(p[mask],truth[mask]) for p in lewm['mlp']])
    summary['per_length'][str(float(length))] = dict(episodes=int(mask.sum()),
        lewm_mlp_seed_mean=at_steps(vals.mean(0)),lehami_knn_k1=at_steps(error_curve(lehami['knn_k1'][mask],truth[mask])))

reference = PROJECT/'upstream/LeHami/paper_figures/data/conservative_pendulum/probe_metrics'
summary['author_reference'] = {}
for model,stem in [('LeHami','exp067_a0m_lehami_w1_multi'),('LeWM','exp068_b0m_lewm_w1_multi')]:
    rows = sorted([r for r in csv.DictReader((reference/f'{stem}_s0.csv').open()) if r['split']=='test'],key=lambda r:int(r['k']))
    c = np.array([[float(r['mae_theta__knn_k1'])*180/np.pi,float(r['mae_omega__knn_k1'])] for r in rows])
    summary['author_reference'][model] = dict(seed=42,readout='knn_k1',source='published CSV, not a newly evaluated checkpoint',metrics=at_steps(c))

(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'curves.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['k','time_s','model','readout','seed','angle_mae_deg','omega_mae_rad_s'])
    for mode,v in curves.items():
        for s,c in enumerate(v):
            for k,row in enumerate(c):w.writerow([k,k/10,'LeWM_old',mode,s,*row])
    for mode,c in hcurves.items():
        for k,row in enumerate(c):w.writerow([k,k/10,'LeHami_official',mode,42,*row])

fig,axes = plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
t=np.arange(71)/10
for index,ax in enumerate(axes):
    mean,std=curves['mlp'][:,:,index].mean(0),curves['mlp'][:,:,index].std(0,ddof=1)
    ax.plot(t,mean,color='#ce7134',label='Old LeWM / frozen MLP (3 seeds)')
    ax.fill_between(t,mean-std,mean+std,color='#ce7134',alpha=.16,label='LeWM seed mean +/- SD')
    ax.plot(t,hcurves['mlp_gt'][:,index],color='#2166ac',label='LeHami / frozen MLP (seed 42)')
    ax.plot(t,hcurves['knn_k1'][:,index],color='#2166ac',linestyle='--',alpha=.7,label='LeHami / paper knn_k1 (seed 42)')
    ax.set(xlabel='Forecast time (s)',ylabel=['Wrapped angle MAE (degrees)','Angular velocity MAE (rad/s)'][index],xlim=(0,7))
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
axes[0].legend(fontsize=8)
fig.suptitle('Same 45 test episodes, aligned start | Different training data and readouts\nTransfer diagnostic: no model training or probe refitting',fontsize=11)
fig.savefig(ROOT/'comparison.png',dpi=180)
print(json.dumps({k:summary[k] for k in ['anchor3_validation_max_diff','lewm','lehami','author_reference']},indent=2))
