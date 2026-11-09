"""Build report facts and plots from completed, immutable evaluations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
FIG = ROOT / 'tmp/pdfs'
FIG.mkdir(parents=True, exist_ok=True)
read = lambda name: json.loads((STUDY / name).read_text())
metrics = read('received/metrics.json')
manifest = read('received/manifest.json')
legacy = read('lewm_received/summary.json')
legacy_manifest = read('lewm_received/manifest.json')
diagnostic = read('teacher_received/diagnostic.json')
analysis = read('analysis_agent/prediction_diagnostics.json')
episodes = read('received/episodes.json')
arrays = np.load(STUDY / 'received/trajectories.npz')
assert manifest['status'] == legacy_manifest['status'] == diagnostic['status'] == 'completed'
assert manifest['completed_full_rollouts'] == 1408 and manifest['failed_rollouts'] == 0
assert legacy_manifest['completed_model_episode_rollouts'] == 768
assert diagnostic['zero_time_max_feature_difference'] == 0

def metric(method, horizon, **group):
    return next(x for x in metrics if x['method'] == method and x['horizon_s'] == horizon
                and x['support'] == 'all_available' and all(x.get(k) == v for k, v in group.items()))

facts = {'manifest': manifest, 'lewm_manifest': legacy_manifest,
         'main_table': [], 'lewm_table': [], 'teacher': [],
         'b0_timepoints': analysis['by_damping']['0.0']['timepoints'],
         'diagnostic': {k: analysis[k] for k in ('decoded_kinematic_defect_rmse',
                                               'reference_kinematic_defect_rmse',
                                               'physical_energy_increase_fraction')}}
for h in (1,4,8,32):
    facts['main_table'].append({'horizon_s': h, **{
        method: metric(method, h, grouping='all')
        for method in ('model', 'fixed_frame', 'equilibrium')}})
for x in legacy['aggregate']:
    if x['primary'] and x['horizon_s'] in (8,32):
        facts['lewm_table'].append({'damping': x['damping'], 'horizon_s': x['horizon_s'],
                                    'ours': metric('model', x['horizon_s'], grouping='damping', damping=x['damping']),
                                    'lewm_mean': x['mean'], 'lewm_sample_sd': x['seed_sd']})
facts['teacher'] = [x for x in diagnostic['metrics'] if x['group'] == 'all']
facts['teacher_scope'] = diagnostic['scope']
facts['teacher_selection'] = diagnostic['selection']

b0 = np.array([e['damping'] == 0 for e in episodes])
def energy(x):
    return 0.5 * x[...,1]**2 + 9.81*(1-np.cos(x[...,0]))

pred = arrays['prediction_model']
truth = arrays['truth']
time = arrays['time']
e0 = arrays['initial_energy']
ep = min((e for e in episodes if e['damping'] == 0 and e['energy_band'] == 2),
         key=lambda e: (e['group_id'], e['episode']))
idx = episodes.index(ep)
legacy_ep = read('lewm_received/lewm_b0p0_seed0/episodes.json')
legacy_arrays = np.load(STUDY / 'lewm_received/lewm_b0p0_seed0/trajectories.npz')
legacy_idx = next(i for i, e in enumerate(legacy_ep) if e['episode'] == ep['episode'])
# Prediction keys are recorded by the reference evaluator, with the original MLP frozen.
legacy_prediction = legacy_arrays['prediction_mlp']
np.testing.assert_allclose(legacy_arrays['truth'][legacy_idx], truth[idx], atol=0, rtol=0)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(1,2,figsize=(9.6,2.9),layout='constrained')
axs[0].plot(time[:41], (energy(truth[b0])/e0[b0,None]).mean(0)[:41], color='#172a3b',label='Truth')
axs[0].plot(time[:41], (energy(pred[b0])/e0[b0,None]).mean(0)[:41], color='#bd5136',label='Our model')
axs[0].set(xlabel='Prediction time (s)',ylabel='Energy / E0 (mean)',ylim=(0,1.08),title='b = 0: mean over 128 trajectories')
axs[0].legend(frameon=False, fontsize=9)
axs[1].plot(time[:81],truth[idx,:81,0],color='#172a3b',label='Truth',lw=1.8)
axs[1].plot(time[:81],pred[idx,:81,0],color='#bd5136',label='Our model',lw=1.8)
axs[1].plot(time[:81],legacy_prediction[legacy_idx,:81,0],color='#138b89',label='Historical LeWM, seed 0',lw=1.4,linestyle='--')
axs[1].set(xlabel='Prediction time (s)',ylabel='Angle (rad)',title='b = 0: fixed example, no error-based selection')
axs[1].legend(frameon=False, fontsize=8,loc='upper right')
for ax in axs:
    ax.grid(alpha=.15)
fig.savefig(FIG/'physical_failure.png',dpi=200)
plt.close(fig)
facts['figure'] = {'example': ep, 'selection': 'Minimum group_id/episode in damping0, energy_band2; LeWM seed0 fixed',
                   'mean_energy_samples': int(b0.sum()), 'energy_denominator': 'Each episode physical initial energy at dataset time0',
                   'prediction_anchor_time_s': 2.0}
(STUDY/'report_facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'facts':str(STUDY/'report_facts.json'),'figure':str(FIG/'physical_failure.png')}))
