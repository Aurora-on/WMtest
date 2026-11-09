"""Plot frozen interventions; angular wrap is broken, never differentiated."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'stall_diagnosis_20261105'
summary = json.loads((HERE / 'damping_ablation_recheck.json').read_text())
previous = json.loads((SOURCE / 'step_sensitivity.json').read_text())
data = np.load(SOURCE / 'step_sensitivity_arrays.npz')
representative = next(i for i, r in enumerate(previous['selected'])
                      if r['damping'] == 0 and r['energy_band'] == 2)
record = previous['selected'][representative]
variants = ['dg12', 'dg12_D01', 'dg12_Dzero']
labels = ['Original D', '0.1 D', 'D = 0']
colors = ['#d65343', '#277da8', '#9774b6']
t = np.arange(81) * .1

def principal_curve(value):
    y = np.arctan2(np.sin(value), np.cos(value))
    y = y.copy()
    y[np.r_[False, abs(np.diff(y)) > np.pi]] = np.nan
    return y

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.labelcolor': '#333333', 'text.color': '#222222'})
fig = plt.figure(figsize=(11.3, 7.0), constrained_layout=False)
gs = fig.add_gridspec(2, 2, left=.085, right=.97, bottom=.17, top=.85,
                      hspace=.52, wspace=.3, height_ratios=[1.25, 1])
ax = fig.add_subplot(gs[0, :])
ax.plot(t, principal_curve(data['truth'][representative, :, 0]),
        color='#222222', linewidth=2.2, label='True undamped motion')
for name, label, color in zip(variants, labels, colors):
    ax.plot(t, principal_curve(data[name + '_physical'][representative, :, 0]),
            color=color, linewidth=1.7, label=label)
ax.set(xlim=(0, 8), ylim=(-np.pi, np.pi), xlabel='Prediction time (s)',
       ylabel='Angle (rad)', yticks=[-np.pi, 0, np.pi],
       yticklabels=['−π', '0', 'π'])
ax.set_title(f'Example chosen by metadata: energy band 2, evaluation index {record["evaluation_index"]}',
             loc='left', fontsize=11)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.14), ncol=4, frameon=False,
          handlelength=2.4, columnspacing=2.0)
ax.grid(alpha=.16)

metrics = {r['variant']: r for r in summary['variants']}
ax1 = fig.add_subplot(gs[1, 0])
v = [metrics[name]['cumulative_angle_rmse_rad'] for name in variants]
ax1.bar(labels, v, color=colors, width=.58)
ax1.set(ylabel='Angular RMSE (rad)', ylim=(0, 2.05))
ax1.set_title('Trajectory error over 0.1–8 s · 4 episodes', loc='left', fontsize=11)
ax1.grid(axis='y', alpha=.16)
for i, value in enumerate(v):
    ax1.text(i, value + .055, f'{value:.3f}', ha='center', fontsize=11)

ax2 = fig.add_subplot(gs[1, 1])
v = [metrics[name]['final_p_rms'] for name in variants]
ax2.bar(labels, v, color=colors, width=.58)
ax2.set(ylabel='Latent momentum RMS', yscale='log', ylim=(.001, 20))
ax2.set_title('Motion remaining at 8 s · same 4 episodes', loc='left', fontsize=11)
ax2.grid(axis='y', alpha=.16)
for i, value in enumerate(v):
    ax2.text(i, value * 1.25, f'{value:.4g}', ha='center', fontsize=11)

fig.suptitle('Reducing damping restores latent movement, but not correct prediction',
             x=.085, y=.98, ha='left', fontsize=15, fontweight='bold')
fig.text(.085, .94, 'Frozen checkpoint 19250 · same DG12 integration, initialization, potential and readout · no training',
         fontsize=10, color='#555555')
fig.text(.085, .065,
         'RMSE: wrapped angular differences pooled equally over 4 episodes × 80 future frames; initial frame excluded.\n'
         'D is a latent damping matrix, not the physical b. Interventions are diagnostic; angles break at ±π.',
         fontsize=9, color='#555555', linespacing=1.65)
fig.savefig(HERE / 'damping_ablation.png', dpi=180, facecolor='white')
plt.close(fig)
(HERE / 'damping_ablation_plot_provenance.json').write_text(json.dumps({
    'representative': record,
    'selection': 'Predeclared medium-high energy band 2 from existing metadata-selected sample; no error-based selection.',
    'metric': 'sqrt(mean(wrapped_angle_error**2)) pooled equally over4episodes x80futureframes, t=.1..8.0s. Not an average of per-episodeRMSE and not endpoint-only.',
    'angle_plot': 'arctan2(sin(theta),cos(theta)); remove connecting segment when consecutive principal angle difference exceeds pi.',
    'motion_metric': 'RMS of final two latent momentum coordinates, pooled across4episodes; not finite difference of wrapped angle.',
    'no_model_or_probe_fitting': True,
}, indent=2) + '\n')
