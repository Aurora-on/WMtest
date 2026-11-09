"""Read-only posthoc quantification of latent stall and renderer discontinuity.

No world-model fitting, parameter modification, or production-code changes.
The scale-normalized stall criterion is a sensitivity diagnostic, not a
predeclared benchmark metric or a universal physical threshold.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from dpbench.generate import render

SRC = ROOT / 'studies/efficient3_evaluation_20261105T023318Z'
read = lambda name: json.loads((SRC / name).read_text())
arrays = np.load(SRC / 'received/trajectories.npz')
teacher = np.load(SRC / 'teacher_received/diagnostic_arrays.npz')
teacher_meta = read('teacher_received/diagnostic.json')
records = read('received/episodes.json')
assert np.all(arrays['valid_steps'] == 320)
b = np.array([r['damping'] for r in records])
q = arrays['features'][...,:2]
p = arrays['features'][...,2:]
H = arrays['latent_energy']
K = .5 * np.sum(p*p, axis=-1)
V = H-K
truth = arrays['truth']
pred = arrays['prediction_model']
times = arrays['time']
dt = float(times[1]-times[0])
assert np.allclose(dt, .1)
dq = np.diff(q,axis=1)
dp = np.diff(p,axis=1)
q_scale = float(np.sqrt(np.mean(np.var(q[:,0],axis=0))))
p_scale = float(np.sqrt(np.mean(p[:,0]**2)))
rms = lambda x: float(np.sqrt(np.mean(np.asarray(x)**2)))
wrap = lambda x: np.arctan2(np.sin(x),np.cos(x))

def summarize(mask):
    points=[]
    for t in [0,.1,.2,.5,1,2,4,8,16,32]:
        k=round(t/dt)
        # Use a true neighboring interval even at the terminal time.
        delta = dq[mask,min(k,len(times)-2)] / dt
        points.append(dict(time_s=t, n=int(mask.sum()),
          latent_q_rms=rms(q[mask,k]),latent_p_rms=rms(p[mask,k]),
          latent_q_interval_velocity_rms=rms(delta),
          q_velocity_interval='forward' if k<len(times)-1 else 'backward',
          mean_H=float(H[mask,k].mean()),mean_K=float(K[mask,k].mean()),mean_V=float(V[mask,k].mean()),
          mean_individual_H_fraction=float(np.mean(H[mask,k]/H[mask,0])),
          reference_omega_rms=rms(truth[mask,k,1]),decoded_omega_rms=rms(pred[mask,k,1])))
    windows=[]
    for t in [1,2,4,8,16,31]:
        k=round(t/dt)
        qw=q[mask,k:k+11];pw=p[mask,k:k+11]
        tw=truth[mask,k:k+11];yw=pred[mask,k:k+11]
        q_excursion=np.max(np.sqrt(np.mean((qw-qw[:,:1])**2,axis=-1)),axis=1)/q_scale
        p_activity=np.sqrt(np.mean(pw**2,axis=(1,2)))/p_scale
        true_speed=np.sqrt(np.mean(tw[...,1]**2,axis=1))
        true_angle_excursion=np.max(np.abs(wrap(tw[...,0]-tw[:,:1,0])),axis=1)
        moving=(true_speed>.1)&(true_angle_excursion>.01)
        pred_angle_excursion=np.max(np.abs(wrap(yw[...,0]-yw[:,:1,0])),axis=1)
        entry=dict(window_s=[t,t+1], n=int(mask.sum()),
            reference_active_count=int(moving.sum()),
            q_excursion_normalized_median=float(np.median(q_excursion)),
            p_activity_normalized_median=float(np.median(p_activity)),
            predicted_angle_excursion_median_rad=float(np.median(pred_angle_excursion)),
            reference_angle_excursion_median_rad=float(np.median(true_angle_excursion)),
            stall_sensitivity=[])
        for eps in [.001,.01,.05]:
            stalled=(q_excursion<eps)&(p_activity<eps)
            entry['stall_sensitivity'].append(dict(epsilon=eps,
                stalled_count=int(stalled.sum()), stalled_fraction=float(stalled.mean()),
                stalled_but_reference_active_count=int((stalled&moving).sum()),
                decoded_omega_rms_on_stalled=rms(yw[stalled,...,1]) if stalled.any() else None))
        windows.append(entry)
    return dict(timepoints=points, windows=windows)

groups={'all':summarize(np.ones(len(b),dtype=bool))}
for value in np.unique(b):
    groups[f'b={value:g}']=summarize(b==value)

teacher_groups={}
tb=np.array([r['damping'] for r in teacher_meta['selected_records']])
for name, mask in [('all',np.ones(len(tb),dtype=bool))]+[(f'b={v:g}',tb==v) for v in np.unique(tb)]:
    rows=[]
    for k,t in enumerate(teacher['relative_time']):
        tf=teacher['teacher_features'][mask,k]
        af=teacher['autonomous_features'][mask,k]
        rows.append(dict(time_s=float(t),n=int(mask.sum()),
            teacher_q_rms=rms(tf[...,:2]),autonomous_q_rms=rms(af[...,:2]),
            teacher_p_rms=rms(tf[...,2:]),autonomous_p_rms=rms(af[...,2:]),
            q_difference_rmse=rms(tf[...,:2]-af[...,:2]),p_difference_rmse=rms(tf[...,2:]-af[...,2:]),
            reference_omega_rms=rms(teacher['truth'][mask,k,1]),
            teacher_decoded_omega_rms=rms(teacher['teacher_prediction'][mask,k,1]),
            autonomous_decoded_omega_rms=rms(teacher['autonomous_prediction'][mask,k,1])))
    teacher_groups[name]=rows

# Rendering a nearly stationary oscillation still switches pixels at theta=0.
zero = render(0)
renderer_rows=[]
for eps in [1e-12,1e-10,1e-8,1e-5,1e-3]:
    minus,plus=render(-eps),render(eps)
    diff=plus.astype(np.int16)-minus.astype(np.int16)
    renderer_rows.append(dict(epsilon_rad=eps,
        positive_equals_zero=bool(np.array_equal(plus,zero)),
        negative_equals_zero=bool(np.array_equal(minus,zero)),
        changed_rgb_components=int(np.count_nonzero(diff)),
        changed_pixels=int(np.any(diff!=0,axis=-1).sum()),
        absolute_rgb_difference_sum=int(np.abs(diff).sum()),
        normalized_frame_difference_rmse=rms(diff/255)))

def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
results=dict(scope='Posthoc observation only; no optimization or checkpoint changes',
    sources={str(p.relative_to(ROOT)):sha256(p) for p in [SRC/'received/trajectories.npz',SRC/'received/episodes.json',
        SRC/'teacher_received/diagnostic_arrays.npz',ROOT/'dpbench/generate.py']},
    definition=dict(q_scale=q_scale,p_scale=p_scale,
        q_scale_definition='sqrt(mean over q dimensions of variance across all test anchor q))',
        p_scale_definition='RMS across dimensions and all test anchor p',
        q_excursion='maximum over following 1s of dimension-RMS(q(tau)-q(t))/q_scale',
        p_activity='RMS over following 1s and dimensions / p_scale',
        stalled='q_excursion < epsilon AND p_activity < epsilon',
        reference_active='same 1s window: omega RMS >0.1rad/s AND max circular angle excursion >0.01rad',
        disclaimer='Thresholds0.001/0.01/0.05 are descriptive sensitivity checks, not externally calibrated pass/fail criteria; normalization uses test anchors, no model fitting',
        kinetic_potential='K=0.5||p||^2; V=H-K, consistent with fixed M=I implementation'),
    no_exact_update_freeze=dict(q_exactly_unchanged_transition_count=int(np.all(dq==0,axis=-1).sum()),
        p_exactly_unchanged_transition_count=int(np.all(dp==0,axis=-1).sum()),
        transition_count=int(np.prod(dq.shape[:2]))),
    groups=groups, teacher=dict(scope='44 metadata-selected trajectories, true future image histories re-encoded; diagnostic only',groups=teacher_groups),
    renderer_zero_discontinuity=dict(rows=renderer_rows,
        interpretation='Rasterization has a discontinuity at zero angle. Establishes a possible source of noisy latent derivatives, not by itself the cause of the learned stall.'))
(OUT/'stall_quantification.json').write_text(json.dumps(results,indent=2)+'\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
minus,plus=render(-1e-12),render(1e-12)
fig,axes=plt.subplots(1,3,figsize=(8,2.8),layout='constrained')
axes[0].imshow(minus,interpolation='nearest');axes[0].set_title('theta = -1e-12 rad')
axes[1].imshow(plus,interpolation='nearest');axes[1].set_title('theta = +1e-12 rad')
axes[2].imshow(np.any(minus!=plus,axis=-1),cmap='magma',interpolation='nearest');axes[2].set_title('Changed pixels (30)')
for ax in axes: ax.set_axis_off()
fig.savefig(OUT/'renderer_zero_discontinuity.png',dpi=160)
plt.close(fig)
print(json.dumps({'result':str(OUT/'stall_quantification.json'),'renderer':renderer_rows[0]}))
