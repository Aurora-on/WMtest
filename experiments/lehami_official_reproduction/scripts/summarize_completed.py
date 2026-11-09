"""Summarize verified seed42 results against the author's matching published seed."""
from pathlib import Path
import csv
import json
import math
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'received/completed_47141'
AUTHOR=ROOT/'upstream/LeHami/paper_figures/data/conservative_pendulum'
OUT=ROOT/'reports'
OUT.mkdir(exist_ok=True)

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def test_curve(path):
    rs=sorted((r for r in rows(path) if r['split']=='test'),key=lambda r:int(r['k']))
    assert [int(r['k']) for r in rs]==list(range(71))
    return rs

ours=test_curve(RUN/'results/exp067_a0m_lehami_w1_multi/probe_metrics.csv')
author=test_curve(AUTHOR/'probe_metrics/exp067_a0m_lehami_w1_multi_s0.csv')
fields={'theta':'mae_theta__knn_k1','omega':'mae_omega__knn_k1'}
summary={'job_id':'47141','status':'COMPLETE','model':'LeHami','seed':42,
         'epochs_completed':100,'optimizer_updates':43900,'evaluated_best_epoch_1based':98,
         'test_episodes':45,'readout':'knn_k1, 25 neighbors fitted on model one-step outputs',
         'time_seconds_per_step':0.1,'checkpoint_audit':'../received/completed_47141/metadata/final_checkpoint_audit_47141.json',
         'published_reference':'Author exported s0 corresponds to original config seed42; not a three-seed average',
         'endpoint_metrics':[],'time_band_means':[]}
for k in [10,30,70]:
    d={'k':k,'seconds':k*.1}
    for name,rs in [('ours',ours),('author_seed42',author)]:
        d[name]={'theta_mae_rad':float(rs[k][fields['theta']]),
                 'theta_mae_deg':math.degrees(float(rs[k][fields['theta']])),
                 'omega_mae_rad_s':float(rs[k][fields['omega']])}
    summary['endpoint_metrics'].append(d)
for lo,hi in [(1,10),(11,20),(21,70)]:
    d={'k_inclusive':[lo,hi]}
    for name,rs in [('ours',ours),('author_seed42',author)]:
        d[name]={key+'_mae':statistics.mean(float(r[field]) for r in rs[lo:hi+1]) for key,field in fields.items()}
    summary['time_band_means'].append(d)
physical={}
for name,base in [('ours',RUN/'results'),('author_seed42',AUTHOR)]:
    jac=[r for r in rows(base/'latent_volume/jacobian_rollout.csv') if r['arm']=='a0m']
    assert len(jac)==3150
    vals=[abs(float(r['logdet_per_dim_full'])) for r in jac]
    reversal=rows(base/('exp067_a0m_lehami_w1_multi' if name=='ours' else 'exp067_a0m')/'physics_reversal.csv')
    k30=next(r for r in reversal if int(r['K'])==30)
    physical[name]={'mean_abs_normalized_logdet':statistics.mean(vals),'max_abs_normalized_logdet':max(vals),
                    'K30_latent_dt_flip_relative_error':float(k30['roundtrip_rel_err_dtflip']),
                    'K30_latent_p_flip_relative_error':float(k30['roundtrip_rel_err_pflip'])}
summary['latent_structure']=physical
summary['limits']=['Single seed only; not a new LeWM comparison.',
 'Original source/data and protocol, with CUDA12.6 adaptation and checkpoint resume; not bitwise training equivalence.',
 'Latent volume/reversibility do not establish true mechanical energy conservation or accurate long-horizon trajectories.',
 'Summary.json un-suffixed physical metrics use ridge_gt; this report follows the paper main knn_k1 readout.']
summary['resume_job_wall_seconds']=json.loads((RUN/'metadata/lehami_one_seed_47141/job.json').read_text())['wall_seconds']
summary['resume_job_time_note']='Includes 85 resumed epochs and evaluations; earlier pre-resume training/setup is excluded.'
(OUT/'results_47141.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),constrained_layout=True)
for ax,key,scale,label in [(axes[0],'theta',180/math.pi,'Wrapped angle MAE (degrees)'),
                           (axes[1],'omega',1,'Angular velocity MAE (rad/s)')]:
    for rs,color,style,name in [(author,'#68747e','--','Author published seed 42'),(ours,'#176bba','-','Our seed 42')]:
        ax.plot([int(r['k'])*.1 for r in rs],[float(r[fields[key]])*scale for r in rs],
                color=color,ls=style,lw=2,label=name)
    ax.set(xlabel='Forecast time (s)',ylabel=label,xlim=(0,7),ylim=(0,None))
    ax.grid(alpha=.18)
axes[0].legend(frameon=False,loc='upper left',fontsize=9)
fig.suptitle('LeHami: official conservative-pendulum protocol | 45 test episodes | knn_k1 readout',fontsize=11)
fig.savefig(OUT/'comparison_47141.png',dpi=180)
plt.close(fig)
print(json.dumps({'endpoint_7s':summary['endpoint_metrics'][-1],'bands':summary['time_band_means'],
                  'physics':physical,'report':str(OUT/'results_47141.json')},indent=2))
