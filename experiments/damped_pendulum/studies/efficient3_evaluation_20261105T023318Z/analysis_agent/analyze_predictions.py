from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
r=ROOT/'received'
a=np.load(r/'trajectories.npz')
records=json.loads((r/'episodes.json').read_text())
b=np.array([x['damping'] for x in records]);groups=np.array([x['group_id'] for x in records])
true=a['truth'];pred=a['prediction_model'];q=a['features'][...,:2];p=a['features'][...,2:]
energy=lambda x:.5*x[...,1]**2+9.81*(1-np.cos(x[...,0]))
et,ep=energy(true),energy(pred);e0=a['initial_energy'];t=a['time'];dt=t[1]-t[0]
angle=np.arctan2(np.sin(pred[...,0]-true[...,0]),np.cos(pred[...,0]-true[...,0]))
omega=pred[...,1]-true[...,1]
rmse=lambda x:float(np.sqrt(np.mean(x*x)))
# Centered finite differences evaluated on decoded angle, with truth control.
dtheta=np.gradient(np.unwrap(pred[...,0],axis=1),dt,axis=1)
dtheta_true=np.gradient(np.unwrap(true[...,0],axis=1),dt,axis=1)
by_b={}
for value in np.unique(b):
 m=b==value
 out=[]
 for sec in [0,.1,.2,.5,1,2,4,8,16,32]:
  k=round(sec/dt)
  out.append(dict(time_s=sec,angle_rmse=rmse(angle[m,k]),omega_rmse=rmse(omega[m,k]),
    true_omega_rms=rmse(true[m,k,1]),pred_omega_rms=rmse(pred[m,k,1]),
    true_mean_energy_fraction=float(np.mean(et[m,k]/e0[m])),pred_mean_energy_fraction=float(np.mean(ep[m,k]/e0[m])),
    q_rms=rmse(q[m,k]),p_rms=rmse(p[m,k]),mean_latent_energy=float(np.mean(a['latent_energy'][m,k]))))
 by_b[str(value)]={'timepoints':out,'late16_32s_angle_rmse':rmse(angle[m,160:]),
   'late16_32s_true_theta_rms':rmse(true[m,160:,0]),'late16_32s_pred_theta_rms':rmse(pred[m,160:,0]),
   'late16_32s_omega_rmse':rmse(omega[m,160:]),'late16_32s_true_omega_rms':rmse(true[m,160:,1]),
   'late16_32s_pred_omega_rms':rmse(pred[m,160:,1]),
   'decoded_kinematic_defect_rmse':rmse((dtheta-pred[...,1])[m,1:-1]),
   'reference_kinematic_defect_rmse':rmse((dtheta_true-true[...,1])[m,1:-1])}
# Midpoint dq/h equals (p0+p1)/2 structurally; this is latent, not decoded kinematics.
latent_defect=np.diff(q,axis=1)/dt-.5*(p[:,1:]+p[:,:-1])
# Single initial q point is based only on an image; distance from rollout q to empirical anchor image-manifold.
# This finite anchor set only approximates the manifold; use zero-step NN spacing as a sensitivity reference.
from scipy.spatial import cKDTree
anchor=q[:,0];tree=cKDTree(anchor)
scale=np.sqrt(np.mean(np.var(anchor,axis=0)))
nd,_=tree.query(q.reshape(-1,2));nd=nd.reshape(q.shape[:2])
# Nearest other q often identical due pixel quantization; use interpolation of image-q vs true angle instead.
# Sort anchor samples by angle; average duplicate angles before linear interpolation.
th=np.arctan2(np.sin(true[:,0,0]),np.cos(true[:,0,0]));idx=np.argsort(th);uniq,inv=np.unique(th[idx],return_inverse=True)
uq=np.zeros((len(uniq),2));np.add.at(uq,inv,anchor[idx]);uq/=np.bincount(inv)[:,None]
interp=np.stack([np.interp(np.arctan2(np.sin(true[...,0]),np.cos(true[...,0])).ravel(),uniq,uq[:,j]) for j in range(2)],axis=-1).reshape(q.shape)
results={'scope':'posthoc diagnostic from frozen test outputs; no fitting or changing world model/readout',
 'episodes':len(records),'all_full_rollouts':bool((a['valid_steps']==320).all()),
 'anchor':{'angle_mae':float(np.mean(abs(angle[:,0]))),'omega_mae':float(np.mean(abs(omega[:,0])))},
 'decoded_kinematic_defect_rmse':rmse((dtheta-pred[...,1])[:,1:-1]),
 'reference_kinematic_defect_rmse':rmse((dtheta_true-true[...,1])[:,1:-1]),
 'macro_latent_midpoint_defect_rmse':rmse(latent_defect),
 'macro_latent_midpoint_defect_note':'Each frame contains3substeps; frameendpoint trapezoid need not equal summed substep midpoint integral.',
 'by_damping':by_b,
 'empirical_anchor_manifold':{'scope':'posthoc nearest distance to all1408initialq points, not definitive manifold distance',
  'anchor_coordinate_scale':float(scale),'mean_distance_by_time':{str(sec):float(nd[:,round(sec/dt)].mean()) for sec in [0,.1,.2,.5,1,2,4,8,16,32]},
  'median_distance_by_time':{str(sec):float(np.median(nd[:,round(sec/dt)])) for sec in [0,.1,.2,.5,1,2,4,8,16,32]}},
 'physical_energy_increase_fraction':float(np.mean(np.diff(ep,axis=1)>1e-5)),
 'latent_energy_increase_fraction':float(np.mean(np.diff(a['latent_energy'],axis=1)>1e-10))}
(ROOT/'analysis_agent/prediction_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps({k:v for k,v in results.items() if k!='by_damping'},indent=2))
for v,out in by_b.items():
 print('b',v,'Etrue/pred@1s',out['timepoints'][4]['true_mean_energy_fraction'],out['timepoints'][4]['pred_mean_energy_fraction'],'@32s',out['timepoints'][-1]['true_mean_energy_fraction'],out['timepoints'][-1]['pred_mean_energy_fraction'],'kinematic',out['decoded_kinematic_defect_rmse'],'late theta rms',out['late16_32s_true_theta_rms'],out['late16_32s_pred_theta_rms'])
