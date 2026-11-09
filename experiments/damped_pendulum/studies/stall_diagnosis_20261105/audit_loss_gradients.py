"""Frozen-checkpoint objective audit. Backward only: zero optimizer updates."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from dpbench.dissipative_model import DissipativeWorldModel
from dpbench.dissipative_data import DissipativeWindowDataset,StepBatchSampler
from dpbench.train_dissipative import move_batch

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
 a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(3200511);started=time.perf_counter()
 saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
 m=DissipativeWorldModel(saved['config']).to(a.device);m.load_state_dict(saved['state_dict']);m.train()
 before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 cfg=saved['train_config']['training'];ds=DissipativeWindowDataset(a.data,'train',m.history,32,cfg['train_frames'])
 ids=StepBatchSampler(ds,cfg['batch_size'],cfg['seed']).indices(saved['step']-1)
 batch=move_batch(ds.batch(ids,32),a.device)
 r=m.predict_batch(batch,32);q,target=r['q'],r['target'];dt=.1
 pred=(q[:,1:]-target[:,1:]).square().mean()
 motion=((q[:,1:]-q[:,:-1])-(target[:,1:]-target[:,:-1])).square().mean()
 reg=m.sigreg(r['all_encoded'].transpose(0,1));init=r['correction'].square().mean()
 terms={'prediction':pred,'motion_weighted':m.config['motion_weight']*motion,
        'sigreg_weighted':m.config['sigreg_weight']*reg,'initializer_weighted':m.config['initializer_weight']*init}
 params=list(m.named_parameters());ps=[v for _,v in params]
 def group(n):
  if n.startswith('encoder.'):return 'encoder'
  if n.startswith('core.potential_net.'):return 'potential'
  if n.startswith('core.damping_net.'):return 'damping'
  if n.startswith('environment_encoder.'):return 'environment_context'
  if n.startswith('velocity_correction.'):return 'velocity_correction'
  if n.startswith('core.input_net.'):return 'input_port'
  return 'other'
 group_names=sorted(set(group(n) for n,_ in params));vectors={};rows={}
 for key,loss in terms.items():
  gg=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
  grouped={name:[] for name in group_names}
  for (n,v),g in zip(params,gg): grouped[group(n)].append(torch.zeros(v.numel(),dtype=torch.float64) if g is None else g.detach().flatten().cpu().double())
  vectors[key]={name:torch.cat(g) for name,g in grouped.items()}
  rows[key]={'value':float(loss.detach()),'gradnorms':{name:float(v.norm()) for name,v in vectors[key].items()}}
 totals={name:sum(vectors[k][name] for k in vectors) for name in group_names}
 global_norm=float(torch.sqrt(sum(v.square().sum() for v in totals.values())))
 cos=[]
 for g in group_names:
  for one,two in [('prediction','motion_weighted'),('prediction','sigreg_weighted'),('motion_weighted','sigreg_weighted')]:
   x,y=vectors[one][g],vectors[two][g];den=float(x.norm()*y.norm())
   cos.append({'group':g,'first':one,'second':two,'cosine':float(x@y)/den if den>0 else None})
 with torch.no_grad():
  # Centers j=2..30 are available in frozen target trajectory q[0..32].
  ps_target=(-2*target[:,:-4]-target[:,1:-3]+target[:,3:-1]+2*target[:,4:])/(10*dt)
  pp=r['p'][:,2:-2]
  p_mse=float((pp-ps_target).square().mean());p_zero=float(ps_target.square().mean())
  predicted_motion_rms=float((q[:,1:]-q[:,:-1]).square().mean().sqrt())
  target_motion_rms=float((target[:,1:]-target[:,:-1]).square().mean().sqrt())
  encoded=r['all_encoded'].double();anchor=m.history-1
  center=(-2*encoded[:,anchor-2]-encoded[:,anchor-1]+encoded[:,anchor+1]+2*encoded[:,anchor+2])/(10*dt)
  corrected=r['p'][:,0];correction=r['correction'].double();causal=corrected-correction
  forward=(target[:,1]-target[:,0])/dt
  init_rows=[]
  regimes=[ds.index[i].regime_id for i in ids]
  for regime in [None,*sorted(set(regimes))]:
   mask=torch.ones(len(ids),dtype=torch.bool,device=a.device) if regime is None else torch.tensor([x==regime for x in regimes],device=a.device)
   init_rows.append({'regime_id':regime,'n':int(mask.sum()),
     'centered_sg_rms':float(center[mask].square().mean().sqrt()),
     'causal_sg_rms':float(causal[mask].square().mean().sqrt()),
     'corrected_p0_rms':float(corrected[mask].square().mean().sqrt()),
     'correction_rms':float(correction[mask].square().mean().sqrt()),
     'correction_abs_gt095_fraction':float((correction[mask].abs()>.95).float().mean()),
     'correction_abs_gt099_fraction':float((correction[mask].abs()>.99).float().mean()),
     'causal_vs_centered_mse':float((causal[mask]-center[mask]).square().mean()),
     'corrected_vs_centered_mse':float((corrected[mask]-center[mask]).square().mean()),
     'forward_delta_vs_centered_mse':float((forward[mask]-center[mask]).square().mean())})
  baseline_rows=[]
  for label,prediction in [('model',q),('hold_anchor',target[:,:1].expand_as(target)),
      ('future_mean_oracle_diagnostic',torch.cat([target[:,:1],target[:,1:].mean(1,keepdim=True).expand_as(target[:,1:])],1))]:
   err=float((prediction[:,1:]-target[:,1:]).square().mean())
   mot=float(((prediction[:,1:]-prediction[:,:-1])-(target[:,1:]-target[:,:-1])).square().mean())
   baseline_rows.append({'method':label,'q_prediction_mse':err,'motion_mse':mot,'sum_q_and_motion':err+m.config['motion_weight']*mot})
  p_state_rms=[float(r['p'][:,j].square().mean().sqrt()) for j in (0,1,2,5,10,20,32)]
  force=[]
  for j in (0,1,10,32):
   d=m.core.damping(q[:,j],r['p'][:,j],r['context']);force.append({'step':j,'D_eigenvalue_mean':torch.linalg.eigvalsh(d).mean(0).cpu().tolist()})
  unchanged=all(torch.equal(before[k],v.detach().cpu()) for k,v in m.state_dict().items())
 assert unchanged
 result={'scope':'One frozen training batch, original objective differentiated. No optimizer created or updates. Not a causal training ablation.',
         'checkpoint_step':saved['step'],'batch_sampler_step0':saved['step']-1,'batch_windows':[vars(ds.index[i]) for i in ids],
         'horizon':32,'batch_size':len(ids),'terms':rows,'gradient_cosines':cos,'total_grad_norm':global_norm,
         'would_clip_scale':min(1,cfg['gradient_clip']/(global_norm+1e-6)),
         'total_grad_norm_by_group':{k:float(v.norm()) for k,v in totals.items()},
         'future_p_sg_mse':p_mse,'future_zero_p_sg_mse':p_zero,'future_p_vs_zero_ratio':p_mse/p_zero,
         'predicted_motion_rms':predicted_motion_rms,'target_motion_rms':target_motion_rms,
         'initializer_comparison':init_rows,
         'initializer_reference_scope':'CenteredSG and forwarddifference are image-derived approximate derivatives, NOT true canonical state.',
         'stationary_objective_comparison':baseline_rows,
         'stationary_scope':'Hold-anchor uses current observations. Future-mean uses actual encoded future labels, oracle diagnostic only, not deployable baseline.',
         'p_state_rms_steps_0_1_2_5_10_20_32':p_state_rms,'damping_eigenvalues':force,
         'state_unchanged':unchanged,'optimizer_updates':0,'elapsed_seconds':time.perf_counter()-started}
 Path(a.output).write_text(json.dumps(result,indent=2)+'\n');ds.close();print(json.dumps({'status':'completed','output':a.output,'seconds':result['elapsed_seconds']}))
if __name__=='__main__':main()
