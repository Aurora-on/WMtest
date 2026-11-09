import json,math,statistics,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'methods':{}}
def command(args):
 r=subprocess.run(args,text=True,capture_output=True);return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
report['squeue']=command(['squeue','-u','weizhixin','-o','%.18i %.24j %.2t %.10M %.20b %R'])
report['queue_start_estimates']=command(['squeue','--start','-u','weizhixin'])
report['sacct']=command(['sacct','-j','48044,48045,48046','--format=JobID,JobName,State,Elapsed,Start,End,ExitCode,MaxRSS','-P'])
report['gpu']=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv'])
for method in ['lehami','lewm','hjepa']:
 item={'formal_stages':{},'profiles':[]}
 for p in sorted((root/'profiles').glob(method+'*.json')):
  v=json.loads(p.read_text());s={'path':str(p),'status':v['status'],'phase':v.get('phase'),'elapsed_seconds':v.get('elapsed_seconds'),'error':v.get('error'),'estimate':v.get('estimate'),'stages':{}}
  for stage,r in v.get('stages',{}).items():
   s['stages'][stage]={'status':r.get('status'),'validation':r.get('validation'),'horizons':{k:{key:b.get(key) for key in ['statistics','peak_allocated_bytes','peak_reserved_bytes','status','measured_steps']} for k,b in r.get('horizons',{}).items()}}
  item['profiles'].append(s)
 base=root/'runs'/method/(method+'_seed0')
 for stage in ['a','single','multi']:
  directory=base/stage
  if not directory.exists():continue
  r={}
  for name in ['worker_status.json','checkpoint_manifest.json']:
   if (directory/name).exists():r[name]=json.loads((directory/name).read_text())
  segments=sorted((directory/'segments').glob('*'))
  if segments:
   seg=segments[-1];r['segment']=str(seg)
   if (seg/'status.json').exists():r['status']=json.loads((seg/'status.json').read_text())
   history=seg/'history.jsonl'
   if history.exists():
    rows=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    if rows:
     r['last_row']=rows[-1];r['history_rows']=len(rows)
     r['all_recorded_losses_and_gradients_finite']=all(all(math.isfinite(v) for v in row['losses'].values()) and math.isfinite(row['gradient_norm_before_clip']) for row in rows)
     r['validations']=[{'step':row['step'],'validation_seconds':row.get('validation_seconds'),'validation':row['validation']} for row in rows if 'validation' in row]
     samples=[row['train_seconds'] for row in rows[-200:]];med=statistics.median(samples);kept=[x for x in samples if x<=max(2,5*med)]
     r['recent_training_timing']={'window_rows':len(samples),'horizon':rows[-1]['horizon'],'median_seconds':med,'mean_seconds_excluding_pauses':statistics.mean(kept),'outlier_threshold_seconds':max(2,5*med),'excluded_rows':len(samples)-len(kept),'scope':'Exclude explicit profiling suspension or isolated large stalls; compute prediction remains early-stage only'}
   if (seg/'last.pt').exists():r['last_checkpoint_bytes']=(seg/'last.pt').stat().st_size
  item['formal_stages'][stage]=r
 report['methods'][method]=item
for name in ['profile_bridge.json','storage_migration.json','submission.json']:
 if (root/name).exists():report[name]=json.loads((root/name).read_text())
path=root/'initial_monitor.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
print(json.dumps({'checked_at_utc':report['checked_at_utc'],'queue':report['squeue']['stdout'],'methods':{m:{'profiles':[(p['status'],p['phase'],p['estimate'].get('single_gpu_sequential_subtotal_hours') if p['estimate'] else None) for p in v['profiles']],'formal_stages':{s:r.get('status') for s,r in v['formal_stages'].items()}} for m,v in report['methods'].items()}},indent=2))
