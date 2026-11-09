import hashlib,importlib.util,io,json,math,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path('/home/weizhixin/ssd/official_comparison_launch_20261108T092500Z')
sys.path.insert(0,str(root/'code'))
import torch
torch.set_num_threads(2)
from dpbench.comparison_protocol import source_manifest,json_digest,digest
spec=importlib.util.spec_from_file_location('frozen_run_comparison',root/'code/scripts/run_comparison.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
launch=json.loads((root/'launch_manifest.json').read_text());sources=source_manifest();assert sources==launch['source_files']
report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':json_digest(sources),'data_sha256':{},'methods':{},'scheduler_accounting':'disabled; jobs absent from live queue; completion verified from application manifests, checkpoints and artifact hashes'}
for stage,path in launch['data'].items():
 report['data_sha256'][stage]=digest(path)
 assert report['data_sha256'][stage]==launch['expected_data_sha256'][stage]
for method in ['lehami','lewm','hjepa']:
 pipe=root/'runs'/method;base=pipe/(method+'_seed0');outer=json.loads((pipe/'status.json').read_text());assert outer['status']=='completed'
 row={'pipeline_status':outer,'pipeline_manifest_created_at_utc':datetime.fromtimestamp((pipe/'pipeline.json').stat().st_mtime,timezone.utc).isoformat(),'pipeline_completion_file_at_utc':datetime.fromtimestamp((pipe/'status.json').stat().st_mtime,timezone.utc).isoformat(),'pipeline_wall_seconds_by_mtime':(pipe/'status.json').stat().st_mtime-(pipe/'pipeline.json').stat().st_mtime,'stages':{}}
 for stage in ['a','single','multi']:
  out=base/stage;worker=json.loads((out/'worker_status.json').read_text());assert worker['status']=='completed'
  ck=json.loads((out/'checkpoint_manifest.json').read_text());path=Path(ck['path']);payload=path.read_bytes();sha=hashlib.sha256(payload).hexdigest();assert sha==ck['sha256'] and ck['step']==20000
  saved=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
  assert saved['step']==20000 and saved['validation_status']=='completed' and saved['source_sha256']==sources
  assert saved['dataset_sha256']==report['data_sha256'][stage]
  assert all(torch.isfinite(v).all().item() for v in saved['state_dict'].values() if v.is_floating_point())
  assert math.isfinite(saved['validation']['score'])
  if stage!='a':assert saved['parent_transfer']['sha256']==row['stages']['a']['checkpoint_sha256']
  directory=out/'evaluation';module.verify_evaluation(directory,sha,report['data_sha256'][stage],json_digest(saved['train_config']),20000)
  summary=json.loads((directory/'summary.json').read_text());train_status=json.loads((path.parent/'status.json').read_text());assert train_status['status']=='completed' and train_status['completed_steps']==20000
  lines=[json.loads(s) for s in (path.parent/'history.jsonl').read_text().splitlines() if s.strip()]
  assert len(lines)==20000 and [r['step'] for r in lines]==list(range(1,20001))
  finite=all(all(math.isfinite(v) for v in r['losses'].values()) and math.isfinite(r['gradient_norm_before_clip']) for r in lines)
  row['stages'][stage]={'training_status':train_status,'checkpoint_step':saved['step'],'checkpoint_sha256':sha,'checkpoint_floating_state_finite':True,'source_and_data_hash_verified':True,'all_history_losses_gradients_finite':finite,'history_contiguous_1_to_20000':True,'evaluation_integrity':'passed','evaluation_summary':summary,'worker_status':worker,'final_latent_validation':saved['validation']}
  del saved,payload
 report['methods'][method]=row
report['all_nine_training_and_evaluations_verified']=True
print(json.dumps(report,indent=2,allow_nan=False))
