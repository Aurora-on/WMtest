import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
migration=json.loads((root/'storage_migration.json').read_text())
assert migration['status']=='completed'
assert all(i['sha256_verified'] and i['source_symlink_preserved'] and Path(i['source']).is_symlink() and Path(i['source']).resolve()==Path(i['destination']).resolve() for i in migration['items'])
assert json.loads((root/'remote_preflight.json').read_text())['status']=='passed'
manifest=json.loads((root/'launch_manifest.json').read_text())
for name,expected in manifest['bundle_files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
path=root/'submission.json'
record=json.loads(path.read_text()) if path.exists() else dict(created_at_utc=datetime.now(timezone.utc).isoformat(),jobs={},status='submitting',methods=['lehami','lewm','hjepa'],seed=0,gpus_per_job=1,training_plan='A20000 -> single20000 -> multi20000',profile_before_training=True)
for method in record['methods']:
 if method in record['jobs']:continue
 command=['sbatch','--parsable',str(root/f'submit_{method}.sh')]
 result=subprocess.run(command,text=True,capture_output=True,check=True)
 jobid=result.stdout.strip().split(';')[0]
 assert jobid.isdigit(),result.stdout
 record['jobs'][method]=dict(job_id=jobid,submitted_at_utc=datetime.now(timezone.utc).isoformat(),command=command)
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
 print(method,jobid,flush=True)
record['status']='submitted'
temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(path)
