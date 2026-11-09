import hashlib,io,json,math,time
from pathlib import Path
from datetime import datetime,timezone
import torch
root=Path(__file__).resolve().parent
methods=['lehami','lewm','hjepa'];deadline=time.monotonic()+180
paths={m:root/'runs'/m/(m+'_seed0')/'a/segments/0001/last.pt' for m in methods}
while not all(p.exists() for p in paths.values()):
 if time.monotonic()>deadline:raise TimeoutError('Not all first checkpoints appeared')
 time.sleep(3)
manifest=json.loads((root/'launch_manifest.json').read_text())
torch.set_num_threads(2)
report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','scope':'early real Stage A checkpoints; not scientific performance acceptance','methods':{}}
for method,path in paths.items():
 payload=path.read_bytes();checkpoint=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
 assert checkpoint['step']>=250
 assert checkpoint['validation_status']=='completed'
 assert checkpoint['source_sha256']==manifest['source_files']
 assert checkpoint['dataset_sha256']==manifest['expected_data_sha256']['a']
 assert all(torch.isfinite(v).all().item() for v in checkpoint['state_dict'].values() if v.is_floating_point())
 assert math.isfinite(checkpoint['validation']['score'])
 report['methods'][method]=dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),step=checkpoint['step'],validation_status=checkpoint['validation_status'],validation_score=checkpoint['validation']['score'],all_model_floats_finite=True,source_and_data_match=True,sampler_next_step=checkpoint['sampler_next_step'],optimizer_saved='optimizer' in checkpoint,rng_saved='rng' in checkpoint)
path=root/'checkpoint_qualification.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
