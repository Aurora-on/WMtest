import hashlib,json,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'code'))
manifest=json.loads((root/'launch_manifest.json').read_text())
for name,expected in manifest['bundle_files'].items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Bundle mismatch: '+name)
from dpbench.comparison_protocol import source_manifest,json_digest,preflight,make_config
from dpbench.damping_stage_data import validate_pair
sources=source_manifest()
assert sources==manifest['source_files']
report=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),bundle_verified_files=len(manifest['bundle_files']),source_sha256=json_digest(sources),preflight={},paired={})
for stage,path in manifest['data'].items():
 value=preflight(path,make_config('lehami',stage))
 assert value['dataset_sha256']==manifest['expected_data_sha256'][stage]
 report['preflight'][stage]=value
for stage in ['single','multi']:report['paired'][stage]=validate_pair(manifest['data']['a'],manifest['data'][stage],multi=stage=='multi')
import torch,numpy,scipy,h5py,einops,omegaconf
report['versions']=dict(python=sys.version,torch=torch.__version__,numpy=numpy.__version__,scipy=scipy.__version__,h5py=h5py.__version__,einops=einops.__version__,omegaconf=omegaconf.__version__)
report['status']='passed';report['optimizer_updates']=0
(root/'remote_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','checked_at_utc','bundle_verified_files','source_sha256','versions','optimizer_updates']},indent=2))
