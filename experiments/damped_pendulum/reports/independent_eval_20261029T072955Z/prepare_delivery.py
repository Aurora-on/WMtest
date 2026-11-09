from pathlib import Path
import json,shutil,hashlib
import numpy as np
root=Path(__file__).resolve().parent; source=root/'results_single'; dest=root/'delivery'; dest.mkdir(exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for path in sorted(source.glob('*/metrics.json')):
 out=dest/path.parent.name
 if (out/'trajectory_provenance.json').exists():continue
 out.mkdir(exist_ok=True)
 with np.load(path.parent/'trajectories.npz') as saved:
  assert np.isfinite(saved['features']).all()
  arrays={k:saved[k] for k in ('predictions','truth','episode_ids','probe_names','dt')}
  assert all(np.isfinite(arrays[k]).all() for k in ('predictions','truth'))
  np.savez_compressed(out/'predictions.npz',**arrays)
  feature_shape=list(saved['features'].shape)
 shutil.copyfile(path,out/'metrics.json')
 (out/'trajectory_provenance.json').write_text(json.dumps({'source_file':str(path.parent/'trajectories.npz'),'source_sha256':sha(path.parent/'trajectories.npz'),'features_shape':feature_shape,'all_features_finite':True,'compact_file':'predictions.npz','compact_sha256':sha(out/'predictions.npz')},indent=2)+'\n')
for name in ('completion.json','evaluation_protocol.json'):
 if (source/name).exists():shutil.copyfile(source/name,dest/name)
files=[p for p in sorted(dest.rglob('*')) if p.is_file() and not p.name.startswith('manifest_')]
n=len(list(dest.glob('*/metrics.json')))
manifest={'completed_run_count':n,'files':[{'path':str(p.relative_to(dest)),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]}
name=f'manifest_{n}.json'
(dest/name).write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'complete_runs':n,'files':len(files),'bytes':sum(p.stat().st_size for p in files),'manifest':name,'manifest_sha256':sha(dest/name)}))
