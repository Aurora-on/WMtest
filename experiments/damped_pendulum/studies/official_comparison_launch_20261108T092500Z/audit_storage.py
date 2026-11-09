import json, os, subprocess
from pathlib import Path
from datetime import datetime, timezone
base=Path('/home/weizhixin/ssd')
protected=[base/'undamped_stage_a_launch_20261105T053810Z/data/undamped_v1',base/'staged_damping_d2_launch_20261106T064931Z/run/data_single',base/'staged_damping_d2_launch_20261106T064931Z/run/data_multi',base/'damped_pendulum_profile_20261028/.venv']
relative=['MOSH_validation_20260924/MOSH_WM_Server/data','MOSH_appearance_20261005/MOSH_WM_Appearance8000/data','MOSH_reproduction_v2_20261008/data','MOSH_motion_split_20261008/data','damped_pendulum_profile_20261028/data','damped_pendulum_profile_20261028/studies/independent_eval_20261029T072955Z/study/data','dissipative_passive_20261103T051548Z/data','lehami_official_reproduction_20261029/data','lehami_official_reproduction_20261029/smoke/data','MOSH_appearance_20261005/moshwm_appearance8000.tar.gz','MOSH_appearance_20261005/early_preflight_payload.tar.gz','mosh-transfer/moshwm_ladhpc.tar.gz']
rows=[]
for rel in relative:
 p=base/rel
 if not p.exists():continue
 files=[p] if p.is_file() else list(p.rglob('*'))
 links=[str(f) for f in files if f.is_symlink()]
 regular=[f for f in files if f.is_file() and not f.is_symlink()]
 rows.append(dict(path=str(p),relative=rel,files=len(regular),bytes=sum(f.stat().st_size for f in regular),symlinks=links,protected_overlap=any(p==q or p in q.parents or q in p.parents for q in protected)))
open_refs=[]
for process in Path('/proc').iterdir():
 if not process.name.isdigit():continue
 try:
  if process.stat().st_uid!=os.getuid():continue
  for fd in (process/'fd').iterdir():
   try: target=Path(os.readlink(fd))
   except OSError:continue
   if any(target==Path(r['path']) or Path(r['path']) in target.parents for r in rows):open_refs.append(dict(pid=process.name,target=str(target)))
 except OSError:continue
print(json.dumps(dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),candidates=rows,protected=[str(p) for p in protected],open_references=open_refs,squeue=subprocess.check_output(['squeue','-h','-u','weizhixin','-o','%i %T %j'],text=True),disk={str(p):dict(zip(['total','used','free'],__import__('shutil').disk_usage(p))) for p in [base,Path('/home/weizhixin/hdd')]},total_candidate_bytes=sum(r['bytes'] for r in rows)),indent=2))
