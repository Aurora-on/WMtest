import hashlib,json,os,shutil,subprocess,time,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/home/weizhixin/ssd/official_comparison_launch_20261108T092500Z')
ARCHIVE=Path('/home/weizhixin/hdd/archive_20261108/ssd')
plan=json.loads((ROOT/'storage_before.json').read_text())
def atomic(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def files(path):return [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file() or p.is_symlink())
if subprocess.check_output(['squeue','-h','-u','weizhixin'],text=True).strip():raise RuntimeError('User jobs appeared before migration')
if plan['open_references']:raise RuntimeError('Active file references')
ARCHIVE.mkdir(parents=True,exist_ok=True)
report=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),status='running',archive_root=str(ARCHIVE),items=[],logical_bytes_migrated=0)
atomic(ROOT/'storage_migration.json',report)
try:
 for index,item in enumerate(plan['candidates']):
  src=Path(item['path']);dst=ARCHIVE/item['relative']
  if item['protected_overlap'] or src.is_symlink() or dst.exists():raise RuntimeError('Unexpected protected/symlink/existing destination: '+str(src))
  entries=files(src);regular=[p for p in entries if not p.is_symlink()]
  if len(regular)!=item['files'] or sum(p.stat().st_size for p in regular)!=item['bytes']:raise RuntimeError('Source inventory changed')
  report['current']=dict(source=str(src),destination=str(dst),phase='copy_and_hash',files=item['files']);atomic(ROOT/'storage_migration.json',report)
  dst.parent.mkdir(parents=True,exist_ok=True)
  temp=dst.with_name('.'+dst.name+'.partial_'+uuid.uuid4().hex)
  manifest=ROOT/f'archive_files_{index:02d}.jsonl'
  before={str(p.relative_to(src) if src.is_dir() else Path('.')):(p.lstat().st_size,p.lstat().st_mtime_ns) for p in entries}
  if src.is_dir():shutil.copytree(src,temp,symlinks=True)
  else:shutil.copy2(src,temp)
  report['current']['phase']='verify_sha256';atomic(ROOT/'storage_migration.json',report)
  with manifest.open('w') as log:
   for p in entries:
    rel=p.relative_to(src) if src.is_dir() else Path('.')
    target=temp/rel if src.is_dir() else temp
    if p.is_symlink():
     if os.readlink(p)!=os.readlink(target):raise RuntimeError('Symlink mismatch')
     record=dict(relative=str(rel),symlink=os.readlink(p))
    else:
     source_sha=sha(p)
     if source_sha!=sha(target):raise RuntimeError('SHA256 mismatch: '+str(p))
     record=dict(relative=str(rel),bytes=p.stat().st_size,sha256=source_sha)
    if before[str(rel)]!=(p.lstat().st_size,p.lstat().st_mtime_ns):raise RuntimeError('Source changed during copy')
    log.write(json.dumps(record)+'\n')
  if set(map(str,entries))!=set(map(str,files(src))):raise RuntimeError('File list changed during copy')
  for p in entries:
   rel=p.relative_to(src) if src.is_dir() else Path('.')
   if before[str(rel)]!=(p.lstat().st_size,p.lstat().st_mtime_ns):raise RuntimeError('Source changed before switch')
  os.replace(temp,dst)
  backup=src.with_name('.'+src.name+'.verified_archive_'+uuid.uuid4().hex)
  os.rename(src,backup)
  try:os.symlink(dst,src,target_is_directory=dst.is_dir())
  except BaseException:
   os.rename(backup,src);raise
  if src.resolve()!=dst.resolve():raise RuntimeError('New source link mismatch')
  if backup.is_dir():shutil.rmtree(backup)
  else:backup.unlink()
  report['items'].append(dict(source=str(src),destination=str(dst),files=item['files'],bytes=item['bytes'],sha256_verified=True,source_symlink_preserved=True,file_manifest=str(manifest),file_manifest_sha256=sha(manifest)))
  report['logical_bytes_migrated']+=item['bytes'];report['current']=None;atomic(ROOT/'storage_migration.json',report)
  print(json.dumps(dict(completed=index+1,total=len(plan['candidates']),bytes_migrated=report['logical_bytes_migrated'],source=str(src))),flush=True)
 report.update(status='completed',completed_at_utc=datetime.now(timezone.utc).isoformat())
except BaseException as exc:
 report.update(status='failed',error=repr(exc));raise
finally:
 atomic(ROOT/'storage_migration.json',report)
