import json,os,signal,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
status=root/'runs/lehami/lehami_seed0/a/segments/0001/status.json'
record=dict(status='waiting_for_first_checkpoint',allocation_job_id='48044',methods=['lewm','hjepa'],formal_jobs_unchanged=['48044','48045','48046'])
def write():
 record['updated_at_utc']=datetime.now(timezone.utc).isoformat();p=root/'profile_bridge.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2)+'\n');t.replace(p)
write();deadline=time.monotonic()+600
while True:
 if status.exists():
  s=json.loads(status.read_text())
  if s['status']=='failed':raise RuntimeError(s)
  if s.get('completed_steps',0)>=250:break
 if time.monotonic()>deadline:raise TimeoutError('First checkpoint not reached; no process stopped')
 time.sleep(2)
task=str(root/'runs/lehami/lehami_seed0/a_task.json')
workers=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().split(b'\0')
  if b'--worker' in args and task.encode() in args:workers.append(int(p.name))
 except OSError:continue
if len(workers)!=1:raise RuntimeError('Expected exactly one owned formal worker: '+str(workers))
pid=workers[0];record.update(status='profiling_on_allocated_gpu',paused_worker_pid=pid,paused_after_step=s['completed_steps'],paused_at_utc=datetime.now(timezone.utc).isoformat());write()
os.kill(pid,signal.SIGSTOP)
try:
 time.sleep(1)
 if '\nState:\tT' not in Path(f'/proc/{pid}/status').read_text():raise RuntimeError('Formal worker did not stop')
 command=['srun','--jobid=48044','--overlap','--ntasks=1','--cpus-per-task=4','--gres=gpu:l40:1','--time=00:15:00','bash',str(root/'profile_pending.sh')]
 record['command']=command;write()
 result=subprocess.run(command,timeout=850)
 record['profile_exit_code']=result.returncode
 record['status']='completed' if result.returncode==0 else 'profile_failed'
except BaseException as exc:
 record.update(status='profile_failed',error=repr(exc));raise
finally:
 os.kill(pid,signal.SIGCONT)
 record.update(formal_worker_resumed=True,resumed_at_utc=datetime.now(timezone.utc).isoformat());write()
