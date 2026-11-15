"""One explicitly authorized 27B trial; global GPU crash hold stays in place."""
from datetime import datetime,timezone
import fcntl,hashlib,json,os,re,subprocess,time
from pathlib import Path
from shutdown_process_probe import python_workers
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');C='spark-prototype-agent';M='spark-prototype-model'
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--prompt',type=int,choices=[256],required=True);parsed=parser.parse_args()
run=parsed.run.resolve();assert run.is_relative_to(R/'runs');locks=[]
plan=json.loads((run/'explicit-post-update-plan.json').read_text());assert plan['user_requested_direct_heavy_trial'] and plan['prompt_tokens']==parsed.prompt==256 and plan['new_tokens']==128
assert plan['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for name in ('controller.lock','benchmark.lock'):
 f=(R/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert (R/'GPU_WORKLOAD_HOLD').exists(),'Global crash hold must remain active'
assert plan['paired_comparison_authorized']
fd=os.open(run/'trial-consumed-once',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
 t=path.with_suffix('.tmp')
 with t.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
 t.replace(path)
def cmd(a):return subprocess.check_output(a,text=True,timeout=30).strip()
def memory():
 d={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  k,_,v=line.partition(':')
  if k in ('MemTotal','MemAvailable','MemFree','Cached','Buffers','SReclaimable','SwapTotal','SwapFree'):d[k]=int(v.split()[0])*1024
 d['swap_used_bytes']=d['SwapTotal']-d['SwapFree'];return d
def inspect(name):return json.loads(cmd(['docker','inspect',name]))[0]
def gpu_probe(label):
 script='import torch,json; torch.cuda.init(); f,t=torch.cuda.mem_get_info(); print(json.dumps({"free_bytes":f,"total_bytes":t,"allocated_bytes":torch.cuda.memory_allocated(),"reserved_bytes":torch.cuda.memory_reserved()}))'
 value=json.loads(cmd(['docker','exec',C,'python3','-I','-c',script]));atomic(run/(label+'.json'),value);return value
def inventory(label):
 value=cmd(['docker','exec',C,'nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'])
 (run/(label+'.txt')).write_text(value+'\n')
 assert not value or 'No running processes' in value,'Existing GPU compute process: '+value
 return value
state={'status':'preflight','started':now(),'boot_id':boot,'scope':'One loaded27B,256 prompt,8/128 outputs,4 alternating AB/BA timed pairs each,warmups,clean release; no profiler','automatic_gpu_resume_held':True,'model_server_loaded':False,'run_directory':str(run),'engine_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}}
def save():
 state['updated']=now();atomic(run/'trial-status.json',state);atomic(R/'feedback/clean-shutdown-status.json',state)
 p=R/'status.json';s=json.loads(p.read_text());s['paired_generation_trial']=state;s.update(status='paired_generation_trial_running' if state['status'] in ('preflight','running','recovery_check') else 'gpu_workload_held_after_clean_shutdown_trial',phase=state['status']);atomic(p,s)
error=None;worker=None;output=P/'benchmark_runs'/run.name;save()
try:
 assert cmd(['systemctl','--user','show','spark-prototype-overnight-20261113.service','--value','-p','MainPID'])=='0'
 assert not inspect(C)['State']['Running'] and not inspect(M)['State']['Running']
 state['host_before']=memory();assert state['host_before']['MemAvailable']>100*1024**3
 subprocess.run(['docker','start',C],check=True,timeout=30,stdout=subprocess.DEVNULL)
 inventory('gpu-processes-before');state['gpu_before']=gpu_probe('gpu-before');assert state['gpu_before']['free_bytes']>80*1024**3
 inventory('gpu-processes-before-model')
 args=['docker','exec','-w','/workspace',C,'timeout','--kill-after=10','900','python3','/workspace/acceptance/check_paired_27b.py',run.name]
 state.update(status='running',command=args);save();print('START_CLEAN_27B '+str(run),flush=True)
 with (run/'27b-worker.log').open('w') as log,(run/'host-memory.jsonl').open('a') as telemetry:
  worker=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);started=time.monotonic();heartbeat=started;swap_start=memory()['swap_used_bytes']
  while worker.poll() is None:
   row={'time':now(),**memory()};row['swap_growth_bytes']=max(0,row['swap_used_bytes']-swap_start)
   telemetry.write(json.dumps(row)+'\n');telemetry.flush();os.fsync(telemetry.fileno())
   if row['MemAvailable']<32*1024**3 or row['swap_growth_bytes']>256*1024**2 or time.monotonic()-started>930:raise RuntimeError('Bounded trial host memory/time guard triggered')
   if time.monotonic()-heartbeat>=5:
    assert not inspect(M)['State']['Running'],'Coding model server started during native trial'
    top=cmd(['docker','top',C,'-eo','pid,ppid,comm,args']);(run/'process-inventory.txt').write_text(top+'\n')
    python_rows=python_workers(top)
    assert len(python_rows)<=1,'More than one Python worker detected'
    state.update(elapsed_seconds=round(time.monotonic()-started,1),host_current=row,python_worker_count=len(python_rows));save();heartbeat=time.monotonic()
   time.sleep(.5)
 state['worker_exit_code']=worker.returncode;assert worker.returncode==0,'Worker failed; inspect retained log'
 data=json.loads((output/'paired-results.json').read_text());assert data['status']=='complete' and data['instance_method_restored']
 assert len(data['rows'])==16
 for count in (8,128):
  rows=[x for x in data['rows'] if x['new_tokens']==count]
  assert len(rows)==8 and len({x['output_sha256'] for x in rows})==1
  for version in ('before','after'):assert len([x for x in rows if x['version']==version])==4
 assert data['released_memory']['cuda_allocated_bytes']<=16*1024**2 and data['released_memory']['cuda_reserved_bytes']<=64*1024**2
 atomic(run/'paired-results.json',data)
 state.update(status='recovery_check',results_file=str(output/'paired-results.json'));save()
 inventory('gpu-processes-after-model');state['gpu_after_worker']=gpu_probe('gpu-after-worker');inventory('gpu-processes-after-probe')
 started=time.monotonic();subprocess.run(['docker','stop','--timeout','10',C],check=True,timeout=20,stdout=subprocess.DEVNULL);state['container_stop_seconds']=time.monotonic()-started
 final=inspect(C)['State'];assert not final['Running'] and final['Pid']==0 and final['ExitCode']==0 and not final['OOMKilled'],final
 state['container_after']={k:final[k] for k in ('Running','Pid','ExitCode','OOMKilled')}
 assert not inspect(M)['State']['Running'];state['host_after']=memory()
 state['host_available_recovered_within_4gib']=state['host_after']['MemAvailable']>=state['host_before']['MemAvailable']-4*1024**3
 assert state['host_available_recovered_within_4gib'],'Host available memory did not recover; no further model may start'
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
 assert state['engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
 state.update(status='complete',scope_result='Same-loaded-model paired27B measurement and explicit teardown passed; speedup decision pending analysis')
 print('CLEAN_SHUTDOWN_TRIAL_COMPLETE',flush=True)
except BaseException as e:
 error=e;state.update(status='failed',error={'type':type(e).__name__,'message':str(e)});raise
finally:
 if worker is not None and worker.poll() is None:
  subprocess.run(['docker','exec',C,'pkill','-TERM','-f','/workspace/acceptance/check_paired_27b.py'],check=False,timeout=10)
  try:worker.wait(timeout=10)
  except subprocess.TimeoutExpired:state['forced_worker_cleanup']=True
 subprocess.run(['docker','stop','--timeout','10',C],check=False,timeout=20,stdout=subprocess.DEVNULL)
 if worker is not None:state['worker_exit_code']=worker.poll()
 state['finished']=now();save()
