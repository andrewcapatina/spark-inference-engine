"""One explicitly authorized 27B trial; global GPU crash hold stays in place."""
from datetime import datetime,timezone
import fcntl,hashlib,json,os,re,subprocess,time
from pathlib import Path
from shutdown_process_probe import python_workers
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');C='spark-prototype-vllm-nvfp4';M='spark-prototype-model'
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parsed=parser.parse_args()
run=parsed.run.resolve();assert run.is_relative_to(R/'runs');locks=[]
plan=json.loads((run/'explicit-post-update-plan.json').read_text());assert plan['profiling_authorized'] and plan['mode'] in ('candidate',)
assert plan['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
for name in ('controller.lock','benchmark.lock'):
 f=(R/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
assert (R/'GPU_WORKLOAD_HOLD').exists(),'Global crash hold must remain active'
assert plan['one_model_at_a_time']
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
state={'status':'preflight','started':now(),'boot_id':boot,'scope':'CPU-only checkpoint metadata and scalar scale audit; no model weights loaded, not inference validation','automatic_gpu_resume_held':True,'model_server_loaded':False,'run_directory':str(run),'engine_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}}
def save():
 state['updated']=now();atomic(run/'trial-status.json',state);atomic(R/'feedback/clean-shutdown-status.json',state)
 p=R/'status.json';s=json.loads(p.read_text());s['gdn_kernel_trial']=state;s.update(status='gdn_kernel_trial_running' if state['status'] in ('preflight','running','recovery_check') else 'gpu_workload_held_after_clean_shutdown_trial',phase=state['status']);atomic(p,s)
error=None;worker=None;output=R/'feedback/nvfp4-reference'/run.name;save()
try:
 assert cmd(['systemctl','--user','show','spark-prototype-overnight-20261113.service','--value','-p','MainPID'])=='0'
 assert all(not inspect(n)['State']['Running'] for n in (C,M,'spark-prototype-agent','spark-prototype-quant-agent'))
 state['host_before']=memory();assert state['host_before']['MemAvailable']>100*1024**3
 subprocess.run(['docker','start',C],check=True,timeout=30,stdout=subprocess.DEVNULL)
 inventory('gpu-processes-before');state['gpu_before']=gpu_probe('gpu-before')
 if state['gpu_before']['free_bytes']<=56*1024**3:
  source=R/'feedback/nvfp4-reference'/plan['inactive_trace_run'];assert source.resolve().is_relative_to(R/'feedback/nvfp4-reference')
  assert json.loads((source/'results.json').read_text())['status']=='complete'
  state['inactive_trace_cache_advice']=[]
  for path in source.glob('*-native-trace.json'):
   before=path.stat();fd=os.open(path,os.O_RDONLY)
   try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
   finally:os.close(fd)
   after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
   state['inactive_trace_cache_advice'].append({'path':str(path),'bytes':before.st_size,'data_unchanged':True})
  state['gpu_after_trace_cache_advice']=gpu_probe('gpu-after-trace-cache-advice')
 if (state.get('gpu_after_trace_cache_advice') or state['gpu_before'])['free_bytes']<=56*1024**3:
  inventory('gpu-before-inactive-weight-cache-advice')
  state['inactive_weight_cache_advice']=[];hinted_bytes=0;recovered=False
  weights=Path('/mnt/nas-aimodels/huggingface/unsloth/Qwen3.8-27B-NVFP4')
  for path in sorted(weights.glob('*.safetensors')):
   before=path.stat()
   for offset in range(0,before.st_size,2*1024**3):
    if hinted_bytes>=24*1024**3:break
    length=min(2*1024**3,before.st_size-offset);fd=os.open(path,os.O_RDONLY)
    try:os.posix_fadvise(fd,offset,length,os.POSIX_FADV_DONTNEED)
    finally:os.close(fd)
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    hinted_bytes+=length;row={'path':str(path),'offset':offset,'bytes':length,'data_unchanged':True};state['inactive_weight_cache_advice'].append(row);state['gpu_after_weight_cache_advice']=gpu_probe('gpu-after-weight-cache-advice-'+str(len(state['inactive_weight_cache_advice'])))
    if state['gpu_after_weight_cache_advice']['free_bytes']>58*1024**3:recovered=True;break
   if recovered or hinted_bytes>=24*1024**3:break
 if (state.get('gpu_after_weight_cache_advice') or state.get('gpu_after_trace_cache_advice') or state['gpu_before'])['free_bytes']<=56*1024**3:
  assert not inspect(M)['State']['Running'];inventory('gpu-before-inactive-author-cache-advice')
  identity=json.loads((R/'model-runtime.json').read_text());nas=Path('/mnt/nas-aimodels/huggingface');path=nas/identity['source_repo']/identity['source_file'];assert path.resolve().is_relative_to(nas.resolve());before=path.stat();state['inactive_author_cache_advice']=[]
  for offset in range(0,min(before.st_size,8*1024**3),2*1024**3):
   length=min(2*1024**3,before.st_size-offset);fd=os.open(path,os.O_RDONLY)
   try:os.posix_fadvise(fd,offset,length,os.POSIX_FADV_DONTNEED)
   finally:os.close(fd)
   after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns);state['inactive_author_cache_advice'].append({'path':str(path),'offset':offset,'bytes':length,'data_unchanged':True});state['gpu_after_author_cache_advice']=gpu_probe('gpu-after-author-cache-advice-'+str(len(state['inactive_author_cache_advice'])))
   if state['gpu_after_author_cache_advice']['free_bytes']>58*1024**3:break
 assert (state.get('gpu_after_author_cache_advice') or state.get('gpu_after_weight_cache_advice') or state.get('gpu_after_trace_cache_advice') or state['gpu_before'])['free_bytes']>56*1024**3,'Preflight headroom did not recover; no model load allowed'
 inventory('gpu-processes-before-model')
 args=['docker','exec','-w','/workspace',C,'timeout','--kill-after=10','1800','python3','/harness/audit_mlp_checkpoint.py',run.name,plan['mode']]
 state.update(status='running',command=args);save();print('START_CLEAN_27B '+str(run),flush=True)
 with (run/'27b-worker.log').open('w') as log,(run/'host-memory.jsonl').open('a') as telemetry:
  worker=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);started=time.monotonic();heartbeat=started;swap_start=memory()['swap_used_bytes']
  while worker.poll() is None:
   row={'time':now(),**memory()};row['swap_growth_bytes']=max(0,row['swap_used_bytes']-swap_start)
   telemetry.write(json.dumps(row)+'\n');telemetry.flush();os.fsync(telemetry.fileno())
   if row['MemAvailable']<32*1024**3 or row['swap_growth_bytes']>256*1024**2 or time.monotonic()-started>1830:raise RuntimeError('Bounded trial host memory/time guard triggered')
   if time.monotonic()-heartbeat>=5:
    assert not inspect(M)['State']['Running'],'Coding model server started during native trial'
    top=cmd(['docker','top',C,'-eo','pid,ppid,comm,args']);(run/'process-inventory.txt').write_text(top+'\n')
    python_rows=python_workers(top)
    assert not inspect('spark-prototype-agent')['State']['Running'],'Native model container started during reference'
    assert not inspect('spark-prototype-quant-agent')['State']['Running'],'Quantized CLI started during trial'
    state.update(elapsed_seconds=round(time.monotonic()-started,1),host_current=row,python_worker_count=len(python_rows));save();heartbeat=time.monotonic()
   time.sleep(.5)
 state['worker_exit_code']=worker.returncode;assert worker.returncode==0,'Worker failed; inspect retained log'
 data=json.loads((output/'results.json').read_text());assert data['status']=='complete' and data['mode']==plan['mode']
 atomic(run/'diagnostic-results.json',data)
 state.update(status='recovery_check',results_file=str(output/'results.json'));save()
 inventory('gpu-processes-after-model');state['gpu_after_worker']=gpu_probe('gpu-after-worker');inventory('gpu-processes-after-probe')
 started=time.monotonic();subprocess.run(['docker','stop','--timeout','10',C],check=True,timeout=20,stdout=subprocess.DEVNULL);state['container_stop_seconds']=time.monotonic()-started
 final=inspect(C)['State'];assert not final['Running'] and final['Pid']==0 and final['ExitCode']==0 and not final['OOMKilled'],final
 state['container_after']={k:final[k] for k in ('Running','Pid','ExitCode','OOMKilled')}
 assert not inspect(M)['State']['Running'];state['host_after']=memory()
 state['host_available_recovered_within_4gib']=state['host_after']['MemAvailable']>=state['host_before']['MemAvailable']-4*1024**3
 assert state['host_available_recovered_within_4gib'],'Host available memory did not recover; no further model may start'
 assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==boot
 assert state['engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
 state.update(status='complete',scope_result='Diagnostic worker completed and teardown verified; numerical analysis remains separate')
 print('CLEAN_SHUTDOWN_TRIAL_COMPLETE',flush=True)
except BaseException as e:
 error=e;state.update(status='failed',error={'type':type(e).__name__,'message':str(e)});raise
finally:
 if worker is not None and worker.poll() is None:
  subprocess.run(['docker','exec',C,'pkill','-TERM','-f','/harness/audit_mlp_checkpoint.py',run.name,plan['mode']],check=False,timeout=10)
  try:worker.wait(timeout=10)
  except subprocess.TimeoutExpired:state['forced_worker_cleanup']=True
 subprocess.run(['docker','stop','--timeout','10',C],check=False,timeout=20,stdout=subprocess.DEVNULL)
 if worker is not None:state['worker_exit_code']=worker.poll()
 state['finished']=now();save()
