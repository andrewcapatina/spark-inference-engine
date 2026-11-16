"""Experiment-only containment and durable monitoring; no inference code."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

R=Path('/home/cap/spark-inference-overnight-20261113')
P=Path('/home/cap/Projects/spark-inference-engine')
C='spark-prototype-agent'
M='spark-prototype-model'

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,value):
    temp=path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
def command(args):return subprocess.check_output(args,text=True,timeout=45).strip()
def mem():
    result={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k,_,v=line.partition(':')
        if k in ('MemAvailable','SwapTotal','SwapFree'):result[k]=int(v.split()[0])*1024
    result['swap_used_bytes']=result['SwapTotal']-result['SwapFree']
    return result

def validate_model_isolation(label,model_server_running,python_processes):
    if label=='qwen-protected':
        raise RuntimeError('Aggregate multi-checkpoint regression disabled; use isolated real-checkpoint processes')
    if model_server_running and label!='local-qwen-author':
        raise RuntimeError('Coding/reference model server active; refusing native inference')
    if python_processes:
        raise RuntimeError('Existing Python worker in experiment container; refusing overlap: '+str(python_processes))

class Runtime:
    def __init__(self,run,phase,swap_policy="absolute"):
        assert json.loads((Path(run)/'manifest.json').read_text())['user_authorized_nvfp4_update'], 'Dedicated authorized NVFP4 update only'
        self.swap_policy=swap_policy
        self.run=Path(run);self.phase=phase;self.locks=[]
        for name in ('controller.lock','benchmark.lock'):
            f=(R/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);self.locks.append(f)
        assert command(['systemctl','--user','show','spark-prototype-overnight-20261113.service','--value','-p','MainPID'])=='0'
        assert command(['docker','inspect','--format','{{.State.Running}}',M])=='false'
        assert command(['docker','inspect','--format','{{.State.Running}}','spark-prototype-vllm-nvfp4'])=='false'
        assert command(['docker','inspect','--format','{{.State.Running}}','spark-prototype-quant-agent'])=='false','Native quantized CLI is active; refusing overlap'
        if command(['docker','inspect','--format','{{.State.Running}}',C])=='true':
            probe='''from pathlib import Path
import json
rows=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=p.read_bytes().decode().split('\\0')
  if ('-m' in a and any(x in a for x in ('spark_engine','pytest'))) or any('benchmark_qwen.py' in x for x in a):rows.append(p.parent.name)
 except (OSError,UnicodeError):pass
print(json.dumps(rows))'''
            assert json.loads(command(['docker','exec',C,'python3','-I','-c',probe]))==[], 'Active inference session; refusing competition'
        self.state={'status':'running','phase':phase,'started':now(),'run_directory':str(self.run),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'completed_commands':[], 'watchdog':{'swap_policy':swap_policy,'swap_limit_bytes':256*1024**2,'min_available_gib':32 if swap_policy=='growth' else 24}}
        self.save()
        assert mem()['MemAvailable']>80*1024**3,'Insufficient starting memory headroom'
        subprocess.run(['docker','start',C],check=True,timeout=45,stdout=subprocess.DEVNULL)
    def save(self):
        self.state['updated']=now()
        atomic(self.run/(self.phase+'-status.json'),self.state)
        atomic(R/'feedback/nvfp4-regression-status.json',self.state)
        p=R/'status.json';state=json.loads(p.read_text())
        state['nvfp4_regression']={**self.state,'implementation_author':'local Qwen only'}
        atomic(p,state)
    def execute(self,label,args,limit=1200):
        probe="""from pathlib import Path
import json,os
rows=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  a=p.read_bytes().decode().split('\\0')
  if p.parent.name!=str(os.getpid()) and a and 'python' in Path(a[0]).name.lower():rows.append({'pid':p.parent.name,'argv':a[:5]})
 except (OSError,UnicodeError):pass
print(json.dumps(rows))"""
        inventory=json.loads(command(['docker','exec',C,'python3','-I','-c',probe]))
        model_running=command(['docker','inspect','--format','{{.State.Running}}',M])=='true'
        validate_model_isolation(label,model_running,inventory)
        self.state['preflight']={'label':label,'model_server_running':model_running,'python_workers':inventory}
        if label.startswith(('before-9b-','before-27b-','after-9b-','after-27b-','qwen-real')):
            subprocess.run(['docker','stop','--timeout','5',C],check=True,timeout=30,stdout=subprocess.DEVNULL)
            subprocess.run(['docker','start',C],check=True,timeout=30,stdout=subprocess.DEVNULL)
            self.state['preflight']['fresh_container_for_checkpoint']=True
        self.state.update(current_command=label,elapsed_seconds=0);self.save()
        print('START '+label,flush=True)
        with (self.run/(label+'.log')).open('w') as log,(self.run/(label+'-host-memory.jsonl')).open('a') as telemetry:
            process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
            started=time.monotonic();saved=started
            initial_swap=mem()["swap_used_bytes"]
            self.state["command_initial_swap_bytes"]=initial_swap
            while process.poll() is None:
                row={'time':now(),**mem()};row['swap_growth_bytes']=max(0,row['swap_used_bytes']-initial_swap);telemetry.write(json.dumps(row)+'\n');telemetry.flush();os.fsync(telemetry.fileno())
                swap_pressure=row['swap_growth_bytes'] if self.swap_policy=='growth' else row['swap_used_bytes']
                floor=32 if self.swap_policy=='growth' else 24
                if row['MemAvailable']<floor*1024**3 or swap_pressure>256*1024**2 or time.monotonic()-started>limit+30:
                    self.state['watchdog_abort']=row;self.save()
                    subprocess.run(['docker','kill',C,M],timeout=30,check=False)
                    process.wait(timeout=30)
                    raise RuntimeError('Memory/time watchdog aborted '+label)
                if time.monotonic()-saved>=5:
                    self.state.update(elapsed_seconds=round(time.monotonic()-started,1),host_memory=row);self.save();saved=time.monotonic()
                time.sleep(.5)
            code=process.returncode
        self.state['completed_commands'].append({'label':label,'exit_code':code,'elapsed_seconds':round(time.monotonic()-started,2)})
        self.save()
        if code:raise RuntimeError(label+' failed, exit '+str(code)+'; log retained')
        print('PASS '+label,flush=True)
    def close(self,error=None):
        subprocess.run(['docker','stop','--timeout','5',C,M],check=False,timeout=30,stdout=subprocess.DEVNULL)
        self.state.update(status='failed' if error else 'complete',finished=now(),error=str(error) if error else None)
        assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==self.state['boot_id'],'Host rebooted'
        self.save()
        for stream in self.locks:stream.close()
