"""Read-only final experiment containment after rejected-candidate restoration."""
from pathlib import Path
import json,subprocess
R=Path('/home/cap/spark-inference-overnight-20261113');run=Path((R/'feedback/prefill-keys-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());assert m['status']=='not_adopted_small_real_prefill_gain_baseline_restored'
def cmd(a):return subprocess.check_output(a,text=True,timeout=30).strip()
containers={}
for name in ['spark-prototype-agent','spark-prototype-model','spark-prototype-vllm-nvfp4','spark-prototype-quant-agent']:
 v=json.loads(cmd(['docker','inspect',name]))[0];s=v['State'];assert not s['Running'] and s['Pid']==0 and s['ExitCode']==0 and not s['OOMKilled'];containers[name]={k:s[k] for k in ['Running','Pid','ExitCode','OOMKilled']}
assert cmd(['systemctl','--user','show','spark-prototype-overnight-20261113.service','--value','-p','MainPID'])=='0';gpu=cmd(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']);assert not gpu or 'No running processes' in gpu
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();assert boot==m['boot_id'];stages=[];telemetry=[]
for path in sorted((R/'runs').glob('prefill-keys-*')):
 s=json.loads((path/'trial-status.json').read_text());assert s['boot_id']==boot;stages.append({'run':path.name,'status':s['status'],'error':s.get('error')})
 for p in path.glob('*host-memory.jsonl'):
  rows=[json.loads(x) for x in p.read_text().splitlines()];assert rows;record={'file':str(p),'min_available_bytes':min(x['MemAvailable'] for x in rows),'max_new_swap_bytes':max(x.get('swap_growth_bytes',0) for x in rows)};assert record['min_available_bytes']>=32*1024**3 and record['max_new_swap_bytes']<=256*1024**2;telemetry.append(record)
holds={name:(R/name).exists() for name in ['STOP','GPU_WORKLOAD_HOLD','LARGE_WORKLOAD_HOLD']};assert all(holds.values());d={'status':'clean','containers':containers,'service_main_pid':0,'gpu_compute_processes':gpu,'boot_id':boot,'same_boot':True,'holds_retained':holds,'stages':stages,'telemetry':telemetry};(run/'cleanup-check.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':'clean','stages':len(stages),'max_new_swap_bytes':max(x['max_new_swap_bytes'] for x in telemetry)}))
