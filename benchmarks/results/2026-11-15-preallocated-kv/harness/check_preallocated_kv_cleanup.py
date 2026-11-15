"""Read-only final containment evidence after all validation stages complete."""
from pathlib import Path
import json,subprocess
R=Path('/home/cap/spark-inference-overnight-20261113');run=Path((R/'feedback/preallocated-kv-main-run.txt').read_text().strip());assert json.loads((run/'manifest.json').read_text())['status']=='retained_not_adopted_no_material_gain'
def cmd(a):return subprocess.check_output(a,text=True,timeout=30).strip()
names=['spark-prototype-agent','spark-prototype-model','spark-prototype-vllm-nvfp4','spark-prototype-quant-agent'];containers={}
for name in names:
 s=json.loads(cmd(['docker','inspect',name]))[0]['State'];containers[name]={k:s[k] for k in ['Running','Pid','ExitCode','OOMKilled']};assert not s['Running'] and s['Pid']==0 and s['ExitCode']==0 and not s['OOMKilled']
service=cmd(['systemctl','--user','show','spark-prototype-overnight-20261113.service','--value','-p','MainPID']);assert service=='0'
gpu=cmd(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']);assert not gpu or 'No running processes' in gpu
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();telemetry={}
roots=[run]+[Path((R/f'feedback/preallocated-kv-{kind}-run.txt').read_text().strip()) for kind in ['small','decode','ab','profile']]
for root in roots:
 for p in root.glob('*host-memory.jsonl'):
  rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()];assert rows
  telemetry[str(p)]={'min_available_bytes':min(x['MemAvailable'] for x in rows),'max_per_worker_swap_growth_bytes':max(x.get('swap_growth_bytes',0) for x in rows)};assert telemetry[str(p)]['max_per_worker_swap_growth_bytes']<=256*1024**2;assert telemetry[str(p)]['min_available_bytes']>=32*1024**3
 if (root/'trial-status.json').exists():assert json.loads((root/'trial-status.json').read_text())['boot_id']==boot
for path in run.glob('local-author-*-status.json'):assert json.loads(path.read_text())['boot_id']==boot
import hashlib
P=Path('/home/cap/Projects/spark-inference-engine');manifest=json.loads((run/'manifest.json').read_text());assert manifest['engine_before']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
result={'status':'clean','accepted_engine_restored_byte_exact':True,'containers':containers,'service_main_pid':int(service),'gpu_compute_processes':gpu,'boot_id':boot,'same_boot_all_stages':True,'global_gpu_hold_retained':(R/'GPU_WORKLOAD_HOLD').exists(),'telemetry':telemetry,'maximum_per_worker_swap_growth_bytes':max(x['max_per_worker_swap_growth_bytes'] for x in telemetry.values())};assert result['global_gpu_hold_retained'];(run/'cleanup-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'clean','maximum_per_worker_swap_growth_bytes':result['maximum_per_worker_swap_growth_bytes']}))
