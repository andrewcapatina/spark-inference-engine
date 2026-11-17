"""Read-only final containment evidence after all validation stages complete."""
from pathlib import Path
import json,subprocess
R=Path('/home/cap/spark-inference-overnight-20261113');run=Path((R/'feedback/fp8-pair-main-run.txt').read_text().strip());assert json.loads((run/'remaining-validation-status.json').read_text())['status']=='complete'
def cmd(a):return subprocess.check_output(a,text=True,timeout=30).strip()
names=['spark-prototype-agent','spark-prototype-model','spark-prototype-vllm-nvfp4','spark-prototype-quant-agent'];containers={}
for name in names:
 value=json.loads(cmd(['docker','inspect',name]))[0];s=value['State'];containers[name]={**{k:s[k] for k in ['Running','Pid','ExitCode','OOMKilled']},'image_id':value['Image']};assert not s['Running'] and s['Pid']==0 and s['ExitCode']==0 and not s['OOMKilled']
service=cmd(['systemctl','--user','show','spark-prototype-overnight-20261113.service','--value','-p','MainPID']);assert service=='0'
gpu=cmd(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']);assert not gpu or 'No running processes' in gpu
boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();telemetry={}
roots=[run]+[Path(p.read_text().strip()) for p in (R/'feedback').glob('fp8-pair-*-run.txt') if p.name!='fp8-pair-main-run.txt']
for root in roots:
 for p in root.glob('*host-memory.jsonl'):
  rows=[json.loads(line) for line in p.read_text().splitlines() if line.strip()];assert rows
  telemetry[str(p)]={'min_available_bytes':min(x['MemAvailable'] for x in rows),'max_per_worker_swap_growth_bytes':max(x.get('swap_growth_bytes',0) for x in rows)};assert telemetry[str(p)]['max_per_worker_swap_growth_bytes']<=256*1024**2;assert telemetry[str(p)]['min_available_bytes']>=32*1024**3
 if (root/'trial-status.json').exists():
  trial=json.loads((root/'trial-status.json').read_text());assert trial['boot_id']==boot
  assert trial['status']=='complete',str(root)
  final=trial['container_after'];assert not final['Running'] and final['Pid']==0 and final['ExitCode']==0 and not final['OOMKilled']
assert json.loads((run/'protected-fp8-pair-regression-status.json').read_text())['boot_id']==boot
result={'status':'clean','containers':containers,'service_main_pid':int(service),'gpu_compute_processes':gpu,'boot_id':boot,'same_boot_all_stages':True,'global_gpu_hold_retained':(R/'GPU_WORKLOAD_HOLD').exists(),'legacy_stop_retained':(R/'STOP').exists(),'large_workload_hold_retained':(R/'LARGE_WORKLOAD_HOLD').exists(),'telemetry':telemetry,'maximum_per_worker_swap_growth_bytes':max(x['max_per_worker_swap_growth_bytes'] for x in telemetry.values())};assert result['global_gpu_hold_retained'] and result['legacy_stop_retained'] and result['large_workload_hold_retained'];(run/'cleanup-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'clean','maximum_per_worker_swap_growth_bytes':result['maximum_per_worker_swap_growth_bytes']}))
