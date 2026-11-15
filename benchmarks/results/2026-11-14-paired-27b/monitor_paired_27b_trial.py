"""Read-only health sampling for the explicitly requested post-update heavy test."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,time
R=Path('/home/cap/spark-inference-overnight-20261113');run=Path((R/'feedback/paired-27b-run.txt').read_text().strip())
with (run/'gpu-thermal-health.jsonl').open('a') as f:
 while True:
  p=run/'trial-status.json'
  if not p.exists():time.sleep(.5);continue
  state=json.loads(p.read_text())
  if state['status'] in ('complete','failed'):break
  row={'time':datetime.now(timezone.utc).isoformat(),'thermal_zones':[],'memory_counters':{}}
  for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
   try:row['thermal_zones'].append({'zone':zone.name,'type':(zone/'type').read_text().strip(),'temp_c':int((zone/'temp').read_text())/1000})
   except (OSError,ValueError):pass
  for line in Path('/proc/meminfo').read_text().splitlines():
   k,_,v=line.partition(':')
   if k in ('MemAvailable','MemFree','Cached','SwapFree','GPUActive','GPUReclaim'):row['memory_counters'][k]=int(v.split()[0])*1024
  try:
   row['memory_pressure']=Path('/proc/pressure/memory').read_text().strip()
   running=subprocess.check_output(['docker','inspect','--format','{{.State.Running}}','spark-prototype-agent'],text=True,timeout=5).strip()=='true'
   if running:
    values=subprocess.check_output(['docker','exec','spark-prototype-agent','nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5).strip().split(',')
    row.update({k:None if 'N/A' in v else float(v.strip()) for k,v in zip(['gpu_temperature_c','gpu_power_w','sm_clock_mhz','memory_clock_mhz','gpu_utilization_pct'],values)})
   else:row['container_running']=False
  except Exception as e:row['query_error']=str(e)
  f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno());time.sleep(2)
print('POST_UPDATE_HEALTH_MONITOR_COMPLETE',flush=True)
