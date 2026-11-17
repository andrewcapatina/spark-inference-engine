from pathlib import Path
import json,hashlib
from nvfp4_update_runtime import Runtime,R,P,C
run=Path((R/'feedback/fp8-prep-main-run.txt').read_text().strip());rt=Runtime(run,'fp8-prepare-legacy-portability',swap_policy='growth');error=None
try:
 expected=json.loads((run/'manifest.json').read_text())['current_engine_sha256'];assert expected=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
 rt.execute('fp8-prepare-legacy-exact',['docker','exec','-w','/workspace','-e','TRITON_CACHE_DIR=/workspace/benchmark_runs/'+run.name+'/triton-legacy-cache',C,'timeout','--kill-after=10','600','python3','/workspace/acceptance/check_fp8_prepare_legacy.py',run.name],600)
 data=json.loads((P/'benchmark_runs'/run.name/'legacy-fp8-prepare-results/results.json').read_text());assert data['status']=='complete';(run/'legacy-portability-results.json').write_text(json.dumps(data,indent=2)+'\n')
except BaseException as e:error=e;raise
finally:rt.close(error)
