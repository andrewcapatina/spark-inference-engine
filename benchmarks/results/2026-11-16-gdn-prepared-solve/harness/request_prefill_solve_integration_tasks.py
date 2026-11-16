"""Serial local-model source requests only; no engine implementation placement."""
from pathlib import Path
import subprocess,sys
r=Path('/home/cap/spark-inference-overnight-20261113');run=Path((r/'feedback/prefill-solve-main-run.txt').read_text().strip());out=Path('/home/cap/Projects/spark-inference-engine/benchmark_runs')/run.name
for number in ['048','049']:
 assert not (out/f'task-{number}-response.json').exists(),'Refuse duplicate author request'
 subprocess.run([sys.executable,str(r/'request_prefill_solve_author.py'),number],check=True);print('LOCAL_AUTHOR_RESPONSE_RETAINED '+number,flush=True)
