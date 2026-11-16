"""Serial local-model authorship requests; no implementation placement."""
from pathlib import Path
import subprocess,sys,json
r=Path('/home/cap/spark-inference-overnight-20261113');run=Path((r/'feedback/prefill-gated-main-run.txt').read_text().strip());p=Path('/home/cap/Projects/spark-inference-engine/benchmark_runs')/run.name
for number in ['038','039','040']:
 assert not (p/f'task-{number}-response.json').exists(), 'Refuse duplicate author request'
 subprocess.run([sys.executable,str(r/'request_prefill_gated_author.py'),number],check=True)
 print('LOCAL_AUTHOR_RESPONSE_RETAINED '+number,flush=True)
