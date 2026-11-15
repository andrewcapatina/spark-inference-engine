"""Place reviewed, exact local-model source fragments; no implementation repair."""
from pathlib import Path
import ast,json,hashlib
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/preallocated-kv-main-run.txt').read_text().strip());out=P/'benchmark_runs'/run.name;manifest=json.loads((run/'manifest.json').read_text());assert manifest['current_engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
def candidate(n):
 response=json.loads((out/f'task-{n}-response.json').read_text());assert response['done'] and response['done_reason']=='stop';s=response['message']['content'];assert s==(out/f'task-{n}-candidate.txt').read_text()
 if s.startswith('```python\n') and s.rstrip().endswith('```'):s=s[len('```python\n'):s.rfind('```')]
 return s.rstrip('\n')+'\n'
foundation=(out/'task-024-retained-foundation.py.txt').read_text();core=foundation+candidate('025');assert core==(out/'task-025-private-module.py.txt').read_text();review25=json.loads((out/'task-025-review.json').read_text());assert review25['assembled_private_module_sha256']==hashlib.sha256(core.encode()).hexdigest();wrapper=candidate('026');review26=json.loads((out/'task-026-review.json').read_text());assert review26['decision']=='allow_exact_placement_for_testing' and review26['candidate_source_sha256']==hashlib.sha256(wrapper.encode()).hexdigest();assert review26['private_core_sha256']==hashlib.sha256(core.encode()).hexdigest()
items=[('qwen_kv_cache_impl.py',core,['024','025']),('qwen_kv_cache.py',wrapper,['026'])]
for name,source,tasks in items:
 assert not (P/'spark_engine'/name).exists();compile(source,name,'exec');assert sum(bool(x.strip()) for x in source.splitlines())<=50
for name,source,tasks in items:
 (out/(name+'.placed.txt')).write_text(source);(P/'spark_engine'/name).write_text(source);manifest['tasks'].append({'tasks':tasks,'target':'spark_engine/'+name,'implementation_author':'local Qwen only','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'transformations':'Verbatim retained local024 prefix + local025 module for private core; verbatim local026 public wrapper; only outer Markdown/trailing newline normalized','validation':'pending helper contracts, not integrated'})
manifest['current_engine_sha256']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};manifest['status']='standalone_helper_placed_for_testing';(run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['tasks'],indent=2))
