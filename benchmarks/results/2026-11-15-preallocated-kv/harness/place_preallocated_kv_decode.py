"""Mechanical placement of retained local-Qwen QwenAttention.decode source only."""
from pathlib import Path
import ast,json,hashlib,difflib,sys
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/preallocated-kv-main-run.txt').read_text().strip());number=sys.argv[1];out=P/'benchmark_runs'/run.name;response_path=out/f'task-{number}-response.json';response=json.loads(response_path.read_text());assert response['done'] and response['done_reason']=='stop';source=response['message']['content'];assert source==(out/f'task-{number}-candidate.txt').read_text()
if source.startswith('```python\n') and source.rstrip().endswith('```'):source=source[len('```python\n'):source.rfind('```')]
source=source.rstrip('\n')+'\n';review=json.loads((out/f'task-{number}-review.json').read_text());assert review['decision']=='allow_exact_placement_for_testing' and review['candidate_source_sha256']==hashlib.sha256(source.encode()).hexdigest();manifest=json.loads((run/'manifest.json').read_text())
for name,digest in manifest['current_engine_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
path=P/'spark_engine/qwen_layers.py';before=path.read_text();tree=ast.parse(before);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenAttention');old=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decode');parsed=ast.parse('class X:\n'+source);assert len(parsed.body)==1 and len(parsed.body[0].body)==1;candidate=parsed.body[0].body[0];assert isinstance(candidate,ast.FunctionDef) and candidate.name=='decode' and ast.dump(candidate.args)==ast.dump(old.args)
lines=before.splitlines(keepends=True);placed=''.join(lines[:old.lineno-1])+source+''.join(lines[old.end_lineno:]);newtree=ast.parse(placed);assert len(tree.body)==len(newtree.body)
for original,new in zip(tree.body,newtree.body):
 if isinstance(original,ast.ClassDef) and original.name=='QwenAttention':
  assert len(original.body)==len(new.body)
  for a,b in zip(original.body,new.body):
   if isinstance(a,ast.FunctionDef) and a.name=='decode':continue
   assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 else:assert ast.dump(original,include_attributes=False)==ast.dump(new,include_attributes=False)
assert sum(x.startswith('+ ') for x in difflib.ndiff(before.splitlines(),placed.splitlines()))<=50;compile(placed,str(path),'exec');(out/f'task-{number}-previous.py.txt').write_text(before);(out/f'task-{number}-placed.py.txt').write_text(placed);path.write_text(placed);manifest['tasks'].append({'task':number,'target':'spark_engine/qwen_layers.py','function':'QwenAttention.decode','implementation_author':'local Qwen only','source_sha256':hashlib.sha256(placed.encode()).hexdigest(),'response_sha256':hashlib.sha256(response_path.read_bytes()).hexdigest(),'transformations':'Outer Markdown fence and trailing newline only; candidate placed verbatim','validation':'pending'});manifest['current_engine_sha256']={str(x.relative_to(P)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (P/'spark_engine').glob('*.py')};(run/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['tasks'][-1]))
