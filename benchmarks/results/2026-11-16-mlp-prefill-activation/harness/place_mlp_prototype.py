"""Copy local-model source verbatim after removing Markdown delimiters only."""
from pathlib import Path
import ast,sys,json,hashlib
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');main=Path((R/'feedback/mlp-main-run.txt').read_text().strip());root=P/'benchmark_runs'/main.name
number=sys.argv[1];target={'057':'nv_pair.py','058':'fp_pair.py','059':'activation.py','060':'activation.py'}[number];raw=(root/f'task-{number}-candidate.txt').read_text();lines=raw.strip().splitlines()
if lines and lines[0].strip() in ['```python','```py','```']:
 assert lines[-1].strip()=='```';body='\n'.join(lines[1:-1])+'\n'
else:body=raw
nodes=ast.parse(body)
for n in ast.walk(nodes):
 if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='torch':assert n.attr not in ['is_cuda','inference_mode_enabled'],n.attr
 if isinstance(n,(ast.Import,ast.ImportFrom)):
  names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module or '']
  assert all(any(name==p or name.startswith(p+'.') for p in ['torch','triton','flashinfer','spark_engine']) for name in names),names
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name):assert n.func.id not in ['exec','eval','open','compile','__import__']
assert any(isinstance(n,ast.FunctionDef) and n.name==('silu_mul' if target=='activation.py' else '_make_pair_projector') for n in nodes.body)
if target=='activation.py':assert any(isinstance(n,ast.keyword) and n.arg=='enable_fp_fusion' and isinstance(n.value,ast.Constant) and n.value.value is False for n in ast.walk(nodes))
(root/target).write_text(body);record={'status':'placed_for_isolated_tests_only','implementation_author':'local Qwen','task':number,'target':str(root/target),'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),'body_sha256':hashlib.sha256(body.encode()).hexdigest(),'only_change':'Markdown delimiter removal; Python body preserved verbatim','engine_writes':False};(main/f'task-{number}-placement.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
