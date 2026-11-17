"""Place only reviewed local-Qwen source; AST guard protects all other definitions."""
from pathlib import Path
import ast,json,hashlib,textwrap,sys
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');main=Path((R/'feedback/fp8-pair-main-run.txt').read_text().strip());root=P/'benchmark_runs'/main.name;m=json.loads((main/'manifest.json').read_text())
def body(raw):
 lines=raw.strip().splitlines()
 return '\n'.join(lines[1:-1])+'\n' if lines[0].strip() in ['```python','```py','```'] and lines[-1].strip()=='```' else raw
assert m['engine_before']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
number=sys.argv[1];assert number in ['064','065','066','067'];raw=(root/f'task-{number}-candidate.txt').read_text();method=body(raw);new=ast.parse(method);assert len(new.body)==1 and isinstance(new.body[0],ast.FunctionDef) and new.body[0].name=='forward';assert [x.arg for x in new.body[0].args.args]==['self','x']
helper_raw=(root/'task-062-candidate.txt').read_text();helper=body(helper_raw);helper_tree=ast.parse(helper)
def callback(t):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='project')
old=body((P/'benchmarks/results/2026-11-16-mlp-prefill-activation/local-model-source/task-058-candidate.txt').read_text());assert ast.dump(callback(ast.parse(old)),include_attributes=False)==ast.dump(callback(helper_tree),include_attributes=False),'Shared preparation/math differs from validated prototype'
p=P/'spark_engine/qwen_model.py';source=p.read_text();tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenMLP');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward');lines=source.splitlines(keepends=True);updated=''.join(lines[:fn.lineno-1])+textwrap.indent(method,'    ')+''.join(lines[fn.end_lineno:]);newtree=ast.parse(updated)
def frozen(t):
 c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='QwenMLP');f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward');f.body=[ast.Pass()];return ast.dump(t,include_attributes=False)
assert frozen(tree)==frozen(newtree)
p.write_text(updated);(P/'spark_engine/qwen_mlp_fp8_pair.py').write_text(helper);(root/'placed-forward.py.txt').write_text(method);(root/'placed-helper.py.txt').write_text(helper);m.update(status='candidate_pending_integration_checks',applied_local_model_tasks={'helper':'062','forward':number},helper_raw_sha256=hashlib.sha256(helper_raw.encode()).hexdigest(),helper_body_sha256=hashlib.sha256(helper.encode()).hexdigest(),forward_raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),forward_body_sha256=hashlib.sha256(method.encode()).hexdigest(),shared_callback_ast_matches_validated_058=True);m['current_engine_sha256']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};(main/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('LOCAL_QWEN_FP8_FACTORY_AND_FORWARD_PLACED_VERBATIM')
