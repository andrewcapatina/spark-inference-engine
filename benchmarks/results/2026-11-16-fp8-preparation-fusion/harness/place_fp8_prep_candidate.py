"""Place reviewed local-Qwen responses verbatim; no implementation authoring."""
from pathlib import Path
import ast,hashlib,json,textwrap
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');main=Path((R/'feedback/fp8-prep-main-run.txt').read_text().strip());work=P/'benchmark_runs'/main.name
m=json.loads((main/'manifest.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert m['engine_before']=={str(p.relative_to(P)):sha(p.read_bytes()) for p in (P/'spark_engine').glob('*.py')}
def body(number):
 s=(work/f'task-{number}-candidate.txt').read_text().strip()
 if s.startswith('```'):
  ls=s.splitlines();assert ls[-1]=='```';s='\n'.join(ls[1:-1])
 ast.parse(textwrap.dedent(s));return s+'\n'
kernel=body('068');assert kernel==(work/'prep_proto.py').read_text();(P/'spark_engine/qwen_fp8_prepare.py').write_text(kernel)
provenance={'spark_engine/qwen_fp8_prepare.py':{'task':'068','sha256':sha(kernel.encode())}}
for number,filename,cls,fn in [('069','qwen_fp8.py','FP8Linear','forward'),('071','qwen_mlp_fp8_pair.py',None,'_make_pair_projector')]:
 path=P/'spark_engine'/filename;old=path.read_text();tree=ast.parse(old);container=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls) if cls else tree
 node=next(n for n in container.body if isinstance(n,ast.FunctionDef) and n.name==fn);s=body(number);s=textwrap.dedent(s);t=ast.parse(s);assert len(t.body)==1 and t.body[0].name==fn
 if cls:s=textwrap.indent(s,' '*node.col_offset)
 lines=old.splitlines(keepends=True);new=''.join(lines[:node.lineno-1])+s+''.join(lines[node.end_lineno:]);ast.parse(new);path.write_text(new)
 provenance[str(path.relative_to(P))]={'task':number,'response_sha256':sha((work/f'task-{number}-candidate.txt').read_bytes()),'placed_body_sha256':sha(s.encode())}
m.update(status='candidate_placed_awaiting_integration',current_engine_sha256={str(p.relative_to(P)):sha(p.read_bytes()) for p in (P/'spark_engine').glob('*.py')},local_source_provenance=provenance)
(main/'placement.json').write_text(json.dumps(provenance,indent=2)+'\n');(main/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
