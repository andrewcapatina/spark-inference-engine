from pathlib import Path
import ast,json,textwrap,hashlib
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');main=Path((R/'feedback/fp8-prep-main-run.txt').read_text().strip());work=P/'benchmark_runs'/main.name
assert json.loads((main/'local-author-071-status.json').read_text())['status']=='complete'
a=(work/'task-070-candidate.txt').read_text().strip();b=(work/'task-071-candidate.txt').read_text().strip()
assert b==a.replace('if e.name not in ("triton", "triton.submodule"):', 'if e.name != "triton" and not (e.name or "").startswith("triton."):')
# A source-locality check independently checks retained fallback and GEMM bodies.
for num,reference,fn in [('069','fp8_forward_before_prepare.py.txt','forward'),('071','fp8_pair_before_prepare.py.txt','_make_pair_projector')]:
 new=ast.parse(textwrap.dedent((work/f'task-{num}-candidate.txt').read_text()));old=ast.parse(textwrap.dedent((R/'acceptance'/reference).read_text()));nf=new.body[0];of=old.body[0]
 if fn=='_make_pair_projector':
  op=next(x for x in of.body if isinstance(x,ast.FunctionDef));np=next(x for x in nf.body if isinstance(x,ast.FunctionDef));oi=of.body.index(op);ni=nf.body.index(np);assert ast.dump(ast.Module(body=of.body[:oi],type_ignores=[]))==ast.dump(ast.Module(body=nf.body[:ni],type_ignores=[]));of,nf=op,np
 first=next(i for i,x in enumerate(of.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scale_a' for t in x.targets));last=next(i for i,x in enumerate(of.body) if isinstance(x,ast.If) and isinstance(x.test,ast.Name) and x.test.id=='pad')
 orig_prep=of.body[first:last+1];fallback=next(x for x in nf.body if isinstance(x,ast.If) and isinstance(x.test,ast.Compare) and isinstance(x.test.left,ast.Name) and x.test.left.id=='prepared')
 dump=lambda v:ast.dump(ast.Module(body=v,type_ignores=[]));assert dump(orig_prep)==dump(fallback.body);idx=nf.body.index(fallback);assert dump(of.body[:first])==dump(nf.body[:first]);assert dump(of.body[last+1:])==dump(nf.body[idx+1:])
(main/'integration-source-review.json').write_text(json.dumps({'status':'reviewed','local069_forward':'Original asserts, reshape, fallback prep, GEMM, cropping and result preserved; narrow import guard and runtime errors propagate','local071_pair':'Original factory guards, asserts, reshape, fallback prep, two GEMMs, cropping and result preserved; only070guard repaired','author':'Local Qwen only'},indent=2)+'\n');print('INTEGRATION_SOURCE_REVIEW_PASS')
