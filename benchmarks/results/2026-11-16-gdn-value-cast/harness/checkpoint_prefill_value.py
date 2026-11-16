"""Checkpoint only validated verbatim local-Qwen implementation."""
from pathlib import Path
import hashlib,json,subprocess
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-value-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text())
for file,status in [('remaining-validation-status.json','complete'),('protected-correctness.json','all_protected_gates_passed'),('cleanup-check.json','clean'),('paired-prefill-audit.json','value_cast_prefill_audit_passed')]:assert json.loads((run/file).read_text())['status']==status
assert m['current_engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert (P/'spark_engine/qwen_ops.py').read_bytes()==(P/'benchmark_runs'/run.name/'task-054-placed.py.txt').read_bytes()
subprocess.run(['git','add','--','spark_engine/qwen_ops.py'],cwd=P,check=True);assert subprocess.check_output(['git','diff','--cached','--name-only'],cwd=P,text=True).splitlines()==['spark_engine/qwen_ops.py']
subprocess.run(['git','commit','--author=Local Qwen <local-qwen@spark.local>','-m','Defer BF16 GDN value widening while preserving FP32 state math'],cwd=P,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip();m.update(status='validated_checkpointed',engine_commit=commit);m['tasks'][0]['validation']='All strict component/tiny/real27B, frozen quality and protected original-model checks pass';(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(commit)
