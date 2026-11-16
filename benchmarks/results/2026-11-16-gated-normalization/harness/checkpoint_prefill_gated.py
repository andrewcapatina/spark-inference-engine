"""Checkpoint validated local-model implementation, never author or repair engine code."""
from pathlib import Path
import json,hashlib,subprocess
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-gated-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text())
for file,status in [('remaining-validation-status.json','complete'),('protected-correctness.json','all_protected_gates_passed'),('cleanup-check.json','clean'),('paired-prefill-audit.json','gated_normalization_prefill_audit_passed')]:assert json.loads((run/file).read_text())['status']==status
assert m['current_engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};raw=P/'benchmark_runs'/run.name
for target,task in [('qwen_ops.py','038'),('qwen_layers.py','040'),('qwen_gated_norm_tail.py','037')]:assert (P/'spark_engine'/target).read_bytes()==(raw/f'task-{task}-placed.py.txt').read_bytes()
subprocess.run(['git','add','--','spark_engine/qwen_ops.py','spark_engine/qwen_layers.py','spark_engine/qwen_gated_norm_tail.py'],cwd=P,check=True)
assert set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=P,text=True).splitlines())=={'spark_engine/qwen_ops.py','spark_engine/qwen_layers.py','spark_engine/qwen_gated_norm_tail.py'}
subprocess.run(['git','commit','--author=Local Qwen <local-qwen@spark.local>','-m','Fuse gated RMSNorm prefill tail with explicit rounding and dispatch'],cwd=P,check=True)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip();m.update(status='validated_checkpointed',engine_commit=commit)
for t in m['tasks']:t['validation']='passed: primitive/tiny/real27B exact checks, frozen quality and protected original-model regressions'
(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(commit)
