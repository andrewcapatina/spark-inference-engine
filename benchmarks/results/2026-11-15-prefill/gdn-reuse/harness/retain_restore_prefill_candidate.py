"""Retain an unadopted prefill candidate, restore exact accepted source bytes."""
from pathlib import Path
import json,hashlib,subprocess,sys,fcntl
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');kind=sys.argv[1];assert kind in ['gdn','fp8'];run=Path((R/f'feedback/prefill-{kind}-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());locks=[]
for name in ['controller.lock','benchmark.lock']:
 f=(R/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
for name in ['spark-prototype-agent','spark-prototype-model','spark-prototype-vllm-nvfp4','spark-prototype-quant-agent']:assert subprocess.check_output(['docker','inspect','--format','{{.State.Running}}',name],text=True).strip()=='false'
assert subprocess.check_output(['systemctl','--user','show','spark-prototype-overnight-20261113.service','--value','-p','MainPID'],text=True).strip()=='0'
stage=Path((R/f'feedback/prefill-{kind}-ab-run.txt').read_text().strip());assert json.loads((stage/'trial-status.json').read_text())['status']=='complete';d=json.loads((R/'feedback/nvfp4-reference'/stage.name/'results.json').read_text());assert d['status']=='complete' and all(c['prefill_logits_and_all_cache_contents_element_exact'] and c['next_decode_logits_and_all_cache_contents_element_exact'] and c['generation_ids_element_exact'] for c in d['cases'])
current={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert current==m['current_engine_sha256'] and current.keys()==m['engine_before'].keys();changed={name for name in current if current[name]!=m['engine_before'][name]};assert changed==({'spark_engine/qwen_ops.py'} if kind=='gdn' else {'spark_engine/qwen_fp8.py'})
retained=P/'benchmark_runs'/run.name/'retained-unadopted-candidate';retained.mkdir(exist_ok=False)
for name,digest in current.items():
 dest=retained/Path(name).name;dest.write_bytes((P/name).read_bytes());assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
(retained/'source-sha256.json').write_text(json.dumps(current,indent=2)+'\n')
for name in changed:
 original=subprocess.check_output(['git','show',m['baseline_commit']+':'+name],cwd=P);assert hashlib.sha256(original).hexdigest()==m['engine_before'][name];(P/name).write_bytes(original)
restored={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert restored==m['engine_before'];m.update(candidate_engine_sha256=current,current_engine_sha256=restored,status='not_adopted_no_measurable_prefill_gain',retained_candidate=str(retained));(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'status':m['status'],'restored':True,'retained':str(retained)}))
