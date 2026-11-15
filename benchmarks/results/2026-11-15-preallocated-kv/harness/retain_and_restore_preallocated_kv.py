"""Retain the complete local candidate, then restore exact accepted bytes."""
from pathlib import Path
import json,hashlib,subprocess,fcntl
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/preallocated-kv-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());locks=[]
for name in ['controller.lock','benchmark.lock']:
 f=(R/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
for name in ['spark-prototype-agent','spark-prototype-model','spark-prototype-vllm-nvfp4','spark-prototype-quant-agent']:assert subprocess.check_output(['docker','inspect','--format','{{.State.Running}}',name],text=True).strip()=='false'
assert subprocess.check_output(['systemctl','--user','show','spark-prototype-overnight-20261113.service','--value','-p','MainPID'],text=True).strip()=='0'
for kind in ['ab','profile']:
 stage=Path((R/f'feedback/preallocated-kv-{kind}-run.txt').read_text().strip());assert json.loads((stage/'trial-status.json').read_text())['status']=='complete'
audit=json.loads((run/'paired-results-audit.json').read_text());assert audit['status']=='cpu_crosschecks_passed'
current={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert current==m['current_engine_sha256'];before=m['engine_before'];new=set(current)-set(before);assert new=={'spark_engine/qwen_kv_cache.py','spark_engine/qwen_kv_cache_impl.py'};changed={name for name in before if current[name]!=before[name]};assert changed=={'spark_engine/qwen_layers.py'}
retained=P/'benchmark_runs'/run.name/'retained-preallocated-candidate';retained.mkdir(exist_ok=False)
for name,digest in current.items():
 dest=retained/Path(name).name;dest.write_bytes((P/name).read_bytes());assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
(retained/'source-sha256.json').write_text(json.dumps(current,indent=2)+'\n');(retained/'manifest-before-restore.json').write_text(json.dumps(m,indent=2)+'\n');restored=subprocess.check_output(['git','show',m['baseline_commit']+':spark_engine/qwen_layers.py'],cwd=P);assert hashlib.sha256(restored).hexdigest()==before['spark_engine/qwen_layers.py'];(P/'spark_engine/qwen_layers.py').write_bytes(restored)
for name in new:
 assert hashlib.sha256((retained/Path(name).name).read_bytes()).hexdigest()==current[name];(P/name).unlink()
actual={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert actual==before;m['candidate_engine_sha256']=current;m['current_engine_sha256']=actual;m['status']='retained_not_adopted_no_material_gain';m['retained_candidate']=str(retained);m['restored_baseline_commit']=m['baseline_commit'];m['candidate_full_quality_and_protected_checks']='not run: performance did not justify adoption; accepted baseline had all protected gates passed and was restored byte-exact';(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps({'status':m['status'],'retained_candidate':str(retained),'baseline_bytes_restored':True}))
