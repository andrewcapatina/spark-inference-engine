"""Retain both real-tested and final cache-safe prototypes before exact rollback."""
from pathlib import Path
import json,hashlib,subprocess,fcntl
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-keys-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());locks=[]
for name in ['controller.lock','benchmark.lock']:
 f=(R/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
current={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert current==m['current_engine_sha256'];out=P/'benchmark_runs'/run.name;retained=out/'full-final-runtime-stride-engine-before-rollback';retained.mkdir()
for p in (P/'spark_engine').glob('*.py'):(retained/p.name).write_bytes(p.read_bytes())
assert all(hashlib.sha256((retained/Path(k).name).read_bytes()).hexdigest()==v for k,v in current.items());(out/'unadopted-tracked.patch').write_bytes(subprocess.check_output(['git','diff','--binary','--','spark_engine'],cwd=P));assert set(current)-set(m['engine_before'])=={'spark_engine/qwen_key_prepare.py'}
for name,sha in m['engine_before'].items():
 data=(Path(m['baseline_directory'])/Path(name).name).read_bytes();assert hashlib.sha256(data).hexdigest()==sha;(P/name).write_bytes(data)
(P/'spark_engine/qwen_key_prepare.py').unlink();restored={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert restored==m['engine_before'];m.update(status='not_adopted_small_real_prefill_gain_baseline_restored',retained_final_prototype_directory=str(retained),retained_final_prototype_engine_sha256=current,current_engine_sha256=restored,restored_engine_sha256=restored);(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('All17final prototype sources retained; all16accepted3139a9b sources restored exactly.')
