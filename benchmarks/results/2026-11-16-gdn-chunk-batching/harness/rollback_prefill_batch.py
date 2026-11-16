"""Retain the whole rejected candidate before exact accepted-source restoration."""
from pathlib import Path
import json,hashlib,subprocess,fcntl
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-batch-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());locks=[]
for name in ['controller.lock','benchmark.lock']:
 f=(R/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
current={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert current==m['current_engine_sha256']
out=P/'benchmark_runs'/run.name;retained=out/'full-engine-before-rollback';retained.mkdir()
for p in (P/'spark_engine').glob('*.py'):(retained/p.name).write_bytes(p.read_bytes())
assert all(hashlib.sha256((retained/Path(k).name).read_bytes()).hexdigest()==v for k,v in current.items());(out/'rejected-tracked.patch').write_bytes(subprocess.check_output(['git','diff','--binary','--','spark_engine'],cwd=P))
before=Path(m['baseline_directory'])
for name,digest in m['engine_before'].items():
 source=(before/Path(name).name).read_bytes();assert hashlib.sha256(source).hexdigest()==digest;(P/name).write_bytes(source)
restored={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert restored==m['engine_before'];m.update(status='not_adopted_component_performance_regression_baseline_restored',decision='Native-layout exact specialization passes304direct44blocks20tiny2gradient11CPU/state cases. Bounded group2/4/8/16 all slower at1024 on actual convolution layout; no real-checkpoint trial warranted.',retained_candidate_directory=str(retained),retained_candidate_engine_sha256=current,current_engine_sha256=restored,restored_engine_sha256=restored);(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('Retained all16candidate sources before restoring accepted3139a9b byte for byte.')
