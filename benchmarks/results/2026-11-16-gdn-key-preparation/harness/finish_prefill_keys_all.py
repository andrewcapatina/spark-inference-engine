"""Serial final-source A/B, then unchanged quality/profile/protected gates."""
from pathlib import Path
import json,subprocess,sys,hashlib,statistics
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-keys-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());expected=m['current_engine_sha256']
def source():assert expected=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
source();integration=Path((R/'feedback/prefill-keys-integration-run.txt').read_text().strip());state=json.loads((integration/'trial-status.json').read_text());assert state['status']=='complete' and state['engine_sha256']==expected
retained=P/'benchmark_runs'/run.name/'final-engine-for-validation';retained.mkdir()
for p in (P/'spark_engine').glob('*.py'):(retained/p.name).write_bytes(p.read_bytes())
subprocess.run([sys.executable,str(R/'prepare_prefill_keys_trial.py'),'ab'],check=True);stage=Path((R/'feedback/prefill-keys-ab-run.txt').read_text().strip());m.update(status='final_source_ab_running',final_source_ab_run=stage.name);(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
subprocess.run([sys.executable,str(R/'launch_prefill_keys_prepared.py'),'ab'],check=True);source();data=json.loads((R/'feedback/nvfp4-reference'/stage.name/'results.json').read_text());assert data['status']=='complete' and len(data['cases'])==2
for c in data['cases']:
 assert c['generation_ids_element_exact'] and c['generation_repeats_per_backend']==2
 assert c['prefill_logits_and_all_cache_contents_element_exact'] and c['next_decode_logits_and_all_cache_contents_element_exact']
 assert len(c['prefill_repetitions'])==32 and c['less_prefill_time_pct']>0
 for offset in range(0,32,4):
  samples=c['prefill_repetitions'][offset:offset+4];medians={b:statistics.median(x['cuda_ms'] for x in samples if x['backend']==b) for b in ['original','candidate']};assert medians['candidate']<medians['original'],'Inconsistent performance block; review before further validation'
m['status']='final_source_ab_exact_and_consistent_remaining_gates';(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('FINAL_SOURCE_AB_EXACT_ALL_EIGHT_BLOCKS_POSITIVE',flush=True)
subprocess.run([sys.executable,str(R/'finish_prefill_keys_validation.py')],check=True)
