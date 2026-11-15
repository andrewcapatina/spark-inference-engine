"""Prepare local authorship boundaries after the previous checkpoint is complete."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine')
previous=Path((R/'feedback/attention-kv-main-run.txt').read_text().strip());assert json.loads((previous/'manifest.json').read_text())['status']=='validated_checkpointed';assert not (P/'spark_engine/qwen_kv_cache.py').exists()
run=R/'runs'/('preallocated-attention-kv-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));run.mkdir();digests={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};m={'user_authorized_nvfp4_update':True,'user_authorized_attention_preallocation':True,'implementation_author':'local Qwen only','baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),'engine_before':digests,'current_engine_sha256':digests,'tasks':[],'status':'local_author_pending','scope':'Standalone tuple-compatible inference cache helper, then separately authored one-method integration; branch snapshots and math preserved; convolution compaction stays quarantined'};(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
for name in ['preallocated-kv-main-run.txt','gdn-storage-run.txt']:(R/'feedback'/name).write_text(str(run)+'\n')
s=json.loads((R/'status.json').read_text());s.update(active_run=str(run),preallocated_kv_status='LocalQwen023 standalone helper authoring; default engine behavior unchanged');(R/'status.json').write_text(json.dumps(s,indent=2)+'\n');print(run)
