"""Aggregate the unchanged protected gate node IDs after isolated processes finish."""
from pathlib import Path
import json,re,hashlib
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-readd-main-run.txt').read_text().strip());status=json.loads((run/'protected-prefill-readd-regression-status.json').read_text());assert status['status']=='complete' and all(x['exit_code']==0 for x in status['completed_commands']);baseline=json.loads((R/'runs/nvfp4-native-20261115T005339Z/protected-correctness.json').read_text());qwen=set();llama=set()
for command in status['completed_commands']:
 text=(run/(command['label']+'.log')).read_text()
 qwen.update(re.findall(r'^PASSED (acceptance/test_qwen_contract.py::\S+)',text,re.M));llama.update(re.findall(r'^PASSED (acceptance/test_contract.py::\S+)',text,re.M))
assert qwen==set(baseline['qwen_nodeids']) and llama==set(baseline['llama_nodeids'])
fixture_sha={name:hashlib.sha256((R/'acceptance'/name).read_bytes()).hexdigest() for name in baseline['acceptance_source_sha256']};assert fixture_sha==baseline['acceptance_source_sha256'];manifest=json.loads((run/'manifest.json').read_text());engine_sha={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert engine_sha==manifest['current_engine_sha256']
d={'status':'all_protected_gates_passed','qwen_unique_gates':len(qwen),'llama_cases':len(llama),'qwen_nodeids':sorted(qwen),'llama_nodeids':sorted(llama),'numerical_tolerances_unchanged':True,'acceptance_source_sha256':fixture_sha,'source_engine_sha256':engine_sha,'commands':status['completed_commands'],'one_checkpoint_per_process':True,'host_floor_gib':32,'new_swap_guard_mib':256,'gpu_allocator_cap_gib':56};(run/'protected-correctness.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'status':d['status'],'qwen':len(qwen),'llama':len(llama)}))
