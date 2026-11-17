"""Audit source scope, exact-output evidence and measured prefill performance."""
from pathlib import Path
import hashlib,json,statistics
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference';run=Path((R/'feedback/fp8-prep-main-run.txt').read_text().strip())
def result(kind):return json.loads((B/Path((R/f'feedback/fp8-prep-{kind}-run.txt').read_text().strip()).name/'results.json').read_text())
ab=result('ab');after=result('profile');quality=result('accuracy');before=json.loads((B/'fp8-pair-profile-20261117T035833Z/results.json').read_text());old_quality=json.loads((B/'fp8-pair-accuracy-20261117T035519Z/results.json').read_text())
assert all(x['status']=='complete' for x in [ab,after,before,quality]);rows=[]
def scope(c,name):return next(x for x in c['profile']['trace_summary']['raw_kernel_ownership']['inclusive_scopes'] if x['phase']=='prefill' and x['name']==name)
for a,n,o in zip(ab['cases'],after['cases'],before['cases']):
 assert a['prompt_tokens']==n['prompt_tokens']==o['prompt_tokens'];assert n['prompt_token_ids']==o['prompt_token_ids'];assert all(a[k] for k in ['prefill_logits_and_all_cache_contents_element_exact','next_decode_logits_and_all_cache_contents_element_exact','generation_ids_element_exact']);assert a['generation_repeats_per_backend']==2 and len(a['generation_validation'])==4
 assert all(x['output_token_ids']==a['generation_validation'][0]['output_token_ids'] for x in a['generation_validation']);assert len(a['generation_validation'][0]['output_token_ids'])==128
 assert n['output_token_ids']==o['output_token_ids']==a['generation_validation'][0]['output_token_ids'][:1];assert n['cuda_event_diagnostic']['observed_decode_calls']==0
 assert len(a['prefill_repetitions'])==32
 blocks=[]
 for offset in range(0,32,4):
  samples=a['prefill_repetitions'][offset:offset+4];m={b:statistics.median(x['cuda_ms'] for x in samples if x['backend']==b) for b in ['original','candidate']};blocks.append({'original_ms':m['original'],'candidate_ms':m['candidate'],'less_time_pct':100*(1-m['candidate']/m['original'])})
 old,new=scope(o,'native.engine.prefill'),scope(n,'native.engine.prefill');assert old['kernels']-new['kernels']==1302
 assert scope(n,'primitive.fp8.prepare')['kernels']==217;assert scope(n,'native.mlp.fp8_pair')['kernels']==24
 assert scope(o,'native.layer.mlp')['kernels']-scope(n,'native.layer.mlp')['kernels']==48
 for name in ['native.mlp.silu_mul','native.gdn.state_update','native.gdn.key_prepare','native.norm.plain','native.norm.gated','primitive.fp8.gemm','primitive.nvfp4.gemm']:assert scope(o,name)['kernels']==scope(n,name)['kernels']
 peak={b:statistics.median(x['peak_allocated_bytes'] for x in a['prefill_repetitions'] if x['backend']==b) for b in ['original','candidate']}
 rows.append({'prompt_tokens':a['prompt_tokens'],'paired_prefill':a['summary'],'less_prefill_time_pct':a['less_prefill_time_pct'],'saved_ms':a['summary']['original']['median_prefill_cuda_ms']-a['summary']['candidate']['median_prefill_cuda_ms'],'abba_blocks':blocks,'median_peak_allocated_bytes':peak,'engine_profile_before':old,'engine_profile_after':new,'kernels_removed':old['kernels']-new['kernels'],'fp8_prepare_profile_after':scope(n,'primitive.fp8.prepare'),'profile_decode_calls':0})
assert after['norm_input_shapes']==before['norm_input_shapes'];assert quality['suite_sha256']==old_quality['suite_sha256'] and quality['scoring_sha256']==old_quality['scoring_sha256'];assert len(quality['cases'])==40 and all(x['output_token_ids']==y['output_token_ids'] for x,y in zip(quality['cases'],old_quality['cases']));assert all(x['token_nll']==y['token_nll'] for x,y in zip(quality['corpus'],old_quality['corpus']))
m=json.loads((run/'manifest.json').read_text());current={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert current==m['current_engine_sha256'];assert set(current)-set(m['engine_before'])=={'spark_engine/qwen_fp8_prepare.py'};assert {n for n in m['engine_before'] if current[n]!=m['engine_before'][n]}=={'spark_engine/qwen_fp8.py','spark_engine/qwen_mlp_fp8_pair.py'}
legacy=json.loads((run/'legacy-portability-results.json').read_text());assert legacy['status']=='complete' and len(legacy['cases'])==359
integration=result('integration');assert integration['status']=='complete' and len(integration['layer_cases'])==48 and len(integration['linear_cases'])==6
out={'status':'fp8_prep_prefill_audit_passed','scope':'Warm direct model.prefill; repeated small-workload test, no confidence interval or output-token speed claim','rows':rows,'quality_generated_ids_exact_cases':40,'quality_nll_exact_count':sum(len(x['token_nll']) for x in quality['corpus']),'quality_perplexity':quality['summary']['perplexity'],'legacy_exact_cases':359,'current_engine_sha256':current};(run/'paired-prefill-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['rows','current_engine_sha256']},indent=2))
