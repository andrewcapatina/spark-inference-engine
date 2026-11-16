"""Audit measured prefill and exact-output evidence; no engine execution."""
from pathlib import Path
import hashlib,json,statistics
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference';run=Path((R/'feedback/prefill-keys-main-run.txt').read_text().strip())
def result(kind):return json.loads((B/Path((R/f'feedback/prefill-keys-{kind}-run.txt').read_text().strip()).name/'results.json').read_text())
ab=result('ab');after=result('profile');quality=result('accuracy');before=json.loads((B/'prefill-value-profile-20261116T064526Z/results.json').read_text());old_quality=json.loads((B/'prefill-value-accuracy-20261116T064210Z/results.json').read_text())
assert all(x['status']=='complete' for x in [ab,after,before,quality]);assert len(ab['cases'])==len(after['cases'])==len(before['cases'])==2;rows=[]
def scope(c,name):return next(x for x in c['profile']['trace_summary']['raw_kernel_ownership']['inclusive_scopes'] if x['phase']=='prefill' and x['name']==name)
for a,n,o in zip(ab['cases'],after['cases'],before['cases']):
 assert a['prompt_tokens']==n['prompt_tokens']==o['prompt_tokens'];assert n['prompt_token_ids']==o['prompt_token_ids'];assert all(a[k] for k in ['prefill_logits_and_all_cache_contents_element_exact','next_decode_logits_and_all_cache_contents_element_exact','generation_ids_element_exact']);assert a['generation_repeats_per_backend']==2 and len(a['generation_validation'])==4
 assert all(x['output_token_ids']==a['generation_validation'][0]['output_token_ids'] for x in a['generation_validation']);assert len(a['generation_validation'][0]['output_token_ids'])==128
 assert n['output_token_ids']==o['output_token_ids']==a['generation_validation'][0]['output_token_ids'][:1];assert n['cuda_event_diagnostic']['observed_decode_calls']==0
 assert len(a['prefill_repetitions'])==32
 for backend in ['original','candidate']:
  samples=[x['cuda_ms'] for x in a['prefill_repetitions'] if x['backend']==backend];assert len(samples)==16 and statistics.median(samples)==a['summary'][backend]['median_prefill_cuda_ms']
 blocks=[]
 for offset in range(0,32,4):
  samples=a['prefill_repetitions'][offset:offset+4];m={b:statistics.median(x['cuda_ms'] for x in samples if x['backend']==b) for b in ['original','candidate']};blocks.append({'original_ms':m['original'],'candidate_ms':m['candidate'],'less_time_pct':100*(1-m['candidate']/m['original'])})
 assert all(b['less_time_pct']>0 for b in blocks)
 old,new=scope(o,'native.gdn.state_update'),scope(n,'native.gdn.state_update');assert old['kernels']-new['kernels']==2*48*((a['prompt_tokens']+63)//64)
 assert scope(n,'native.gdn.key_prepare')['kernels']==48*((a['prompt_tokens']+63)//64)
 for name in ['native.norm.plain','native.norm.gated','primitive.fp8.gemm','primitive.nvfp4.gemm']:assert scope(o,name)['kernels']==scope(n,name)['kernels']
 peak={b:statistics.median(x['peak_allocated_bytes'] for x in a['prefill_repetitions'] if x['backend']==b) for b in ['original','candidate']}
 rows.append({'prompt_tokens':a['prompt_tokens'],'paired_prefill':a['summary'],'less_prefill_time_pct':a['less_prefill_time_pct'],'abba_blocks':blocks,'median_peak_allocated_bytes':peak,'gdn_profile_before':old,'gdn_profile_after':new,'gdn_kernels_removed':old['kernels']-new['kernels'],'profile_decode_calls':0})
assert after['norm_input_shapes']==before['norm_input_shapes'];assert quality['suite_sha256']==old_quality['suite_sha256'] and quality['scoring_sha256']==old_quality['scoring_sha256'];assert len(quality['cases'])==40 and all(x['output_token_ids']==y['output_token_ids'] for x,y in zip(quality['cases'],old_quality['cases']));assert all(x['token_nll']==y['token_nll'] for x,y in zip(quality['corpus'],old_quality['corpus']))
m=json.loads((run/'manifest.json').read_text());current={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};assert current==m['current_engine_sha256'];assert set(current)-set(m['engine_before'])=={'spark_engine/qwen_key_prepare.py'};assert {n for n in m['engine_before'] if current[n]!=m['engine_before'][n]}=={'spark_engine/qwen_ops.py'}
d={'status':'key_preparation_prefill_audit_passed','scope':'Warm direct model.prefill; repeated small-workload test, no confidence interval or output-token speed claim','rows':rows,'quality_generated_ids_exact_cases':40,'quality_nll_exact_count':sum(len(x['token_nll']) for x in quality['corpus']),'quality_perplexity':quality['summary']['perplexity'],'current_engine_sha256':current};(run/'paired-prefill-audit.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
