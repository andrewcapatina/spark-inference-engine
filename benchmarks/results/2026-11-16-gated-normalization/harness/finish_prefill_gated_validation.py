"""Serial correctness, prefill-only profiling and protected gates after paired prefill A/B."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,subprocess,importlib.util
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference';run=Path((R/'feedback/prefill-gated-main-run.txt').read_text().strip());expected=json.loads((run/'manifest.json').read_text())['current_engine_sha256'];state={'status':'running','phase':'preflight','started':datetime.now(timezone.utc).isoformat(),'main_run':str(run),'primary_metrics':['prefill_cuda_ms','prefill_wall_ms','prompt_tokens_per_second'],'stages':[]}
def check_source():assert expected=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
def save():
 state['updated']=datetime.now(timezone.utc).isoformat();(run/'remaining-validation-status.json').write_text(json.dumps(state,indent=2)+'\n');p=R/'status.json';s=json.loads(p.read_text());s['prefill_gated_validation']=state;s.update(status='prefill_gated_validation_'+state['status'],phase=state['phase']);p.write_text(json.dumps(s,indent=2)+'\n')
def launch(kind):
 check_source();prior=next((x for x in state['stages'] if x['stage']==kind and x['status']=='complete'),None)
 if prior:
  stage=R/'runs'/prior['run'];s=json.loads((stage/'trial-status.json').read_text());assert s['status']=='complete' and s['engine_sha256']==expected;data=json.loads((B/stage.name/'results.json').read_text());assert data['status']=='complete';return stage,data
 state['phase']=kind;save();subprocess.run([sys.executable,str(R/'prepare_prefill_gated_trial.py'),kind],check=True);stage=Path((R/f'feedback/prefill-gated-{kind}-run.txt').read_text().strip());state['active_run']=str(stage);save();print('BEGIN '+kind+' '+stage.name,flush=True);subprocess.run([sys.executable,str(R/'launch_prefill_gated_prepared.py'),kind],check=True);data=json.loads((B/stage.name/'results.json').read_text());assert data['status']=='complete';state['stages'].append({'stage':kind,'run':stage.name,'status':'complete'});save();return stage,data
if '--resume' in sys.argv:
 prior=json.loads((run/'remaining-validation-status.json').read_text());assert prior['status']=='failed';backup=run/('validation-before-resume-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');backup.write_text(json.dumps(prior,indent=2)+'\n');state['stages']=prior['stages'];state['resumed_from']=str(backup)
save()
try:
 check_source();abrun=Path((R/'feedback/prefill-gated-ab-run.txt').read_text().strip());ab=json.loads((B/abrun.name/'results.json').read_text());assert ab['status']=='complete' and len(ab['cases'])==2;assert all(c['prefill_logits_and_all_cache_contents_element_exact'] and c['next_decode_logits_and_all_cache_contents_element_exact'] and c['generation_ids_element_exact'] for c in ab['cases'])
 stage,data=launch('accuracy');suite=json.loads((P/'benchmarks/accuracy/nvfp4-smoke-v1.json').read_text());spec=importlib.util.spec_from_file_location('quality_compare',R/'compare_accuracy_evaluation.py');compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare);comparisons={}
 for label,baseline in [('accepted_native','prefill-norm-accuracy-20261115T174938Z'),('vllm_reference','accuracy-reference-20261115T040352Z')]:
  prior=json.loads((B/baseline/'results.json').read_text());result=compare.compare(data,prior,suite);assert result['status']=='screen_passed';result['reference_run']=baseline;result['native_run']=stage.name;result['same_generated_ids_cases']=sum(x['output_token_ids']==y['output_token_ids'] for x,y in zip(data['cases'],prior['cases']));comparisons[label]=result
 prior=json.loads((B/'prefill-norm-accuracy-20261115T174938Z/results.json').read_text());assert len(data['cases'])==len(prior['cases'])==40;assert all(x['output_token_ids']==y['output_token_ids'] for x,y in zip(data['cases'],prior['cases']));assert len(data['corpus'])==len(prior['corpus'])==4;assert all(x['token_nll']==y['token_nll'] for x,y in zip(data['corpus'],prior['corpus']));(stage/'quality-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n');print('FROZEN40_QUALITY_AND446LOSSES_EXACT',flush=True)
 stage,data=launch('profile');assert data['new_tokens']==1;baseline=json.loads((B/'prefill-norm-profile-20261115T175256Z/results.json').read_text())
 for c,a,p in zip(data['cases'],ab['cases'],baseline['cases']):
  assert c['prompt_tokens']==a['prompt_tokens'];assert c['prompt_token_ids']==p['prompt_token_ids'];assert c['output_token_ids']==a['generation_validation'][0]['output_token_ids'][:1]==p['output_token_ids'][:1];assert c['cuda_event_diagnostic']['observed_decode_calls']==0;assert not any(x['phase']=='decode' for x in c['profile']['events']['inclusive_scopes'])
 state['phase']='protected_regressions';save();subprocess.run([sys.executable,str(R/'run_prefill_gated_preserved_regression.py')],check=True);protected=json.loads((run/'protected-prefill-gated-regression-status.json').read_text());assert protected['status']=='complete';state['stages'].append({'stage':'protected_regressions','status':'complete','commands':protected['completed_commands']});check_source();state.update(status='complete',phase='all_validations_complete');print('PREFILL_GATED_VALIDATION_COMPLETE',flush=True)
except BaseException as e:state.update(status='failed',error={'type':type(e).__name__,'message':str(e)});raise
finally:state['finished']=datetime.now(timezone.utc).isoformat();save()
