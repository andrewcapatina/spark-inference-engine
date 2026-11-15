"""Serial supervised validation stages with durable checkpoints; no engine edits."""
from pathlib import Path
import sys,subprocess,json,hashlib,importlib.util
from datetime import datetime,timezone
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/qkv-main-run.txt').read_text().strip());manifest=json.loads((run/'manifest.json').read_text());expected=manifest['current_engine_sha256'];state={'status':'running','phase':'preflight','started':datetime.now(timezone.utc).isoformat(),'stages':[],'main_run':str(run)}
def save():
 state['updated']=datetime.now(timezone.utc).isoformat();(run/'remaining-validation-status.json').write_text(json.dumps(state,indent=2)+'\n');p=R/'status.json';d=json.loads(p.read_text());d['qkv_validation']=state;d.update(status='qkv_validation_running' if state['status']=='running' else 'qkv_validation_'+state['status'],phase=state['phase']);p.write_text(json.dumps(d,indent=2)+'\n')
def check_source():assert expected=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
def launch(kind):
 check_source()
 completed=next((x for x in state['stages'] if x['stage']==kind and x['status']=='complete'),None)
 if completed:
  stage=R/'runs'/completed['run'];trial=json.loads((stage/'trial-status.json').read_text());assert trial['status']=='complete' and trial['engine_sha256']==expected;data=json.loads((R/'feedback/nvfp4-reference'/stage.name/'results.json').read_text());assert data['status']=='complete';print('REUSE_PASSED '+kind+' '+stage.name,flush=True);return stage,data
 state['phase']=kind;save();subprocess.run([sys.executable,str(R/'prepare_qkv_trial.py'),kind],check=True);stage=Path((R/('feedback/qkv-'+kind+'-run.txt')).read_text().strip());state['active_run']=str(stage);save();print('BEGIN '+kind+' '+stage.name,flush=True)
 subprocess.run([sys.executable,str(R/'launch_qkv_prepared.py'),kind],check=True)
 data=json.loads((R/'feedback/nvfp4-reference'/stage.name/'results.json').read_text());assert data['status']=='complete';state['stages'].append({'stage':kind,'run':stage.name,'status':'complete'});save();return stage,data
error=None
if '--resume' in sys.argv:
 previous=json.loads((run/'remaining-validation-status.json').read_text());assert previous['status']=='failed';backup=run/('remaining-validation-before-resume-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json');backup.write_text(json.dumps(previous,indent=2)+'\n');state['stages']=previous['stages'];state['resumed_from']=str(backup)
save()
try:
 check_source();ab=Path((R/'feedback/qkv-ab-run.txt').read_text().strip());a=json.loads((R/'feedback/nvfp4-reference'/ab.name/'results.json').read_text());assert a['status']=='complete' and len(a['cases'])==2 and all(x['logits_and_all_cache_contents_element_exact'] for x in a['cases'])
 stage,data=launch('accuracy');spec=importlib.util.spec_from_file_location('comparison',R/'compare_accuracy_evaluation.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);suite=json.loads((P/'benchmarks/accuracy/nvfp4-smoke-v1.json').read_text());comparisons={}
 for label,reference_run in [('prior_native','accuracy-native-20261115T035947Z'),('vllm_reference','accuracy-reference-20261115T040352Z')]:
  reference=json.loads((R/'feedback/nvfp4-reference'/reference_run/'results.json').read_text());result=module.compare(data,reference,suite);result.update(engine_changes=True,native_run=stage.name,reference_run=reference_run);result['same_generated_ids_cases']=sum(x['output_token_ids']==y['output_token_ids'] for x,y in zip(data['cases'],reference['cases']));comparisons[label]=result;assert result['status']=='screen_passed',label+' quality regression'
 (stage/'quality-comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n');print('QUALITY_CHECKS_PASSED',flush=True)
 stage,profile=launch('profile')
 baseline_profile=json.loads((R/'feedback/nvfp4-reference/quant-profile-native-20261115T041946Z/results.json').read_text())
 for current,previous,frozen in zip(profile['cases'],a['cases'],baseline_profile['cases']):assert current['prompt_tokens']==previous['prompt_tokens'];assert current['prompt_token_ids']==frozen['prompt_token_ids'];assert current['output_token_ids']==previous['generation_repetitions'][0]['output_token_ids'][:16]
 state['phase']='protected_regressions';save();print('BEGIN protected56Qwen15Llama',flush=True);subprocess.run([sys.executable,str(R/'run_qkv_preserved_regression.py')],check=True);protected=json.loads((run/'protected-qkv-regression-status.json').read_text());assert protected['status']=='complete';state['stages'].append({'stage':'protected_regressions','status':'complete','commands':protected['completed_commands']});check_source();state.update(status='complete',phase='all_validations_complete');print('QKV_REMAINING_VALIDATION_COMPLETE',flush=True)
except BaseException as e:error=e;state.update(status='failed',error={'type':type(e).__name__,'message':str(e)});raise
finally:state['finished']=datetime.now(timezone.utc).isoformat();save()
