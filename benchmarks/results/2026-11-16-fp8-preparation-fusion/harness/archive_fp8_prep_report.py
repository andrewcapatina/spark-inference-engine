"""Export compact reproducibility evidence; omit weights, caches, traces and tensors."""
from pathlib import Path
import json,shutil,hashlib
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');main=Path((R/'feedback/fp8-prep-main-run.txt').read_text().strip());work=P/'benchmark_runs'/main.name;out=P/'benchmarks/results/2026-11-16-fp8-preparation-fusion';out.mkdir(exist_ok=True);h=out/'harness';h.mkdir(exist_ok=True);a=out/'local-author';a.mkdir(exist_ok=True)
for name in ['manifest.json','placement.json','integration-source-review.json','paired-prefill-audit.json','protected-correctness.json','cleanup-check.json','remaining-validation-status.json','legacy-portability-results.json','final-source-ownership.json']:
 shutil.copy2(main/name,out/name)
for number in ['065','066','067','068','069','070','071']:
 for suffix in ['candidate.txt','context.json','response.json']:
  shutil.copy2(work/f'task-{number}-{suffix}',a/f'task-{number}-{suffix}')
 shutil.copy2(R/f'prompts/nvfp4-task-{number}.txt',a/f'task-{number}-prompt.txt')
 review=main/f'task-{number}-review.json'
 if review.exists():shutil.copy2(review,a/review.name)
for name in ['prepare_fp8_prep_trial.py','launch_fp8_prep_trial.py','launch_fp8_prep_author.py','request_fp8_prep_author.py','place_fp8_prep_candidate.py','review_fp8_prep_integration.py','run_fp8_prep_probe.py','run_fp8_prep_prototype.py','run_fp8_prep_integration.py','run_fp8_prep_ab.py','run_fp8_prep_accuracy.py','run_fp8_prep_profile.py','run_fp8_prep_legacy.py','run_fp8_prep_preserved_regression.py','finish_fp8_prep_validation.py','compare_fp8_prep_results.py','summarize_fp8_prep_protected.py','check_fp8_prep_cleanup.py','archive_fp8_prep_report.py','write_fp8_prep_report.py','finish_fp8_prep_checkpoint.py','nvfp4_update_runtime.py','shutdown_process_probe.py','request_nvfp4_candidate.py','compare_accuracy_evaluation.py']:
 path=R/name
 if not path.exists():path=R/'acceptance'/name
 shutil.copy2(path,h/name)
for name in ['probe_fp8_prep_reference.py','check_fp8_prepare_prototype.py','check_fp8_prepare_integration.py','check_fp8_prepare_legacy.py','benchmark_fp8_prep_ab.py','profile_fp8_prep_candidate.py','fp8_forward_before_prepare.py.txt','fp8_pair_before_prepare.py.txt','evaluate_quant_accuracy.py','quant_profile_summary.py','prefill-canonical-generation-v1.json']:
 shutil.copy2(R/'acceptance'/name,h/name)
stages={}
for pointer in (R/'feedback').glob('fp8-prep-*-run.txt'):
 if pointer.name=='fp8-prep-main-run.txt':continue
 run=Path(pointer.read_text().strip());data=R/'feedback/nvfp4-reference'/run.name/'results.json'
 if data.exists():
  target=out/(run.name+'-results.json');shutil.copy2(data,target);stages[run.name]={'status':json.loads(data.read_text())['status'],'results_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
# Explicitly retain the two executed rejected prototypes, not just the latest pointer.
for name in ['fp8-prep-prototype-20261117T062623Z','fp8-prep-prototype-20261117T063136Z']:
 path=R/'feedback/nvfp4-reference'/name/'results.json';shutil.copy2(path,out/(name+'-results.json'));stages[name]={'status':'failed','results_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(out/'stage-index.json').write_text(json.dumps(stages,indent=2)+'\n');print(out)
