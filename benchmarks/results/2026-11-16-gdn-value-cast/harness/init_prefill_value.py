"""Prepare a local-model-only value-cast experiment; no engine edits."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, subprocess
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=P,text=True).strip()
old=Path((R/'feedback/prefill-solve-main-run.txt').read_text().strip());prior=json.loads((old/'manifest.json').read_text())
sha={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
assert sha==prior['restored_engine_sha256'] and len(sha)==16
run=R/'runs'/('prefill-gdn-value-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));run.mkdir()
out=P/'benchmark_runs'/run.name;out.mkdir();baseline=out/'engine-before';baseline.mkdir()
for path in (P/'spark_engine').glob('*.py'):(baseline/path.name).write_bytes(path.read_bytes())
source=(P/'spark_engine/qwen_ops.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='gated_delta_chunk');fn='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
(R/'acceptance/qwen_ops_before_value_processing.py.txt').write_text(source)
m={'user_authorized_nvfp4_update':True,'user_authorized_prefill_optimization':True,'implementation_author':'local Qwen only','baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip(),'engine_before':sha,'current_engine_sha256':sha,'tasks':[],'status':'local_author_requested','scope':'Native Torch BF16 CUDA inference value cast deferral, FP32 state and output preserved; no new helper kernels','primary_metrics':['prefill_cuda_ms','prompt_tokens_per_second'],'baseline_directory':str(baseline),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
(run/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(R/'feedback/prefill-value-main-run.txt').write_text(str(run)+'\n')
prompt='''Return ONLY the COMPLETE gated_delta_chunk function below with the same signature. You alone write implementation. No Markdown, no explanation. Change ONLY this function, <=50 changed lines. No new helper, import, kernel, compiler or solve/layout change.

Focused experiment: avoid eager full-value FP32 widening ONLY for original query dtype BF16, value dtype BF16, value CUDA, torch.is_inference_mode_enabled(). Keep v as its transposed BF16 view for that eligible case. Otherwise keep original transposed v.float() behavior (CPU/other dtype/no_grad/autograd all original). q/k/g/beta conversions and all math stay original.

The existing v_c * beta_c.unsqueeze(-1) has FP32 beta, so Torch promotion must produce FP32 v_beta without an explicit v_c.float(). Do not add per-chunk casts or change product order. Keep both original triangular solves, exponentials, GEMMs, masks, padding, chunk loop, finalstate expression and return cast untouched.

CRITICAL: out must remain FP32 regardless of v dtype (zeros_like with explicit torch.float32). Initial state allocated when None must remain FP32, same device/shape. Provided initial_state must move to v.device and torch.float32 (not .to(v) when v BF16). Saved query return dtype stays unchanged. Do not mutate inputs/state. Include only the minimal eligibility condition, value cast deferral and explicit FP32 output/state allocations/conversion. Everything else must remain source-identical.

Baseline complete function:
'''+fn
(R/'prompts/nvfp4-task-054.txt').write_text(prompt)
(R/'request_prefill_value_author.py').write_text((R/'request_prefill_solve_author.py').read_text().replace('prefill-solve-main','prefill-value-main'))
(R/'place_prefill_value_integration.py').write_text((R/'place_prefill_solve_integration.py').read_text().replace('prefill-solve-main','prefill-value-main'))
q=json.loads((R/'prompts/prefill-queue.json').read_text());q.append({'id':'prefill_gdn_value_cast','task':'054','status':'local_author_requested','run':str(run),'scope':m['scope']});(R/'prompts/prefill-queue.json').write_text(json.dumps(q,indent=2)+'\n')
s=json.loads((R/'status.json').read_text());s.update(current_checkpoint_commit=m['baseline_commit'],status='prefill_value_local_authoring',phase='native_torch_value_cast_experiment',experiment_idle=False,next_optimization_topic={'topic':m['scope'],'status':'local_author_requested','run':str(run)},paired_solve_status='Not adopted; exact accepted16-file baseline restored; report9624e5a; long-prompt gain0.21%, copies erased kernel savings.');(R/'status.json').write_text(json.dumps(s,indent=2)+'\n')
latest='Paired solve completed, not adopted; exact accepted16-file engine restored; report9624e5a. New focused native Torch value-cast deferral task054 awaiting local Qwen. Preserve FP32 output/state, original two solves and all math; component exactness before real27B testing. No speedup claimed. One model at a time, service inactive, STOP/GPU/LARGE holds retained.\n'
(R/'feedback/latest.txt').write_text(latest);(P/'run_feedback/latest.txt').write_text(latest)
print(run)
