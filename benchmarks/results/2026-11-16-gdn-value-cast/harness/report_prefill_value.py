"""Archive measured results, reviewed local source and reproducible harness."""
from pathlib import Path
import json,shutil,hashlib
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference';run=Path((R/'feedback/prefill-value-main-run.txt').read_text().strip())
def read(p):return json.loads(p.read_text())
m=read(run/'manifest.json');a=read(run/'paired-prefill-audit.json');assert m['status']=='validated_checkpointed';assert a['status']=='value_cast_prefill_audit_passed';assert read(run/'cleanup-check.json')['status']=='clean';assert read(run/'protected-correctness.json')['status']=='all_protected_gates_passed'
out=P/'benchmarks/results/2026-11-16-gdn-value-cast';out.mkdir(parents=True,exist_ok=True);h=out/'harness';h.mkdir(exist_ok=True);raw=P/'benchmark_runs'/run.name
for kind in ['integration','ab','accuracy','profile']:
 stage=Path((R/f'feedback/prefill-value-{kind}-run.txt').read_text().strip());assert read(stage/'trial-status.json')['status']=='complete';shutil.copy2(B/stage.name/'results.json',out/(kind+'-results.json'));shutil.copy2(stage/'trial-status.json',out/(kind+'-trial-status.json'))
 if kind=='accuracy':shutil.copy2(stage/'quality-comparisons.json',out/'quality-comparisons.json')
shutil.copy2(B/'prefill-gated-profile-20261116T034451Z/results.json',out/'baseline-profile-results.json')
for name in ['manifest.json','paired-prefill-audit.json','protected-correctness.json','cleanup-check.json','remaining-validation-status.json','next-options.md']:shutil.copy2(run/name,out/name)
for path in run.glob('gdn-cast-breakdown-*.json'):shutil.copy2(path,out/path.name)
for path in raw.iterdir():
 if path.is_file():shutil.copy2(path,out/path.name)
files=['init_prefill_value.py','setup_prefill_value_harness.py','request_prefill_value_author.py','place_prefill_value_integration.py','prepare_prefill_value_trial.py','launch_prefill_value_prepared.py','run_prefill_value_integration.py','run_prefill_value_ab.py','run_prefill_value_accuracy.py','run_prefill_value_candidate_profile.py','finish_prefill_value_validation.py','run_prefill_value_preserved_regression.py','summarize_prefill_value_protected.py','check_prefill_value_cleanup.py','compare_prefill_value_results.py','checkpoint_prefill_value.py','finalize_prefill_value_docs.py','report_prefill_value.py','diagnose_gdn_value_cpu.py','nvfp4_update_runtime.py','shutdown_process_probe.py','compare_accuracy_evaluation.py','acceptance/check_gdn_value_integration.py','acceptance/benchmark_prefill_value_ab.py','acceptance/profile_prefill_value_candidate.py','acceptance/qwen_ops_before_value_processing.py.txt','acceptance/request_nvfp4_candidate.py','acceptance/quant_profile_summary.py','acceptance/evaluate_quant_accuracy.py','prompts/nvfp4-task-054.txt']
for name in files:shutil.copy2(R/name,h/Path(name).name)
(h/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in h.iterdir() if p.name!='SHA256.json'},indent=2)+'\n')
rows=[];profiles=[];peaks=[]
for c in a['rows']:
 old,new=c['paired_prefill']['original'],c['paired_prefill']['candidate'];rows.append('| {} | {:.2f} → {:.2f} ms | {:.2f}% | {:.1f} → {:.1f} |'.format(c['prompt_tokens'],old['median_prefill_cuda_ms'],new['median_prefill_cuda_ms'],c['less_prefill_time_pct'],old['prompt_tokens_per_second_cuda'],new['prompt_tokens_per_second_cuda']));o,n=c['gdn_profile_before'],c['gdn_profile_after'];profiles.append('| {} | {:.2f} → {:.2f} ms | {} → {} |'.format(c['prompt_tokens'],o['gpu_ms'],n['gpu_ms'],o['kernels'],n['kernels']));p=c['median_peak_allocated_bytes'];peaks.append('{} tokens: {:.3f} → {:.3f} GiB'.format(c['prompt_tokens'],p['original']/1024**3,p['candidate']/1024**3))
text='''# GDN value-cast deferral

Local Qwen task 054 authored the only engine change, checkpoint **ENGINE**. Baseline engine ea70e47 includes the accepted QKV, FP8 preparation and normalization optimizations. Rejected state-tail and paired-solve candidates remain excluded. The supervisor wrote the harness, diagnostics, prompts and report; engine source is placed verbatim from the retained local response.

## Warm prefill result

One resident native 27B NVFP4 model alternates baseline/candidate/candidate/baseline four times, eight warm samples per method per length. Direct CUDA-event model.prefill excludes loading, tokenization and compilation. All four timing blocks improve at both lengths. These small repeated measurements are not a confidence interval or a claim for arbitrary workloads. Percentages from separate experiments must not be added.

| Prompt tokens | Baseline → accepted prefill | Less processing time | Prompt tokens/sec baseline → accepted |
| --- | --- | --- | --- |
ROWS

Median peak CUDA allocated memory: PEAKS. The 1,024-token peak increases by 11.1 MiB; no peak-memory saving is claimed. This is per-process tensor allocation, not total host/unified memory. Generation is a correctness gate here; no sustained output-token speedup is claimed.

## Change and precision

The original GDN chunk function eagerly converts all values to FP32 before their first use. For BF16 queries and BF16 CUDA values in inference mode, the new function retains the transposed BF16 view. The existing value-times-FP32-beta product promotes to FP32 directly. Output storage and both initial-state allocation/conversion explicitly remain FP32. All query/key normalization, two original triangular solves, chunk/padding logic, exponentials, GEMMs and state propagation expressions remain unchanged. CPU, other dtypes, no_grad and autograd retain the original widening path. No new kernel, module, dependency or cache mutation is introduced.

## Validation

- 298 direct cases, 32 dense/mixed-quantized blocks, 20 tiny whole models, 2 exact gradient cases and 11 extra CPU/provided-state-dtype cases pass. Inputs and supplied states remain unmodified. Output dtype, FP32 state, decode continuation and caches match exactly. Observational instrumentation sees zero value.float calls in eligible inference, one in the original/fallback path. These are component/tiny tests, distinct from real-checkpoint validation.
- Real27B at256/1,024 tokens preserves entire prefill logits and every cache tensor, next-decode logits/cache, and two repeated 128-ID generations per backend and length exactly.
- Frozen 40 quality sequences and all 446 corpus token losses match the accepted native baseline exactly. Perplexity PPL. The frozen vLLM-reference quality screen passes; this does not expand independent full-logit quantized reference coverage.
- All 56 Qwen and 15 Llama protected gates pass, including separate original 9B/27B numeric and conversation checks. Fixture-source hashes and numerical tolerances are unchanged.

## Matched profile

Each profile generates exactly one token and asserts zero decode calls. Prompt IDs and normalization input shapes match the preceding accepted profile. Forty-eight GDN kernels disappear, one eager value-copy kernel per GDN layer. Other normalization and quantized GEMM kernel counts remain unchanged.

| Prompt tokens | GDN GPU time baseline → accepted | GDN kernels baseline → accepted |
| --- | --- | --- |
PROFILES

Profiled GPU times include profiler overhead and are separate from unprofiled A/B latency. Inclusive scope times overlap; do not sum parent and child times. Raw large traces and model weights remain outside Git.

## Containment and next priority

All models ran serially, with per-worker memory/time/swap guards, source hashes and same-boot checks. Final four experiment containers are stopped exit0/OOMfalse, service MainPID 0, no GPU jobs; STOP/GPU/LARGE holds retained. cleanup-check.json contains actual memory telemetry. No duplicate controller or unrelated service changes.

Model unsloth/Qwen3.8-27B-NVFP4 revf0b7c9e722f5565102fff8481c99e4d86ae099c7; experimental Torch 2.13+cu130 and original protected Torch 2.11 runtime. Main run RUN.

The next-options.md review prioritizes batching independent GDN chunk-local preparation before the sequential state scan. GDN state processing remains the largest measured scope (~36% before this change). That is an unimplemented option with numerical/workspace risks, not a promised improvement. Stop extending tiny pointwise experiments without a measured reason.

Reproduce using the existing installed containers, no other workers:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_prefill_value_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_prefill_value_prepared.py ab
```
'''.replace('ENGINE',m['engine_commit'][:7]).replace('ROWS','\n'.join(rows)).replace('PROFILES','\n'.join(profiles)).replace('PEAKS','; '.join(peaks)).replace('PPL',str(a['quality_perplexity'])).replace('RUN',str(run))
(out/'README.md').write_text(text);print(out/'README.md')
