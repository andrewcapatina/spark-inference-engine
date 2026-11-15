"""Archive fully validated prefill improvement; report only measured prefill metrics."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference';run=Path((R/'feedback/prefill-fp8-main-run.txt').read_text().strip())
def read(p):return json.loads(p.read_text())
m=read(run/'manifest.json');assert m['status']=='validated_checkpointed';assert read(run/'remaining-validation-status.json')['status']=='complete';assert read(run/'protected-correctness.json')['status']=='all_protected_gates_passed';assert read(run/'cleanup-check.json')['status']=='clean';audit=read(run/'paired-prefill-audit.json');assert audit['status']=='prefill_cpu_audit_passed';assert m['current_engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
out=P/'benchmark_runs'/run.name;assert (P/'spark_engine/qwen_fp8.py').read_bytes()==(out/'task-029-placed.py.txt').read_bytes();root=P/'benchmarks/results/2026-11-15-prefill';dest=root/'fp8-reduction';dest.mkdir(parents=True,exist_ok=True);h=dest/'harness';h.mkdir(exist_ok=True)
for kind in ['small','ab','accuracy','profile']:
 stage=Path((R/f'feedback/prefill-fp8-{kind}-run.txt').read_text().strip());state=read(stage/'trial-status.json');assert state['status']=='complete' and state['engine_sha256']==m['current_engine_sha256'];shutil.copy2(B/stage.name/'results.json',dest/(kind+'-results.json'));shutil.copy2(stage/'trial-status.json',dest/(kind+'-trial-status.json'))
 if kind=='accuracy':shutil.copy2(stage/'quality-comparisons.json',dest/'quality-comparisons.json')
for name in ['manifest.json','paired-prefill-audit.json','protected-correctness.json','cleanup-check.json','remaining-validation-status.json','fixture-recovery.json','initial-fixture-harness-reconstructed.py.txt']:shutil.copy2(run/name,dest/name)
failed=R/'runs/prefill-fp8-small-20261115T143039Z';shutil.copy2(failed/'trial-status.json',dest/'initial-fixture-failure-status.json');shutil.copy2(failed/'27b-worker.log',dest/'initial-fixture-failure.log')
for name in ['task-029-response.json','task-029-candidate.txt','task-029-review.json']:shutil.copy2(out/name,dest/name)
shutil.copy2(out/'task-029-previous.py.txt',dest/'baseline-qwen_fp8.py.txt');shutil.copy2(P/'spark_engine/qwen_fp8.py',dest/'accepted-qwen_fp8.py.txt')
files=['acceptance/check_prefill_fp8_reduce.py','acceptance/benchmark_prefill_fp8_ab.py','acceptance/fp8_forward_before_prefill_reduce.py.txt','acceptance/profile_prefill_runtime.py','acceptance/quant_profile_summary.py','acceptance/test_native_fp8.py','acceptance/test_quantized_loader.py','acceptance/evaluate_quant_accuracy.py','compare_accuracy_evaluation.py','prepare_prefill_fp8_trial.py','launch_prefill_fp8_prepared.py','run_prefill_fp8_small.py','run_prefill_fp8_ab.py','run_prefill_fp8_accuracy.py','run_prefill_fp8_profile.py','run_prefill_fp8_preserved_regression.py','finish_prefill_fp8_validation.py','compare_prefill_fp8_results.py','summarize_prefill_fp8_protected.py','check_prefill_fp8_cleanup.py','place_prefill_fp8_candidate.py','request_gdn_storage_author.py','acceptance/request_nvfp4_candidate.py','nvfp4_update_runtime.py','shutdown_process_probe.py','prompts/nvfp4-task-029.txt','report_prefill_results.py']
for name in files:shutil.copy2(R/name,h/Path(name).name)
(dest/'harness-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in h.iterdir()},indent=2)+'\n')
rows=[];prep=[]
for c in audit['rows']:
 old,new=c['paired_prefill']['original'],c['paired_prefill']['reuse'];rows.append('| {} | {:.2f} → {:.2f} ms | {:.1f}% | {:.1f} → {:.1f} |'.format(c['prompt_tokens'],old['median_prefill_cuda_ms'],new['median_prefill_cuda_ms'],c['less_prefill_time_pct'],old['prompt_tokens_per_second_cuda'],new['prompt_tokens_per_second_cuda']));prep.append('| {} | {:.2f} → {:.2f} ms | {:.2f} → {:.2f} GiB |'.format(c['prompt_tokens'],c['fp8_preparation_profile_before']['gpu_ms'],c['fp8_preparation_profile_after']['gpu_ms'],c['fp8_preparation_allocation_before']['allocated_bytes_sum']/1024**3,c['fp8_preparation_allocation_after']['allocated_bytes_sum']/1024**3))
text='''# FP8 input preparation: accepted prefill improvement

Engine checkpoint **'''+m['engine_commit'][:7]+'''**, authored by Local Qwen, changes one expression in `FP8Linear.forward`. It computes the absolute-value row maximum in BF16, then widens the small row-max result to FP32 before the original division by 448 and clamp to 1e-6. The old method widened the entire activation matrix first.

For finite BF16 inputs, maximum selects an exactly representable input magnitude; widening that selected value preserves the original FP32 maximum. Scale dtype, division/clamp precision, quantized activation bytes, matrix-multiply settings, padding and output shape remain unchanged. Tests verify those properties directly. No new quantization approximation, compiler, kernel backend or cache mechanism is introduced. The supervisor wrote the harness/tests/docs and placed the local-model source verbatim.

## Primary result: prompt processing

One resident native 27B NVFP4 model alternated baseline/candidate/candidate/baseline twice: four warm samples per method and length. These measurements exclude loading and tokenization and use direct `model.prefill` CUDA events; wall-time medians agree. Generated output throughput is not the objective or acceptance metric. The improvement is additional to the already accepted QKV reuse.

| Prompt tokens | Prefill latency, baseline → accepted | Less prefill time | Prompt tokens/sec, baseline → accepted |
| --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

Both ABBA blocks improve at each length (2.3–3.1% at 256; 5.6–5.8% at 1024). This is a small repeated workload measurement, not a statistical confidence interval or a claim about all contexts/models. A separate first-token CUDA diagnostic agrees; it excludes load/tokenization and records zero decode calls.

## Prefill-only profile

The new profile generates exactly one token and asserts zero decode calls. Only prefill scopes are compared with the earlier accepted-engine profile; whole traces are not compared because the earlier trace also contained decoding. Component GPU time is separate from unprofiled paired latency.

| Prompt tokens | FP8 input-preparation GPU time, before → after | Cumulative FP8 preparation allocation requests, before → after |
| --- | --- | --- |
'''+ '\n'.join(prep)+'''

The roughly 65 ms reduction in the 1,024-token FP8 preparation scope is consistent with the roughly 65 ms saved in paired prefill. FP8 projection count remains 233, its preparation kernel count remains 1,631, and GDN state-kernel count is unchanged. Overall prefill kernel counts remain 14,906 / 37,274. The change processes smaller temporary data rather than reducing kernel count. Allocation-request totals above are cumulative transient work, not peak memory or measured memory bandwidth. All CUDA kernels are attributed.

## Correctness and preserved behavior

- 149 projection cases preserve exact FP32 scales, FP8 activation bytes and BF16 GEMM outputs. Coverage includes all 65,280 finite BF16 bit patterns, zeros/subnormals, clamp boundaries, large finite values, vector inputs, padding boundaries, batch dimensions and noncontiguous tensors. Exhaustive value coverage concerns activation quantization, not every possible matrix/model.
- 20 tiny mixed-quantized model cases preserve full prefill logits/cache, next decode logits/cache and 16 generated IDs exactly. The unchanged independent FP8 and quantized-loader tests also pass with their original tolerances.
- Real 27B at 256 / 1,024 tokens preserve entire prefill logits/every cache tensor, complete next-decode logits/cache and all 128 generated IDs exactly. Generation is a correctness check only.
- The frozen 40-case quality screen passes 40/40; all 40 generated sequences and all 446 corpus token losses match the accepted baseline exactly. Weighted perplexity remains 8.946730. The frozen vLLM-reference screen also passes; broad independent quantized full-logit parity remains limited to previously documented coverage.
- All 56 unique Qwen and 15 Llama gates pass, including separate original 9B and 27B numeric and conversation processes. Acceptance-source hashes and numerical tolerances are unchanged. No two large models are loaded together.

The first new extreme-range fixture used random projection weights and added a finiteness assertion that even the baseline could not meet: large matrix sums overflowed, after exact scale/byte/output comparisons had already passed. The exhaustive-range fixture now uses bounded identity weights, while random-weight tests and every exact/finiteness assertion remain. The failed run, logs, recovery note and a clearly labelled reconstructed pre-fix harness are retained. No engine code was changed for this harness recovery.

Every completed native stage releases CUDA allocations/reservations to 0/0; final container/service/GPU/swap/boot evidence is in `cleanup-check.json`. Legacy service and GPU holds remain inactive/held as before. The rejected GDN mask/intermediate-reuse experiment is documented separately and was restored before this FP8 experiment; convolution-history, compiler and preallocated-cache candidates remain excluded.

## Provenance and reproduction

Model: `unsloth/Qwen3.8-27B-NVFP4`, revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`. Runtime: installed pinned Torch 2.13+cu130 experiment container, with its exact image ID recorded in cleanup evidence. Protected original-model tests use their preserved runtime. Main run: `'''+str(run)+'''`.

The result JSON, fixed quality comparisons, protected fixture hashes, raw local-model response/review, before/after source and exact harness/checksums are included. Weights and large raw traces remain outside Git.

With this engine source, the installed references/runtime and no active experiment CLI/controller:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_prefill_fp8_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_prefill_fp8_prepared.py ab
```

The harness switches retained baseline versus accepted FP8 projection on the same loaded weights. Its primary metrics are prefill milliseconds and prompt tokens/sec.
'''
(dest/'README.md').write_text(text)
(root/'README.md').write_text('''# Prefill optimization results

This package targets prompt processing, measured as warm prefill latency and prompt tokens/sec. Loading, tokenization and sustained output-token throughput are separate.

The accepted [FP8 input-preparation change](fp8-reduction/README.md), engine **'''+m['engine_commit'][:7]+'''**, reduces prefill latency by **2.7% at 256 prompt tokens** and **5.6% at 1,024** on the tested native 27B NVFP4 checkpoint. The one-line engine change was authored by local Qwen; exact numerical behavior and all protected checks are preserved.

'''+ '\n'.join(['| Prompt tokens | Before → after | Prompt tokens/sec before → after |','| --- | --- | --- |']+['| {} | {:.2f} → {:.2f} ms | {:.1f} → {:.1f} |'.format(c['prompt_tokens'],c['paired_prefill']['original']['median_prefill_cuda_ms'],c['paired_prefill']['reuse']['median_prefill_cuda_ms'],c['paired_prefill']['original']['prompt_tokens_per_second_cuda'],c['paired_prefill']['reuse']['prompt_tokens_per_second_cuda']) for c in audit['rows']])+'''

The first [GDN mask/intermediate-reuse experiment](gdn-reuse/README.md) passed exact checks but showed no prefill gain and was not adopted. It is excluded from the accepted engine. Earlier decode-cache experiments are separate and were not used to judge this prefill result.

The initial profile put GDN state computation, normalization and FP8 preparation at roughly 357 / 196 / 143 ms for 1,024 tokens. The accepted change lowers FP8 preparation to about 78 ms in the new prefill-only profile. Normalization and deeper GDN changes remain options for a later measured proposal; no implementation of those options is included here.
''')
print(root/'README.md')
