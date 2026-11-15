"""Archive an unadopted local-model experiment after byte-exact restoration."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference';run=Path((R/'feedback/preallocated-kv-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());assert m['status']=='retained_not_adopted_no_material_gain';assert m['engine_before']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')};audit=json.loads((run/'paired-results-audit.json').read_text());assert audit['status']=='cpu_crosschecks_passed';dest=P/'benchmarks/results/2026-11-15-preallocated-kv';dest.mkdir(exist_ok=True);h=dest/'harness';h.mkdir(exist_ok=True);sources=dest/'candidate-source';sources.mkdir(exist_ok=True)
for kind in ['small','decode','ab','profile']:
 stage=Path((R/f'feedback/preallocated-kv-{kind}-run.txt').read_text().strip());assert json.loads((stage/'trial-status.json').read_text())['status']=='complete';shutil.copy2(B/stage.name/'results.json',dest/(kind+'-results.json'));shutil.copy2(stage/'trial-status.json',dest/(kind+'-trial-status.json'))
failed=R/'runs/preallocated-kv-small-20261115T125845Z/trial-status.json';shutil.copy2(failed,dest/'initial-preflight-failure.json')
for name in ['manifest.json','paired-results-audit.json','cleanup-check.json']:shutil.copy2(run/name,dest/name)
retained=Path(m['retained_candidate'])
for path in retained.glob('*.py'):shutil.copy2(path,sources/(path.name+'.txt'))
for path in (P/'benchmark_runs'/run.name).glob('task-0*'):
 if path.name.endswith(('response.json','candidate.txt','review.json','retained-foundation.py.txt')):shutil.copy2(path,dest/path.name)
files=['check_preallocated_kv_cleanup.py','prepare_preallocated_kv_experiment.py','prepare_preallocated_kv_trial.py','launch_preallocated_kv_prepared.py','run_preallocated_kv_small.py','run_preallocated_kv_decode.py','run_preallocated_kv_ab.py','run_preallocated_kv_profile.py','compare_preallocated_kv_results.py','retain_and_restore_preallocated_kv.py','place_preallocated_kv_bundle.py','place_preallocated_kv_decode.py','report_preallocated_kv.py','nvfp4_update_runtime.py','shutdown_process_probe.py','acceptance/check_preallocated_attention_cache.py','acceptance/check_preallocated_attention_decode.py','acceptance/benchmark_preallocated_kv_ab.py','acceptance/profile_qkv_runtime.py','acceptance/quant_profile_summary.py','acceptance/qwen_attention_decode_before_preallocation.py.txt']
for name in files:shutil.copy2(R/name,h/Path(name).name)
(dest/'harness-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in h.iterdir()},indent=2)+'\n')
for number in ['023','024','025','026','027']:shutil.copy2(R/f'prompts/nvfp4-task-{number}.txt',h/f'task-{number}-prompt.txt')
rows=[];details=[]
for row in audit['rows']:
 t=row['paired_timing'];saved=(row['allocator_before']['allocated_bytes_sum']-row['allocator_after']['allocated_bytes_sum'])/1024**2;rows.append('| {} | {:.3f} → {:.3f} | {:+.2f}% | {:.1f} MiB |'.format(row['prompt_tokens'],t['original']['median_total128output_tok_s'],t['reuse']['median_total128output_tok_s'],row['throughput_change_pct'],saved));details.append('{} tokens: profile kernels {}→{}, decode concatenations {}→{}, peak allocation {:.3f}→{:.3f} GiB.'.format(row['prompt_tokens'],row['kernels_before'],row['kernels_after'],row['decode_cat_before']['calls'],row['decode_cat_after']['calls'],row['profile_peak_allocated_before']/1024**3,row['profile_peak_allocated_after']/1024**3))
text='''# Preallocated attention-cache experiment: retained, not enabled

The local model implemented per-request capacity-backed K/V buffers, with a shared append cursor and copy-on-branch ownership. The supervisor wrote the harness, reviewed candidates and assembled/placed exact retained local-model source. No supervisor-written engine implementation was used.

The experiment passes its targeted numerical checks but does not establish a useful end-to-end speedup at the measured lengths. Its extra ownership machinery is not enabled. The complete candidate was retained before exact restoration to accepted engine6621762 (documentation baselineaf5bc77), including removal of the two experimental helper modules. Accepted QKV reuse and duplicate-concatenation reuse remain; convolution-history compaction stays quarantined.

## Paired measurements

One resident native27B NVFP4 model alternates accepted two-concatenation decode/preallocated/preallocated/accepted. Two warm uninstrumented128-output samples per method and length exclude loading. Differences below0.5% in this small sample do not establish a robust speed benefit.

| Prompt tokens | Total128-output tokens/s, accepted → experimental | Change | Fewer cumulative allocator-request bytes per separate16-output profile |
| --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

The256-prompt decode-only diagnostic changes9.169→9.212tok/s; the1024-prompt diagnostic changes8.911→8.897tok/s. Prefill is unchanged by this decode-only work. Saved allocation requests represent cumulative transient allocation, not peak memory or measured bandwidth. Profile timings include instrumentation overhead and are separate from the128-output throughput measurements.

'''+ '\n\n'.join(details)+'''

Removing concatenations does not remove all copy launches: initial buffer creation copies the prefix and every append still writes new K/V. Raw profiles and operator counts provide the evidence; kernel counts alone are not throughput.

## What was actually validated

- Standalone helper:144 exact value/storage/snapshot/branch cases,696 observed sequential backing-storage reuses, FP32/BF16, batch1/2, KVheads1/2/4, strided inputs, prefix lengths0/1/63/64/65/255/256/257/511/512/513/1024 and interleaved branches. Four autograd cases and separate no_grad, requires-grad-inside-inference and dtype-promotion fallbacks pass. These are synthetic helper tests.
- Integrated attention decode:110 dense/mixed-FP8 tiny cases preserve output/cache/continuation and branch values exactly; two KV concatenations per decode become zero in inference mode.
- Real27B256/1024: complete prefill logits/cache and complete next-decode logits/all cache values are exact. All128 generated token IDs match across four alternating runs at each prompt length.
- Profile inputs/output IDs agree with accepted-engine profiles and128-output prefixes; all CUDA kernels are attributed. Native CUDA allocations/reservations release to0/0 after each worker.
- The experimental candidate did not run the full frozen40-case quality screen or protected original-model gates: the measured performance did not justify adoption. The restored accepted engine already passed all56Qwen/15Llama gates and the40-case screen, and every engine source hash matches that accepted baseline. Numerical tests/tolerances were not weakened.

## Recovery and provenance

Initial local023/024 append implementations were rejected before placement for concrete representation/ownership errors. Local024's tuple wrapper was retained verbatim; local025 provided the private fast core, local026 the public inference/gradient/dtype guard, and local027 the one-method decode integration. Raw responses, reviews, prompts and full candidate source are archived. Mechanical assembly removed only outer Markdown fences/trailing-newline differences; no implementation was repaired by the supervisor.

The first standalone trial stopped BEFORE execution at about55.7GiB CUDA free, below the unchanged56GiB floor. Existing inactive NVFP4 cache hints did not recover enough. Bounded hints for the inactive coding-model GGUF (at most8GiB, in2GiB windows) recovered enough headroom; file data/size/mtime were unchanged. The failed trial and successful retry are retained. Host/new-swap/CUDA allocator guards remained unchanged. No duplicate controller or concurrent large-model load was used.

Final cleanup/restoration evidence is in `cleanup-check.json`, source hashes in `manifest.json`, and measured outputs/profile summaries in the result JSON files. Weights and large raw traces remain outside Git. Runtime is the installed pinned Torch2.13+cu130 image and the same `unsloth/Qwen3.8-27B-NVFP4` revision used in the accepted attention-cache report. Main run: `'''+str(run)+'''`.

The `candidate-source/` files are archived text, not active engine modules. Reproducing this experiment requires deliberately restoring that isolated source in the experiment environment; the normal engine retains the accepted implementation. This report is not a claim that preallocation is production-ready or validated across all models/contexts.
'''
(dest/'README.md').write_text(text);print(dest)
