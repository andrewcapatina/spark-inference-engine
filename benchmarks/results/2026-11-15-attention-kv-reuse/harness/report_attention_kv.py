"""Archive validated evidence and reproducible harness; no engine edits."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference';run=Path((R/'feedback/attention-kv-main-run.txt').read_text().strip());dest=P/'benchmarks/results/2026-11-15-attention-kv-reuse'
def read(p):return json.loads(p.read_text())
manifest=read(run/'manifest.json');assert manifest['current_engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
assert read(run/'remaining-validation-status.json')['status']=='complete';assert read(run/'protected-correctness.json')['status']=='all_protected_gates_passed';audit=read(run/'paired-results-audit.json');assert audit['status']=='cpu_crosschecks_passed'
assert (P/'spark_engine/qwen_layers.py').read_bytes()==(P/'benchmark_runs'/run.name/'task-022-placed.py.txt').read_bytes()
dest.mkdir(exist_ok=True);h=dest/'harness';h.mkdir(exist_ok=True)
for kind,name in [('small','tiny-exact-checks'),('ab','same-model-ab'),('accuracy','quality-native'),('profile','profile-native')]:
 trial=Path((R/f'feedback/attention-kv-{kind}-run.txt').read_text().strip());state=read(trial/'trial-status.json');assert state['status']=='complete' and state['engine_sha256']==manifest['current_engine_sha256'];shutil.copy2(B/trial.name/'results.json',dest/(name+'.json'));shutil.copy2(trial/'trial-status.json',dest/(kind+'-trial-status.json'))
 if kind=='accuracy':shutil.copy2(trial/'quality-comparisons.json',dest/'quality-comparisons.json')
for name in ['protected-correctness.json','paired-results-audit.json','remaining-validation-status.json','manifest.json','cleanup-check.json']:shutil.copy2(run/name,dest/name)
for name in ['task-022-response.json','task-022-candidate.txt','task-022-review.json']:shutil.copy2(P/'benchmark_runs'/run.name/name,dest/name)
files=['report_attention_kv.py','check_attention_kv_cleanup.py','nvfp4_update_runtime.py','shutdown_process_probe.py','compare_accuracy_evaluation.py','acceptance/evaluate_quant_accuracy.py','acceptance/run_capped_pytest.py','prepare_attention_kv_trial.py','launch_attention_kv_prepared.py','run_attention_kv_small.py','run_attention_kv_ab.py','run_attention_kv_accuracy.py','run_attention_kv_profile.py','run_attention_kv_preserved_regression.py','finish_attention_kv_validation.py','summarize_attention_kv_protected.py','compare_attention_kv_results.py','acceptance/check_attention_kv_reuse.py','acceptance/benchmark_attention_kv_ab.py','acceptance/profile_qkv_runtime.py','acceptance/qwen_attention_decode_before_kv_reuse.py.txt','acceptance/quant_profile_summary.py']
for name in files:shutil.copy2(R/name,h/Path(name).name)
(dest/'harness-sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in h.iterdir()},indent=2)+'\n')
rows=[]
for row in audit['rows']:
 t=row['paired_timings'];rows.append('| {} | {:.3f} → {:.3f} | {} | {:.1f} MiB |'.format(row['prompt_tokens'],t['original']['median_total128output_tok_s'],t['reuse']['median_total128output_tok_s'],row['profile_gpu_kernels_before']-row['profile_gpu_kernels_after'],row['saved_cumulative_cat_allocated_bytes']/1024**2))
text='''# Reuse attention KV concatenations

Local Qwen task022 changes only `QwenAttention.decode`: save the first, unrepeated K/V concatenations and return them as the cache, instead of concatenating the same tensors again. Group expansion remains temporary attention work. The supervisor wrote the harness and placed the retained local-model source verbatim. Accepted QKV reuse remains present; convolution-history compaction stays quarantined. No preallocation is included here.

## Measurements

One resident native27B NVFP4 model alternates original/reuse/reuse/original. Two warm uninstrumented 128-output timing samples per method and prompt exclude model loading. Small throughput differences do not establish a meaningful speedup.

| Prompt tokens | Total 128-output tokens/s, old → reuse | Fewer CUDA kernels per 16-output profile | Fewer cumulative cat allocation requests |
| --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

The profile loses exactly480 decode concatenations and480 kernels at each length. All decode `aten::cat` calls fall2655→2175; the remaining calls include other engine work. Allocation-request totals are cumulative, not peak memory or measured memory bandwidth. Prefill is unchanged by this method; its small timing variations are noise. Prefill peak allocation is identical in the paired runs. This is a small cleanup with reduced redundant work, not a substantial performance optimization.

## Validation

- 110 tiny dense/mixed-FP8 numerical cases pass exactly, including batch1/2, grouped and ungrouped attention, noncontiguous hidden inputs, three decode steps and branching. Prefixes and retained caches remain unchanged; KV-axis concatenations fall4→2.
- Real27B256/1024 prompts preserve entire prefill logits/cache values exactly, and all128 generated IDs agree across all alternating runs. Failure diagnostics retain expected and actual token IDs before raising.
- Frozen quality screen40/40, all generated IDs and446 token losses exactly match accepted QKV baseline. Weighted perplexity remains8.946730. The fixed vLLM-reference screen also passes. This remains a small screen; broad quantized reference parity is not claimed.
- All56 unique Qwen gates and15 Llama cases pass, including separate real9B/27B numeric and conversation processes. Original fixture hashes and numerical tolerances are unchanged.
- Profile inputs and outputs match baseline/control/paired-generation prefixes. All CUDA kernels are attributed. Profile overhead is excluded from the 128-output throughput measurements.
- Native workers release CUDA allocations/reservations to0/0; containers exit cleanly with no OOM, host boot unchanged and bounded swap growth recorded in cleanup evidence. One large model at a time, service MainPID0 and global holds retained. Cleanup evidence is retained separately.

## Reproduction and artifacts

Use the installed pinned runtime image `sha256:fa7334538384d67c72284edc081c78b1149a36d53aba1fb3c5f7d1d534154ec6` (Torch2.13+cu130), `unsloth/Qwen3.8-27B-NVFP4` revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`, and this exact source manifest. Protected original-model tests use their preserved runtime. Main run: `'''+str(run)+'''`.

With no experiment CLI/controller running:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_attention_kv_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_attention_kv_prepared.py ab
python3 /home/cap/spark-inference-overnight-20261113/finish_attention_kv_validation.py
```

Result JSON, source provenance and exact harness copies are included here; weights and large raw traces stay outside Git. Next experiment: per-request capacity-backed attention cache with preserved snapshots and branch ownership, evaluated independently before adoption.
'''
(dest/'README.md').write_text(text)
print(dest)
