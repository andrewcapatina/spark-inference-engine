"""Archive validation and local-model provenance; no implementation changes."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');run=Path((R/'feedback/prefill-norm-main-run.txt').read_text().strip());B=R/'feedback/nvfp4-reference'
def read(p):return json.loads(p.read_text())
m=read(run/'manifest.json');a=read(run/'paired-prefill-audit.json');assert m['status']=='validated_checkpointed';assert a['status']=='normalization_prefill_audit_passed';assert read(run/'cleanup-check.json')['status']=='clean';assert read(run/'protected-correctness.json')['status']=='all_protected_gates_passed';assert read(run/'remaining-validation-status.json')['status']=='complete'
assert m['current_engine_sha256']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
out=P/'benchmarks/results/2026-11-15-normalization';out.mkdir(parents=True,exist_ok=True);h=out/'harness';h.mkdir(exist_ok=True);raw=P/'benchmark_runs'/run.name
for kind in ['baseline','small','integration','ab','accuracy','profile']:
 stage=Path((R/f'feedback/prefill-norm-{kind}-run.txt').read_text().strip());assert read(stage/'trial-status.json')['status']=='complete';shutil.copy2(B/stage.name/'results.json',out/(kind+'-results.json'));shutil.copy2(stage/'trial-status.json',out/(kind+'-trial-status.json'))
 if kind=='accuracy':shutil.copy2(stage/'quality-comparisons.json',out/'quality-comparisons.json')
for name in ['manifest.json','paired-prefill-audit.json','protected-correctness.json','cleanup-check.json','remaining-validation-status.json','baseline-profile-recovery.json','initial-profile-harness.py.txt','protected-cache-recovery.json','protected-harness-before-cache-recovery.py.txt','protected-before-cache-recovery-status.json','qwen-real9b-conversation-before-cache-recovery.log','protected-exec-cache-recovery.json','protected-noexec-cache-status.json','protected-harness-noexec-cache.py.txt','qwen-real9b-conversation-noexec-cache.log','legacy-cache-small-kernel.log','normalization-legacy-cache-preflight-status.json']:
 shutil.copy2(run/name,out/name)
for p in raw.iterdir():
 if p.is_file():shutil.copy2(p,out/p.name)
assert (P/'spark_engine/qwen_ops.py').read_bytes()==(raw/'task-035-placed.py.txt').read_bytes();assert (P/'spark_engine/qwen_norm_tail.py').read_bytes()==(raw/'task-033-placed.py.txt').read_bytes()
files=['prepare_prefill_norm_trial.py','launch_prefill_norm_prepared.py','request_prefill_norm_author.py','place_prefill_norm_helper.py','place_prefill_norm_integration.py','run_prefill_norm_small.py','run_prefill_norm_integration.py','run_prefill_norm_ab.py','run_prefill_norm_accuracy.py','run_prefill_norm_profile.py','run_prefill_norm_candidate_profile.py','run_prefill_norm_preserved_regression.py','finish_prefill_norm_validation.py','compare_prefill_norm_results.py','summarize_prefill_norm_protected.py','check_prefill_norm_cleanup.py','report_prefill_norm.py','checkpoint_prefill_norm.py','check_prefill_norm_legacy_cache.py','compare_accuracy_evaluation.py','nvfp4_update_runtime.py','shutdown_process_probe.py','acceptance/check_norm_tail.py','acceptance/check_norm_integration.py','acceptance/benchmark_prefill_norm_ab.py','acceptance/profile_prefill_norm_runtime.py','acceptance/profile_prefill_norm_candidate.py','acceptance/quant_profile_summary.py','acceptance/evaluate_quant_accuracy.py','acceptance/request_nvfp4_candidate.py']
files += ['prompts/nvfp4-task-'+f'{n:03d}'+'.txt' for n in range(30,36)]
for name in files:shutil.copy2(R/name,h/Path(name).name)
(h/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in h.iterdir() if p.name!='SHA256.json'},indent=2)+'\n')
rows=[];profiles=[]
for c in a['rows']:
 old,new=c['paired_prefill']['original'],c['paired_prefill']['candidate'];rows.append('| {} | {:.2f} → {:.2f} ms | {:.2f}% | {:.1f} → {:.1f} |'.format(c['prompt_tokens'],old['median_prefill_cuda_ms'],new['median_prefill_cuda_ms'],c['less_prefill_time_pct'],old['prompt_tokens_per_second_cuda'],new['prompt_tokens_per_second_cuda']));profiles.append('| {} | {:.2f} → {:.2f} ms | {} → {} |'.format(c['prompt_tokens'],c['plain_norm_profile_before']['gpu_ms'],c['plain_norm_profile_after']['gpu_ms'],c['plain_norm_profile_before']['kernels'],c['plain_norm_profile_after']['kernels']))
text='''# Ordinary normalization: accepted prefill improvement

Engine **ENGINE**, authored by Local Qwen (raw tasks033/035), adds an inference-only Triton pointwise tail to ordinary `qwen_rms_norm`. Baseline checkpoint tag `checkpoint-prefill-fp8-20261115` protects engine2beb3fd/report e085ea0. The supervisor authored the harness, prompts, diagnostics and documentation, and placed reviewed local-model source verbatim.

The selected baseline profile showed ordinary/gated normalization at 114.40/78.49ms for 1,024 prompt tokens. This candidate targets only the ordinary pointwise tail. Original FP32 conversion, square/mean reduction and reciprocal square root remain eager Torch. The new kernel multiplies by that inverse, adds one to FP32 weight, multiplies and casts to BF16, with floating-point fusion explicitly disabled to preserve rounding.

## Warm prompt-processing result

One resident native 27B NVFP4 model, two ABBA blocks, four warm samples per method and length. CUDA-event prefill latency excludes loading, tokenization, import and first-use compilation. Generation validates correctness only.

| Prompt tokens | Before → after | Less prefill time | Prompt tokens/sec before → after |
| --- | --- | --- | --- |
ROWS

Both ABBA blocks are retained in `paired-prefill-audit.json`. These small repeated measurements establish improvement on these workloads, not all contexts or hardware. Gains are additional to the accepted FP8 and QKV changes; older measurements cannot be added together to estimate a cumulative percentage.

Fresh private Triton-cache testing records approximately 1.00s initial compiler startup and roughly 0.03s for later width/dtype specializations. Real first candidate prefill at 256 tokens took 1.255s, compared with roughly 0.266s warm. The baseline ran first and warmed unrelated GEMM/runtime state, so the two initial validation requests are not a controlled cold-start speed comparison. Module-import time and first-use records are archived separately. Cached-process startup was not benchmarked.

## Profile evidence

Matched profile runs generate exactly one token and assert zero decode calls. Ordinary normalization includes its nested tail; do not add overlapping scopes together.

| Prompt tokens | Ordinary normalization GPU time before → after | Ordinary kernels before → after |
| --- | --- | --- |
PROFILES

There are 161 tail calls/kernels per prefill, replacing enough eager operations to remove 644 ordinary-normalization kernels. Gated normalization stays at 576 kernels, FP8 GEMMs at 233, and GDN state kernel counts remain unchanged. All CUDA kernels are attributed. Profile times include profiling overhead and are separate from paired unprofiled latency.

## Correctness and dispatch

- 215 standalone bitwise cases cover all 65,280 finite BF16 patterns, signed zero, shapes/layouts, BF16/FP32 weights and input ownership, using the original eager inverse. Exhaustive activation-value coverage is not exhaustive model coverage.
- 86 primitive integration cases, two exact gradient cases and 20 tiny mixed-quantized models pass. Tiny tests preserve prefill/next-decode logits, all cache tensors and 16 IDs exactly. Missing Triton imports fall back; unrelated import errors and helper execution failures propagate.
- Real 27B at 256/1,024 tokens preserves entire prefill and next-decode logits/every cache tensor and all 128 generated IDs exactly. Validation observes 161 fast-path calls during prefill and zero during single-token decode.
- All 40 frozen quality output sequences and 446 token losses match the accepted native baseline exactly; perplexity remains 8.946730. The existing frozen reference screen passes unchanged.
- All 56 Qwen and 15 Llama gates pass with unchanged acceptance-source hashes/tolerances, including separate original 9B/27B numerical and conversation checks. Those legacy-runtime results are distinct from NVFP4 performance measurements.

Dispatch requires BF16 CUDA inference, rank at least three, at least 16 tokens, supported contiguous converted input/inverse/weight, matching devices/shapes and bounded indexing. CPU, autograd/no_grad, unsupported metadata and decode retain Torch behavior. Gated normalization remains unchanged. The private helper assumes a common device; the engine dispatch enforces it. Installed runtime compatibility was tested on both the experiment and original protected runtimes. No production, arbitrary hardware, arbitrary long-context or nonfinite-input equivalence claim is made.

## Recovery and provenance

Local responses030/031/032/034 were retained and rejected before placement for indexing/vectorization/name/import-handling problems. Local033 helper and035 integration are the only engine additions. An initial profile harness failed before model load because a local `importlib` binding shadowed the module; its original source and recovery note are retained. The protected 9B conversation test initially failed because Triton attempted to write its default compiler cache under the read-only container home. A writable `/tmp` cache then failed because that mount has `noexec`. The harness now uses the executable project bind mount for its cache; an actual tiny kernel preflight passed on protected Torch2.11 before unfinished model checks resumed. Both failed logs, original harnesses and recovery notes are retained. Only harness configuration was repaired. Numerical checks and engine source were not changed.

All four experiment containers are stopped with exit0/OOMfalse, no GPU compute process, service MainPID0, the same boot and GPU holds retained; exact memory/swap evidence is in `cleanup-check.json`. Large models ran serially. Compiler/GDN/cache candidates previously rejected remain excluded.

Model `unsloth/Qwen3.8-27B-NVFP4`, revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`. Experiment Torch2.13+cu130/Triton3.7.1 and protected Torch2.11 runtime; exact image IDs are in cleanup evidence. Main run: RUN.

Raw author responses/reviews, full before/after source, results, source hashes, frozen fixture hashes and harness are archived here. Weights, compiler cache binaries and large traces remain outside Git. Reproduce paired prefill using the installed original experiment directories, with all other workers stopped:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_prefill_norm_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_prefill_norm_prepared.py ab
```

Next measured option: gated-normalization pointwise fusion. It has a separate SiLU gate and different shapes, so it needs its own numerical investigation and local-model task; this checkpoint does not implement it.
'''.replace('ENGINE',m['engine_commit'][:7]).replace('ROWS','\n'.join(rows)).replace('PROFILES','\n'.join(profiles)).replace('RUN',str(run))
(out/'README.md').write_text(text);print(out/'README.md')
