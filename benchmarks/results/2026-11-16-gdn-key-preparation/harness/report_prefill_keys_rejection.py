"""Archive small/inconsistent real gains, original failure and reference safeguards."""
from pathlib import Path
import json,hashlib,shutil
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference';run=Path((R/'feedback/prefill-keys-main-run.txt').read_text().strip());m=json.loads((run/'manifest.json').read_text());assert m['status']=='not_adopted_small_real_prefill_gain_baseline_restored';assert json.loads((run/'cleanup-check.json').read_text())['status']=='clean';assert m['engine_before']=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
out=P/'benchmarks/results/2026-11-16-gdn-key-preparation';out.mkdir(parents=True,exist_ok=True);h=out/'harness';h.mkdir(exist_ok=True);raw=P/'benchmark_runs'/run.name
for stage in sorted((R/'runs').glob('prefill-keys-*')):
 for name in ['trial-status.json','diagnostic-results.json']:
  if (stage/name).exists():shutil.copy2(stage/name,out/(stage.name+'-'+name))
 for name in ['results.json','standalone-results.json']:
  if (B/stage.name/name).exists():shutil.copy2(B/stage.name/name,out/(stage.name+'-'+name))
 if (stage/'27b-worker.log').exists():shutil.copy2(stage/'27b-worker.log',out/(stage.name+'-worker.log'))
for name in ['manifest.json','cleanup-check.json','historical-generation-comparison.json','expanded-ab-review.json','canonical-fixture-check.json','handoff-context-check.json','compile-cache-review-note.txt']:shutil.copy2(run/name,out/name)
for p in raw.iterdir():
 if p.is_file():shutil.copy2(p,out/p.name)
for directory in ['failed-real-generation-engine','final-engine-for-validation','real-tested-constant-stride-engine','full-final-runtime-stride-engine-before-rollback']:
 src=raw/directory;target=out/'variants'/directory;target.mkdir(parents=True,exist_ok=True)
 for name in ['qwen_ops.py','qwen_key_prepare.py']:shutil.copy2(src/name,target/(name+'.txt'))
 (target/'all-engine-SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in src.glob('*.py')},indent=2)+'\n')
files=['prepare_prefill_keys_trial.py','launch_prefill_keys_prepared.py','run_prefill_keys_small.py','run_prefill_keys_integration.py','run_prefill_keys_layout.py','run_prefill_keys_variants.py','run_prefill_keys_pack.py','run_prefill_keys_ab.py','run_prefill_keys_diagnostic.py','finish_prefill_keys_all.py','finish_prefill_keys_validation.py','compare_prefill_keys_results.py','rollback_prefill_keys.py','check_prefill_keys_rollback_cleanup.py','report_prefill_keys_rejection.py','verify_keys_handoff_context.py','nvfp4_update_runtime.py','shutdown_process_probe.py','acceptance/check_gdn_key_prepare.py','acceptance/check_gdn_keys_integration.py','acceptance/check_keys_final_components.py','acceptance/check_key_compile_reuse.py','acceptance/tune_gdn_key_prepare.py','acceptance/profile_gdn_keys_small.py','acceptance/benchmark_prefill_keys_ab.py','acceptance/diagnose_keys_generation.py','acceptance/qwen_ops_before_keys_processing.py.txt','acceptance/prefill-canonical-generation-v1.json','acceptance/request_nvfp4_candidate.py']
for name in files:shutil.copy2(R/name,h/Path(name).name)
(h/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in h.iterdir() if p.name!='SHA256.json'},indent=2)+'\n')
review=json.loads((run/'expanded-ab-review.json').read_text());rows=[]
for c in review['rows']:
 a,b=c['summary']['original'],c['summary']['candidate'];rows.append('| {} | {:.3f} → {:.3f} | {:.3f}% | {}/{} |'.format(c['prompt_tokens'],a['median_prefill_cuda_ms'],b['median_prefill_cuda_ms'],c['less_prefill_time_pct'],c['positive_blocks'],c['blocks']))
text='''# Fused GDN key preparation: not adopted

Codex implemented this prototype under the user's explicit direct-authoring approval. It was retained and removed from the active engine: **all 16 accepted source files match engine 3139a9b byte for byte**. No engine speedup from this experiment is accepted. Prior QKV, FP8 preparation, normalization and value-cast improvements remain.

## Implementation and scope

One Triton kernel computes K*beta once, then its ordered FP32 product with caller-supplied eager exp. It writes separate weighted/decayed-key buffers in the original compact time-major or head-major pointwise dimension order. It does not concatenate RHS tensors. The original two triangular solves, GEMMs, eager exponentials, carried-state scan and output dtype are retained; floating-point fusion is disabled. The dispatcher preserves CPU/FP32/autograd/unsupported-layout and broadcast-prefix fallbacks, with missing-Triton-only import fallback and propagated execution errors.

The initial fully checked helper was slower than Torch: metadata checks and Python launch overhead outweighed the device saving. An internal per-prefill plan validates stable metadata once and supplies only fixed-size slices of those inputs. Output allocations remain request-owned; it captures metadata, not model tensors or mutable scratch shared between requests.

## Measured performance

A graph-replay diagnostic, used only to separate launch overhead, measured about 9.82 µs for three Torch products versus 4.13 µs for the fused kernel. These are warm small-tensor measurements, not engine speedups. Eager checked helper cost was 18.9 µs; the planned helper 12.3 µs versus Torch 13.2 µs. Different launch blocks128–2048 did not materially reduce the eager launch cost. No CUDA graph was added to engine execution.

On a synthetic native 48-head block's actual convolution-produced layout, component medians improved 1.632→1.606ms at 256 tokens and 7.664→7.265ms at 1,024. Copy time stayed essentially unchanged. This was only an early gate; larger model measurements gave much smaller gains.

One resident native 27B NVFP4 model then alternated baseline/candidate/candidate/baseline. The first short-prompt trial regressed 0.265% and stopped at a generation discrepancy (below). A subsequent fixed-reference trial, eight warm samples/backend/length, improved 0.623%/0.835%. The expanded final real trial used **16 warm samples/backend/length**, eight ABBA blocks:

| Prompt tokens | Baseline → candidate prefill ms | Less time | Positive timing blocks |
| --- | --- | --- | --- |
ROWS

One short-prompt block regressed 0.076%. The longer-prompt blocks were positive, but the overall reduction was only about 4.65 ms in a roughly 1-second prefill. The expanded controller's performance filter stopped before further acceptance gates. The decision is that these small gains do not justify the additional kernel/plan machinery. Loading, tokenization and compilation are excluded; no sustained output-token throughput gain is claimed. Repeated samples are not a confidence interval or a claim for other hardware/contexts. All raw repetitions and earlier trials remain archived.

## Generation discrepancy and recovery evidence

The first real trial preserved entire 256-token prefill logits/every cache tensor and next-decode logits/cache, but its 128-token comparison diverged at zero-based index 5. The candidate was quarantined immediately and the failure retained. Its first baseline sequence was an outlier: the candidate sequence matches previous accepted benchmarks.

A fresh diagnostic reproduced the A/B priming sequence, then ran six alternating original/candidate generations. All six matched the older accepted sequence. Original-v-original and original-v-candidate teacher-forced comparisons preserved full logits/every cache tensor through 127 decode steps. Saved earlier FP8/norm/gated/value/paired-solve benchmark sequences also match that canonical sequence rather than the first failed-run baseline.

**The cause remains unresolved.** This identifies an outlying baseline result, not the source of its variation. The baseline executed after candidate warmups, so these observations do not rule out a candidate/allocation/runtime interaction. A matching repeat does not erase the original failure or establish production reliability. Earlier state-tail/convolution failures remain separate rejected experiments; they are not exonerated by this result.

Both later real A/B trials added stronger checks: actual prompt IDs must match the accepted profile, and every baseline/candidate generation must match the frozen accepted 128-ID sequence. Both lengths preserve full prefill/next-decode/cache and two generations per backend exactly. Existing numerical tests/tolerances were not loosened. The durable harness-only fixture prefill-canonical-generation-v1.json is derived from the previously accepted3139a9b benchmark/profile, with source hashes. It never supplies answers to engine implementation. A baseline or candidate that drifts must fail; the first generation is no longer an unquestioned oracle.

## Growing-context compilation review

The real-tested plan used constexpr strides; a full-prompt-dependent stride caused different kernel binaries at 256/1,024 tokens. That can add compilation to a growing conversation despite unchanged chunk width. Before accepting anything, a final address-only prototype changed strides to runtime values while preserving the known within-layout geometry.

That final version reuses **one binary per layout** across full prompt lengths128/256/512/1024. It passed 101 standalone bitwise/layout/ownership cases, 4 multi-chunk plan cases, 304 direct GDN cases, 44 dense/quantized blocks, 20 tiny models, 2 gradient cases, 11 CPU/provided-state-dtype cases and 3 broadcast-prefix fallbacks. Edge cases include signed zeros, subnormals, finite extremes, zero-stride beta/decay and metadata bounds; the ordered-product test contains 139,017 reassociation counterexamples. Layout checks cover all nontrivial dimensions; singleton-axis strides are immaterial to storage order.

**No further real-checkpoint trial was run for the runtime-stride variant**, because the design's whole-model gain was already too small. Real timings above apply to the retained constant-stride version. Both versions and full source hashes are separately archived; do not treat the final cache prototype as real-checkpoint validated.

## Retention, checks and cleanup

Main run RUN. Full failed, real-tested and final 17-file candidate trees remain under ignored benchmark_runs; the two changed source files, all-engine hashes, intermediate helpers and patches are copied here. Large traces, compiler binaries and weights stay out of Git. The metadata-only timing harness error from the first standalone launch is retained and distinguished from the generation discrepancy. Runtime status now reads implementation authorship from the manifest, rather than incorrectly labeling direct work as local-Qwen-authored.

No new frozen 40-quality or protected 56 Qwen/15 Llama suite was run after deciding against adoption. The accepted baseline retains its prior validation by exact source restoration. All four experiment containers stopped exit0/OOMfalse, GPU idle, service MainPID 0, same boot, zero new swap over 15 bounded stages; STOP/GPU/LARGE holds retained. See cleanup-check.json for telemetry, including failed and successful stages.

LOCAL_MODEL_HANDOFF.md records the findings and is automatically included/fingerprinted in future raw local-author requests; the offline request smoke check passes. The next priority is investigating generation repeatability and reducing repeated checkpoint loading in the validation harness. Cheaper launch dispatch or broader fusion would be separate prototypes, not an accepted improvement here.

Model: unsloth/Qwen3.8-27B-NVFP4, revisionf0b7c9e722f5565102fff8481c99e4d86ae099c7; existing pinned Torch 2.13/CUDA 13 runtime. No packages, models or arbitrary code were downloaded; no unrelated services or reference serving engines were used. Reproduction requires restoring the chosen retained version in an isolated checkout/container mount. The active project intentionally contains the accepted baseline, so it cannot reproduce candidate timings simply by running the archived worker against current code.
'''.replace('ROWS','\n'.join(rows)).replace('RUN',str(run))
(out/'README.md').write_text(text);print(out/'README.md')
