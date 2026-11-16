# Fused GDN key preparation: not adopted

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
| 256 | 259.124 → 258.014 | 0.429% | 7/8 |
| 1024 | 1005.728 → 1001.083 | 0.462% | 8/8 |

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

Main run /home/cap/spark-inference-overnight-20261113/runs/prefill-gdn-keys-20261116T133845Z. Full failed, real-tested and final 17-file candidate trees remain under ignored benchmark_runs; the two changed source files, all-engine hashes, intermediate helpers and patches are copied here. Large traces, compiler binaries and weights stay out of Git. The metadata-only timing harness error from the first standalone launch is retained and distinguished from the generation discrepancy. Runtime status now reads implementation authorship from the manifest, rather than incorrectly labeling direct work as local-Qwen-authored.

No new frozen 40-quality or protected 56 Qwen/15 Llama suite was run after deciding against adoption. The accepted baseline retains its prior validation by exact source restoration. All four experiment containers stopped exit0/OOMfalse, GPU idle, service MainPID 0, same boot, zero new swap over 15 bounded stages; STOP/GPU/LARGE holds retained. See cleanup-check.json for telemetry, including failed and successful stages.

LOCAL_MODEL_HANDOFF.md records the findings and is automatically included/fingerprinted in future raw local-author requests; the offline request smoke check passes. The next priority is investigating generation repeatability and reducing repeated checkpoint loading in the validation harness. Cheaper launch dispatch or broader fusion would be separate prototypes, not an accepted improvement here.

Model: unsloth/Qwen3.8-27B-NVFP4, revisionf0b7c9e722f5565102fff8481c99e4d86ae099c7; existing pinned Torch 2.13/CUDA 13 runtime. No packages, models or arbitrary code were downloaded; no unrelated services or reference serving engines were used. Reproduction requires restoring the chosen retained version in an isolated checkout/container mount. The active project intentionally contains the accepted baseline, so it cannot reproduce candidate timings simply by running the archived worker against current code.
