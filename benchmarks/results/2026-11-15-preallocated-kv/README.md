# Preallocated attention-cache experiment: retained, not enabled

The local model implemented per-request capacity-backed K/V buffers, with a shared append cursor and copy-on-branch ownership. The supervisor wrote the harness, reviewed candidates and assembled/placed exact retained local-model source. No supervisor-written engine implementation was used.

The experiment passes its targeted numerical checks but does not establish a useful end-to-end speedup at the measured lengths. Its extra ownership machinery is not enabled. The complete candidate was retained before exact restoration to accepted engine6621762 (documentation baselineaf5bc77), including removal of the two experimental helper modules. Accepted QKV reuse and duplicate-concatenation reuse remain; convolution-history compaction stays quarantined.

## Paired measurements

One resident native27B NVFP4 model alternates accepted two-concatenation decode/preallocated/preallocated/accepted. Two warm uninstrumented128-output samples per method and length exclude loading. Differences below0.5% in this small sample do not establish a robust speed benefit.

| Prompt tokens | Total128-output tokens/s, accepted → experimental | Change | Fewer cumulative allocator-request bytes per separate16-output profile |
| --- | --- | --- | --- |
| 256 | 9.091 → 9.126 | +0.39% | 215.5 MiB |
| 1024 | 8.310 → 8.344 | +0.41% | 897.6 MiB |

The256-prompt decode-only diagnostic changes9.169→9.212tok/s; the1024-prompt diagnostic changes8.911→8.897tok/s. Prefill is unchanged by this decode-only work. Saved allocation requests represent cumulative transient allocation, not peak memory or measured bandwidth. Profile timings include instrumentation overhead and are separate from the128-output throughput measurements.

256 tokens: profile kernels 135328→135360, decode concatenations 2175→1695, peak allocation 21.649→21.649 GiB.

1024 tokens: profile kernels 157696→157728, decode concatenations 2175→1695, peak allocation 22.778→22.778 GiB.

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

Final cleanup/restoration evidence is in `cleanup-check.json`, source hashes in `manifest.json`, and measured outputs/profile summaries in the result JSON files. Weights and large raw traces remain outside Git. Runtime is the installed pinned Torch2.13+cu130 image and the same `unsloth/Qwen3.8-27B-NVFP4` revision used in the accepted attention-cache report. Main run: `/home/cap/spark-inference-overnight-20261113/runs/preallocated-attention-kv-20261115T083924Z`.

The `candidate-source/` files are archived text, not active engine modules. Reproducing this experiment requires deliberately restoring that isolated source in the experiment environment; the normal engine retains the accepted implementation. This report is not a claim that preallocation is production-ready or validated across all models/contexts.
