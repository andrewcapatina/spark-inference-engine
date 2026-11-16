# GDN value-cast deferral

Local Qwen task 054 authored the only engine change, checkpoint **3139a9b**. Baseline engine ea70e47 includes the accepted QKV, FP8 preparation and normalization optimizations. Rejected state-tail and paired-solve candidates remain excluded. The supervisor wrote the harness, diagnostics, prompts and report; engine source is placed verbatim from the retained local response.

## Warm prefill result

One resident native 27B NVFP4 model alternates baseline/candidate/candidate/baseline four times, eight warm samples per method per length. Direct CUDA-event model.prefill excludes loading, tokenization and compilation. All four timing blocks improve at both lengths. These small repeated measurements are not a confidence interval or a claim for arbitrary workloads. Percentages from separate experiments must not be added.

| Prompt tokens | Baseline → accepted prefill | Less processing time | Prompt tokens/sec baseline → accepted |
| --- | --- | --- | --- |
| 256 | 259.17 → 255.27 ms | 1.50% | 987.8 → 1002.8 |
| 1024 | 1022.76 → 1012.95 ms | 0.96% | 1001.2 → 1010.9 |

Median peak CUDA allocated memory: 256 tokens: 21.644 → 21.644 GiB; 1024 tokens: 22.775 → 22.786 GiB. The 1,024-token peak increases by 11.1 MiB; no peak-memory saving is claimed. This is per-process tensor allocation, not total host/unified memory. Generation is a correctness gate here; no sustained output-token speedup is claimed.

## Change and precision

The original GDN chunk function eagerly converts all values to FP32 before their first use. For BF16 queries and BF16 CUDA values in inference mode, the new function retains the transposed BF16 view. The existing value-times-FP32-beta product promotes to FP32 directly. Output storage and both initial-state allocation/conversion explicitly remain FP32. All query/key normalization, two original triangular solves, chunk/padding logic, exponentials, GEMMs and state propagation expressions remain unchanged. CPU, other dtypes, no_grad and autograd retain the original widening path. No new kernel, module, dependency or cache mutation is introduced.

## Validation

- 298 direct cases, 32 dense/mixed-quantized blocks, 20 tiny whole models, 2 exact gradient cases and 11 extra CPU/provided-state-dtype cases pass. Inputs and supplied states remain unmodified. Output dtype, FP32 state, decode continuation and caches match exactly. Observational instrumentation sees zero value.float calls in eligible inference, one in the original/fallback path. These are component/tiny tests, distinct from real-checkpoint validation.
- Real27B at256/1,024 tokens preserves entire prefill logits and every cache tensor, next-decode logits/cache, and two repeated 128-ID generations per backend and length exactly.
- Frozen 40 quality sequences and all 446 corpus token losses match the accepted native baseline exactly. Perplexity 8.946730084578427. The frozen vLLM-reference quality screen passes; this does not expand independent full-logit quantized reference coverage.
- All 56 Qwen and 15 Llama protected gates pass, including separate original 9B/27B numeric and conversation checks. Fixture-source hashes and numerical tolerances are unchanged.

## Matched profile

Each profile generates exactly one token and asserts zero decode calls. Prompt IDs and normalization input shapes match the preceding accepted profile. Forty-eight GDN kernels disappear, one eager value-copy kernel per GDN layer. Other normalization and quantized GEMM kernel counts remain unchanged.

| Prompt tokens | GDN GPU time baseline → accepted | GDN kernels baseline → accepted |
| --- | --- | --- |
| 256 | 79.94 → 76.26 ms | 8496 → 8448 |
| 1024 | 357.82 → 340.73 ms | 30960 → 30912 |

Profiled GPU times include profiler overhead and are separate from unprofiled A/B latency. Inclusive scope times overlap; do not sum parent and child times. Raw large traces and model weights remain outside Git.

## Containment and next priority

All models ran serially, with per-worker memory/time/swap guards, source hashes and same-boot checks. Final four experiment containers are stopped exit0/OOMfalse, service MainPID 0, no GPU jobs; STOP/GPU/LARGE holds retained. cleanup-check.json contains actual memory telemetry. No duplicate controller or unrelated service changes.

Model unsloth/Qwen3.8-27B-NVFP4 revf0b7c9e722f5565102fff8481c99e4d86ae099c7; experimental Torch 2.13+cu130 and original protected Torch 2.11 runtime. Main run /home/cap/spark-inference-overnight-20261113/runs/prefill-gdn-value-20261116T063116Z.

The next-options.md review prioritizes batching independent GDN chunk-local preparation before the sequential state scan. GDN state processing remains the largest measured scope (~36% before this change). That is an unimplemented option with numerical/workspace risks, not a promised improvement. Stop extending tiny pointwise experiments without a measured reason.

Reproduce using the existing installed containers, no other workers:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_prefill_value_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_prefill_value_prepared.py ab
```
