# FP8 input preparation: accepted prefill improvement

Engine checkpoint **2beb3fd**, authored by Local Qwen, changes one expression in `FP8Linear.forward`. It computes the absolute-value row maximum in BF16, then widens the small row-max result to FP32 before the original division by 448 and clamp to 1e-6. The old method widened the entire activation matrix first.

For finite BF16 inputs, maximum selects an exactly representable input magnitude; widening that selected value preserves the original FP32 maximum. Scale dtype, division/clamp precision, quantized activation bytes, matrix-multiply settings, padding and output shape remain unchanged. Tests verify those properties directly. No new quantization approximation, compiler, kernel backend or cache mechanism is introduced. The supervisor wrote the harness/tests/docs and placed the local-model source verbatim.

## Primary result: prompt processing

One resident native 27B NVFP4 model alternated baseline/candidate/candidate/baseline twice: four warm samples per method and length. These measurements exclude loading and tokenization and use direct `model.prefill` CUDA events; wall-time medians agree. Generated output throughput is not the objective or acceptance metric. The improvement is additional to the already accepted QKV reuse.

| Prompt tokens | Prefill latency, baseline → accepted | Less prefill time | Prompt tokens/sec, baseline → accepted |
| --- | --- | --- | --- |
| 256 | 275.65 → 268.26 ms | 2.7% | 928.7 → 954.3 |
| 1024 | 1153.92 → 1089.40 ms | 5.6% | 887.4 → 940.0 |

Both ABBA blocks improve at each length (2.3–3.1% at 256; 5.6–5.8% at 1024). This is a small repeated workload measurement, not a statistical confidence interval or a claim about all contexts/models. A separate first-token CUDA diagnostic agrees; it excludes load/tokenization and records zero decode calls.

## Prefill-only profile

The new profile generates exactly one token and asserts zero decode calls. Only prefill scopes are compared with the earlier accepted-engine profile; whole traces are not compared because the earlier trace also contained decoding. Component GPU time is separate from unprofiled paired latency.

| Prompt tokens | FP8 input-preparation GPU time, before → after | Cumulative FP8 preparation allocation requests, before → after |
| --- | --- | --- |
| 256 | 23.75 → 15.62 ms | 4.25 → 2.31 GiB |
| 1024 | 143.19 → 77.71 ms | 16.83 → 9.06 GiB |

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

Model: `unsloth/Qwen3.8-27B-NVFP4`, revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`. Runtime: installed pinned Torch 2.13+cu130 experiment container, with its exact image ID recorded in cleanup evidence. Protected original-model tests use their preserved runtime. Main run: `/home/cap/spark-inference-overnight-20261113/runs/prefill-fp8-reduce-20261115T142642Z`.

The result JSON, fixed quality comparisons, protected fixture hashes, raw local-model response/review, before/after source and exact harness/checksums are included. Weights and large raw traces remain outside Git.

With this engine source, the installed references/runtime and no active experiment CLI/controller:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_prefill_fp8_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_prefill_fp8_prepared.py ab
```

The harness switches retained baseline versus accepted FP8 projection on the same loaded weights. Its primary metrics are prefill milliseconds and prompt tokens/sec.
