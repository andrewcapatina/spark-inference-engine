# GDN prefill common-subexpression reuse: not adopted

Local Qwen task028 hoisted the causal chunk-mask predicate once per call and reused identical per-chunk weighted-K and exponential-decay intermediates. All other arithmetic, precision, chunking, solves and state propagation stayed unchanged. The supervisor placed the reviewed source verbatim and authored only harness/tests/docs.

Primary metric: warm prefill CUDA latency, excluding model load. One resident native 27B NVFP4 model used two alternating ABBA blocks (four samples per method per length).

| Prompt tokens | Baseline → candidate prefill | Prompt tokens/sec, baseline → candidate |
| --- | --- | --- |
| 256 | 279.74 → 279.39 ms | 915.1 → 916.3 |
| 1024 | 1151.51 → 1152.31 ms | 889.3 → 888.7 |

These differences show no measurable benefit. No output-token throughput was used as the optimization objective.

298 synthetic direct-function cases and 32 tiny dense/mixed-quantized integration cases preserve exact outputs, recurrent state, inputs and continuation. Partial chunks, noncontiguous inputs, FP32/BF16, batch1/2, initial states and alternate chunk size are covered. Per-function mask builds fall from one per chunk to one per call. Isolated synthetic real-head-layout timings show no clear speed gain either.

Real 27B at both prompt lengths preserves entire prefill logits/all caches, complete next-decode logits/all caches and all128 generated token IDs. Generation was used only for correctness. This rejected candidate did not proceed to the full quality/protected suite. The full source was retained before restoring all14 accepted engine files exactly to engine6621762/documentation4b4b549. Acceptance tolerances were unchanged. Native CUDA allocated/reserved bytes released to0/0 and container shutdown passed existing guards.

The larger prefill profile identified GDN state computation (~357ms), normalization (~196ms), and FP8 input preparation (~143ms) at1024 tokens. This particular GDN reuse change did not improve real prefill latency; the next isolated experiment targets FP8 input preparation. Raw result/source/provenance files and the exact harness are included. Weights and large traces remain outside Git.
