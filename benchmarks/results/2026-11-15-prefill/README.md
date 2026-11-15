# Prefill optimization results

This package targets prompt processing, measured as warm prefill latency and prompt tokens/sec. Loading, tokenization and sustained output-token throughput are separate.

The accepted [FP8 input-preparation change](fp8-reduction/README.md), engine **2beb3fd**, reduces prefill latency by **2.7% at 256 prompt tokens** and **5.6% at 1,024** on the tested native 27B NVFP4 checkpoint. The one-line engine change was authored by local Qwen; exact numerical behavior and all protected checks are preserved.

| Prompt tokens | Before → after | Prompt tokens/sec before → after |
| --- | --- | --- |
| 256 | 275.65 → 268.26 ms | 928.7 → 954.3 |
| 1024 | 1153.92 → 1089.40 ms | 887.4 → 940.0 |

The first [GDN mask/intermediate-reuse experiment](gdn-reuse/README.md) passed exact checks but showed no prefill gain and was not adopted. It is excluded from the accepted engine. Earlier decode-cache experiments are separate and were not used to judge this prefill result.

The initial profile put GDN state computation, normalization and FP8 preparation at roughly 357 / 196 / 143 ms for 1,024 tokens. The accepted change lowers FP8 preparation to about 78 ms in the new prefill-only profile. Normalization and deeper GDN changes remain options for a later measured proposal; no implementation of those options is included here.
