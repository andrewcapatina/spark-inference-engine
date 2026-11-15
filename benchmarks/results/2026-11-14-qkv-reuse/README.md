# Reuse GDN QKV projection

Local Qwen's engine checkpoint **6dd8d9f** reuses the first raw QKV projection in `QwenGatedDeltaNet.prefill` for convolution history. It removes the second identical projection. The change is two replaced lines and one deleted line; numerical formulas, cache layout, forward and decode are unchanged. Only local Qwen authored the engine source; the supervisor placed the retained candidate verbatim and wrote diagnostics/tests/controllers/docs.

## Measured result

One resident native27B NVFP4 model alternated original/reuse/reuse/original, with two timing samples per method and length. Warm inference excludes loading. These paired results are workload-specific, not statistical confidence intervals.

| Prompt tokens | Original prefill CUDA median | Reused QKV | Less prefill time | Total128-output tokens/s, original → reuse |
| --- | --- | --- | --- | --- |
|256|297.54ms|275.49ms|7.4%|9.033 →9.044|
|1024|1226.97ms|1173.27ms|4.4%|8.213 →8.233|

This improves prompt processing. Sustained generation changed very little. Peak tensor allocation was essentially unchanged; convolution-history backing storage is a separate followup.

## Correctness and profiling


-28tiny dense/mixed-quantized cases (batch1/2; lengths1,3,4,5,64,256,1024) preserve element-exact output, cache and continuation. QKV forward calls fall2→1 per GDN prefill.

-Real27B256/1024 prompts preserve element-exact entire logits and every cache tensor. All128 generated token IDs match across the paired repetitions. GDN QKV calls fall96→48.

-The frozen40-case screen remains40/40, with identical generated token IDs on all40cases versus the prior native engine. All446 token losses are identical; weighted perplexity remains8.946730, ratio1.0 versus prior native and0.986059 versus frozen vLLM reference. This is a small screening set, not broad quality certification.

-All56unique Qwen gates and15Llama cases pass, including real9B and real27B numerical/conversation checks. Original acceptance-source hashes and numerical tolerances are unchanged. Independent quantized full-logit parity with reference implementations remains partial, as documented before this change.

-Separate CPU/CUDA/memory profiles use256/1024 prompts and16 outputs, with warmup and three unprofiled controls. Profile outputs match controls, prior profiles and paired128-output prefixes. These16-output controls are not the128-output throughput measurement above.

-Prefill FP8 calls fall281→233. Each16-output trace loses exactly384CUDA kernels:136192→135808 and158560→158176. No CUDA kernels are unmatched. The remaining QKV projection scope accounts for18.91/54.20ms of GPU time, consistent with the duplicate cost removed; inclusive scopes overlap.

The paired benchmark and every completed native stage release CUDA allocations/reservations to0/0. Final checkpoint cleanup: all four experiment containers stopped with exit0/OOMfalse, no GPU compute processes, legacy service MainPID0, same host boot and no new swap. Only one large model is loaded at a time.

The first profile attempt stopped before loading at55.03GiB CUDA free, below the unchanged56GiB preflight floor. That failure is retained. A fresh attempt passed the original floor without extra weight-cache hints; the new bounded, file-specific cache-advice fallback was unused. No memory or numerical threshold was weakened.

## Artifacts and reproduction

`same-model-ab.json`, `tiny-exact-checks.json`, `quality-native.json`, `quality-comparisons.json`, `profile-native.json`, `paired-results-audit.json`, `protected-correctness.json` and `provenance.json` retain the evidence. `task-020-response.json` and candidate/review files establish local-Qwen authorship. `harness/` contains the focused test, old reference method and controllers. The launch snapshot and subsequent explicit CPU cross-audit are identified in `harness-version-note.json`. Large raw traces and weights remain outside Git.

Runtime: installed pinned image `sha256:fa7334538384d67c72284edc081c78b1149a36d53aba1fb3c5f7d1d534154ec6`, Torch2.13.0+cu130; model `unsloth/Qwen3.8-27B-NVFP4`, revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`. Protected original-model tests use their preserved runtime. Main run: `/home/cap/spark-inference-overnight-20261113/runs/gdn-qkv-reuse-20261115T054800Z`.

From engine checkpoint6dd8d9f with the installed experiment runtime and no active CLI/controller:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_qkv_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_qkv_prepared.py ab
```

The rejected [GDN compiler experiments](../2026-11-14-gdn-prefill/README.md) are separate. Next authorized step: compact convolution history, then attention-cache allocation/copy work, each independently checked.
