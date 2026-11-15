# Prompt-processing optimization options

The selected target is **prefill latency**, following the user's request to start at the beginning of inference. The [current matched NVFP4 profiles](benchmarks/results/2026-11-14-nvfp4-profile/README.md) separate unprofiled controls from CUDA/CPU/memory traces. No new engine implementation has been selected or placed.

At1024 prompt tokens, native prefill is about1.22s by CUDA events. Its exclusive GPU kernel time is30% GDN state math,16% normalization and14% FP8 activation quantization/preparation. The reference model-forward body takes491ms versus1197ms native, excluding output heads for alignment.

| Priority | Candidate | Evidence | Main risk / next check |
| --- | --- | --- | --- |
|1|Standalone fused GDN prefill|Largest individual prefill component,363ms/30%; installed reference uses FlashInfer's standalone prefill kernel|Compare output and FP32 final state, including normalization, decay convention and state orientation, before local-Qwen integration; preserve decode/fallback paths|
|2|Fused normalization|199ms/16% of1024 prefill; also3% of decode GPU time|Preserve zero-centered RMS and gated RMS semantics and intermediate precision|
|3|Fused FP8 input quantization|171ms/14% of1024 prefill; also3% of decode|Preserve per-token scales, floor, dtype and layout; benchmark actual shapes|
|Later|Reuse duplicate GDN QKV projection and inspect conv-cache backing storage|Duplicate is explicit in source; isolated saved time/retention not measured here|Small prefill/memory improvements, with exact cache contents/ownership preserved|
|Low priority|Last-position-only head in generation|Entire current head is only1–2% of prefill GPU time|Public full-logit prefill behavior must remain; potential savings smaller than whole-head share|
|Low priority|Precompute constant FP4 alpha|Under1% of decode kernel time, negligible prefill share|Useful cleanup, insufficient as the main performance package|

The first bounded step should benchmark **flashinfer.gdn_prefill.chunk_gated_delta_rule** against current state math, then give the local Qwen model a narrow implementation task if it passes. Do not turn the30% component share into a promised request-speed improvement; even eliminating it entirely bounds GPU prefill improvement near1.43x. No mathematical, quality or regression tolerances should be weakened to accept a faster kernel.

For a later sustained-generation target, FP8/FP4 GEMMs dominate decode at47–49%/35–36%. Projection fusion/backend experiments are separate from the current prefill priority. CUDA graphs require cache/address/layout work and are not selected by kernel counts alone. Loading and tokenization were excluded from this profile, and the buffered CLI needs a separate streaming feature for terminal-visible early tokens.

The preliminary fixed quality screen passed40/40 in native and vLLM, with perplexity8.947/9.073 on446 prose tokens. This is a small screen; full independent numerical parity and broad quality remain unvalidated. Original56Qwen/15Llama protected gates remain required for any engine change. All implementation remains the local Qwen model's job.

[Previous options snapshot](benchmarks/results/2026-11-14-nvfp4-profile/previous-options.md) preserves earlier BF16/crash-era alternatives. Current profile evidence supersedes their performance priorities; earlier host-poweroff causes remain unproven.
