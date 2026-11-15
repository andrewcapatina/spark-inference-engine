# NVFP4 profiles and prompt-processing options

The user selected **faster prompt processing (prefill)** as the priority. The first candidate is the existing standalone FlashInfer GDN prefill primitive, followed by fused normalization and FP8 input quantization. No engine implementation or optimization changed in this profiling package.

## Controlled measurements

Same Unsloth Qwen3.8-27B-NVFP4 revision f0b7c9e722f5565102fff8481c99e4d86ae099c7, batch1, raw prompts256/1024,16 outputs, greedy/EOS disabled. One warmup and three unprofiled completion controls precede each separate CPU/CUDA/memory trace. Reference uses explicit BF16KV, eager execution, no prefix caching/MTP/graphs, fixed512MiB KV budget. Both runtimes return identical16 output IDs at each context; all profile/control output checks pass.

| Prompt tokens | Native unprofiled completion | vLLM unprofiled completion | Native traced completion | vLLM traced completion |
| ---: | ---: | ---: | ---: | ---: |
|256|1.925s|1.537s|2.198s|1.556s|
|1024|2.902s|1.884s|3.104s|1.897s|

These16-output latencies include prefill, all generation and output-ID transfer/API completion. They are not a replacement for the prior128-output throughput baseline9.17/8.35tok/s. GPU event diagnostics and traces are separate, intrusive observations. Trace overhead is14.2%/7.0% for native and1.3%/0.7% for reference. Loading, tokenization and chat-template rendering are outside these timings. Clocks and cache history were not locked; each profile is one diagnostic observation, not an isolated one-change A/B trial.

## Where prefill time goes

At1024 tokens the native prefill CUDA-event duration is about1.22s. Exclusive raw-kernel categories show:

| Native prefill component |256-token GPU kernel time|1024-token GPU kernel time|Share at1024|
| --- | ---: | ---: | ---: |
|GDN state math|80.5ms|363.1ms|30.0%|
|Normalization|28.4ms|198.6ms|16.4%|
|FP8 activation quantization/padding/misc|28.0ms|171.2ms|14.1%|
|FP8 GEMMs|78.7ms|154.1ms|12.7%|
|FP4 GEMMs|49.3ms|100.3ms|8.3%|

Remaining GPU time includes convolution/gating, attention, activation/residual operations and other model work. Categories are exclusive; inclusive scopes must not be summed. Native full prefill kernel sums are299.3ms/1211.1ms. GDN state plus normalization/FP8 preparation account for about60.5% of the1024 prefill kernel time, so optimizing only matrix multiplies would miss the main prompt-processing opportunity.

For an aligned model-body comparison, excluding the output head:

| Prompt tokens | Native prefill body GPU time | Reference prefill body GPU time |
| ---: | ---: | ---: |
|256|292.6ms|150.2ms|
|1024|1196.8ms|491.4ms|

Reference context/generation annotations cover its model-forward body; head/sampling occurs outside those scopes. Native body values subtract its explicitly labelled lm_head. Reference unscoped postprocessing remains separate rather than being incorrectly assigned to decode or prefill. These are summed kernel times, not terminal-visible TTFT.

## Options in the selected order

| Candidate | Measured opportunity | Scope and safeguards |
| --- | --- | --- |
| **Standalone fused GDN prefill** | Largest single native prefill component:363ms,30% at1024. The reference trace uses FlashInfer's fused SM120 delta-rule kernel. | First benchmark the standalone API against current output and FP32 final recurrent state. Respect q/k L2 normalization, log-decay versus multiplicative decay, batch/layout and cache-state orientation. Preserve the original fallback and decode behavior. Local Qwen authors any engine integration. |
| **Fused normalization** |199ms,16% at1024. About3.2–3.4% of decode GPU time too. | Zero-centered Qwen RMS weights and gated RMSNorm have distinct semantics; retain FP32 intermediate/weight behavior and existing numerical gates. Test realistic shapes before choosing a backend. |
| **Fused FP8 activation quantization** |171ms,14% at1024; about3.2–3.4% of decode GPU time. The reference uses a dynamic per-token FP8 quantization kernel. | Preserve per-token scales, floor behavior, output layout, padding and numerical checks. This is narrower than changing projection storage/layout. |
| Reuse duplicate GDN QKV prefill projection | Source computes the same projection twice. | Smaller prefill-only cleanup; its isolated saved time was not measured. Preserve convolution-cache values and storage ownership. Do not treat all FP8 GEMM time as removable duplicate work. |
| Produce only final-position logits in generation prefill | The entire current head costs only6.6ms/14.4ms,2.2%/1.2% of prefill GPU time. | Even a free head cannot save more than that measured region; actual savings are smaller. Public full-logit prefill behavior must remain available. Not the first target. |
| Precompute invariant FP4 alpha | Under1% of decode kernel time and negligible prefill share. | Low-scope cleanup, not a primary prompt-processing improvement. |

A30% component is not a promised30% end-to-end speedup. Removing the entire GDN state component would bound the1024 prefill kernel-time improvement near1.43x, before integration overhead, remaining work or changed clocks. The fused reference kernel is evidence of a candidate, not an achieved native-engine improvement. Combined fusion options require their own measured comparison; no projected gains are added together as promises.

The installed reference source calls **flashinfer.gdn_prefill.chunk_gated_delta_rule** directly. Its adapter normalizes q/k, converts log-decay with exp, uses contiguous sequence-major tensors and FP32 initial state, and returns output/final state. Thus this candidate does not require replacing the engine with a serving runtime. Backend shape/device support and final-state layout must be verified independently before changing code.

## Decode and other findings

Decode is dominated by projection kernels: native FP8 GEMMs47–49%, FP4 GEMMs35–36%. GDN state math is only3.7–3.9% there. Faster prefill therefore will not multiply sustained output tok/s. Native/reference forward-body decode GPU times are102.8/86.1ms per call at256 and105.5/86.6ms at1024, with output heads excluded for alignment. Reference projection mappings fuse q/k/v, GDN qkv+z, and MLP gate/up; those are larger follow-on throughput options, not the selected first work.

Native traces contain136,192/158,560 kernels versus42,383/42,335 for reference. Fewer kernels do not imply a3x throughput opportunity. CPU launch costs overlap GPU work, profiler gaps are intrusive, and projection kernels still dominate decoding. No memory-bandwidth saturation or compute-utilization claim is established by these profiles.

Peak traced CUDA allocator usage including weights is21.65/22.78GiB native and20.94/21.06GiB reference. Different cache and weight-buffer layouts prevent assigning the whole difference to one feature. Allocation sums in summaries are cumulative allocator requests, not peak live memory or measured DRAM traffic. Record-shapes and Python stacks were disabled to limit retention/overhead; memory events were retained.

The CLI currently buffers the reply. Kernel-level first-token improvements will not make terminal output stream; streaming is a separate user-experience feature and is not a tokens/s optimization.

## Profiling correctness and recovery

A tiny CUDA preflight verified CUPTI/kernel support. It exposed duplicate automatic attribution through an Activity Buffer Request bookkeeping event; the original summary is retained and this profiler bookkeeping is excluded. A second tiny check matches raw kernel totals. Native Torch event summaries also omit part of FlashInfer driver-launched kernel time because those kernels lack Torch External IDs. Final component totals therefore use CUDA correlation IDs to join each raw GPU kernel to its CPU launch timestamp/thread and containing scopes. Every kernel in all four profiles has a matched launch, and category sums match raw kernel totals. Reference phases use explicit execute_context/generation labels; component families use kernel names and do not split fused work. Six CPU controls cover overlaps, custom ops, allocation counters, missing External IDs and phase labels. The original partial Torch operator summaries remain raw evidence and must not be used as complete GPU percentages.

The first reference attempt was rejected BEFORE model load because raw GPU-allocatable memory was just below the unchanged56GiB floor, despite ample MemAvailable. Advice to release only cache pages of the completed, inactive native trace files recovered headroom from59,732,901,888 to61,072,736,256 bytes. Trace data/size/mtime were retained, hashes recorded, and the floor was not lowered. This was a resource-preflight recovery, not an engine failure or numerical-test change.

Native model release passed weak-reference checks and tracked CUDA0/0. Both successful parent workers/container exits are0 with PID0/OOMfalse, same boot, no new swap, recovered host available memory and empty post-run GPU process lists. **The vLLM library force-killed its EngineCore after its5-second profiler-shutdown timeout.** Parent/container cleanup succeeded, but fully graceful reference child shutdown is not claimed. The cleanup excerpt and failed preflight are retained. All experiment containers were stopped afterward; the legacy service remains inactive and global holds remain in place.

## Evidence and next bounded work

Raw1GB-plus native traces and reference gzip traces stay local under /home/cap/spark-inference-overnight-20261113/feedback/nvfp4-reference; local-trace-manifest.json records hashes. Compact raw results, corrected trace summaries, control CSV, source snapshots, plans and status are committed here. Installed reference source hashes and the pinned runtime identify the inspected API/implementations.

Next: a standalone GDN-prefill timing/output/state comparison at256 and1024 tokens, then a focused local-Qwen task if that candidate passes. Preserve56Qwen/15Llama original numerical gates, independent operator/real-shape checks and the fixed40-task/446-token quality screen. Those passed checkpoints were not rerun for this harness-only profiling work; engine hashes remain unchanged. No implementation patch has been selected or placed.
