# Performance measurements and first optimization

Latest [preliminary accuracy screen](benchmarks/results/2026-11-14-nvfp4-accuracy-screen/README.md): native and matched vLLM both40/40 tasks, no truncation; native perplexity8.947 versus9.073 on446 prose tokens. All predeclared criteria passed. This does not establish general quality or resolve independent numerical parity. No engine change or new optimization selected.

Earlier [layer/reference investigation](benchmarks/results/2026-11-14-nvfp4-layer-reference/README.md): official HF graph with shared native projection operators matches the real273-token native score vector exactly; independent QDQ/FP32 arithmetic still diverges. Real projection probes pass unchanged bounds. Representative quantized-model task accuracy remains unvalidated; no new optimization selected.

Follow-up [identical-input diagnostic](benchmarks/results/2026-11-14-nvfp4-teacher-forcing/README.md):125/128 forced top choices agree; at the first free divergence both fresh-prefill engines agree, while vLLM cached generation differs. This narrows the investigation but does not establish full quantized-model numerical correctness.

Earlier performance checkpoint: [native NVFP4 benchmark and accuracy scope](benchmarks/results/2026-11-14-native-nvfp4-benchmark/README.md):9.17/8.35 output tok/s at256/1024 prompt tokens with128 outputs,21.04GiB resident model tensors, protected56Qwen/15Llama regressions passed. Later-token256-prompt divergence from vLLM remains unresolved; broad quality validation is pending. Earlier results below are historical.

For the current cross-run conclusion and comparison limits, see the [performance summary](benchmarks/results/SUMMARY.md).

## What is established

The native engine runs both downloaded models on one GB10 using CUDA BF16. The complete original 9B benchmark produced approximately 10–11 output tokens/s. The completed original 27B cases produced approximately 4.1 output tokens/s. A separate capped 27B diagnostic completed the missing 1024-token prompt at 4.05 output tokens/s. These are development measurements, not production capacity guarantees.

The original baseline remains incomplete: its 27B 1024-token case and profiler did not finish before a host shutdown. Recovery diagnostics are reported separately. Subsequently, local Qwen implemented the final-decode change. Protected correctness passed, but its performance comparison was interrupted by another host reboot; see the [optimization report](benchmarks/results/2026-11-14-final-decode/README.md).

## Original benchmark, before the shutdown

Batch 1; exact repeated raw-token prompts; 128 new tokens with EOS disabled; one warmup and three uninstrumented timed repetitions per prompt length. Wall times are medians, excluding model load/tokenization/template rendering/input transfer/output text decoding. First-token, prefill and decode figures come from a separate single CUDA-event diagnostic, not those three wall-time repetitions. Peak CUDA allocation is the maximum of measured repetitions and includes weights.

| Model | Prompt tokens | Generation wall time (s) | Output tokens/s | First token (ms) | Prefill tokens/s | Decode tokens/s | Peak CUDA allocated (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9B Safetensors | 32 | 11.838 | 10.81 | 113.0 | 283 | 10.89 | 16.77 |
| 9B Safetensors | 256 | 11.975 | 10.69 | 175.7 | 1457 | 10.85 | 16.96 |
| 9B Safetensors | 1024 | 12.611 | 10.15 | 621.4 | 1648 | 10.67 | 17.62 |
| 27B GGUF | 32 | 30.777 | 4.16 | 339.0 | 94 | 4.21 | 50.31 |
| 27B GGUF | 256 | 31.121 | 4.11 | 532.3 | 481 | 4.18 | 50.62 |

First-token latency marks selection on the CUDA stream; it is not terminal-visible response latency. The CLI currently prints only after the reply finishes. Prefill tokens/s measures native prefill alone. Decode tokens/s uses the 127 intervals between 128 selections, excluding prefill and the unused final decode. Output tokens/s divides all 128 output tokens by complete generation wall time, including prefill and that wasted final decode. Peak allocated memory is distinct from allocator reservation, process RSS and host physical-memory usage.

## Recovery diagnostic, after a fresh host boot

The same native inference source was tested with a 56 GiB PyTorch allocator cap, 32 GiB available-memory checks between stages, an external host watchdog, durable telemetry and no profiler. The following full-length case used one warmup and three timed repetitions plus separate CUDA-event observation. This changed protocol must remain separate from the original table.

| Model | Prompt tokens | Generation wall time (s) | Output tokens/s | First token (ms) | Prefill tokens/s | Decode tokens/s | Peak CUDA allocated (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27B capped | 1024 | 31.619 | 4.05 | 1886.6 | 543 | 4.30 | 51.73 |

All three full-length repetitions had identical retained CUDA allocation (50.106 GiB), reservation (52.477 GiB) and peak allocation (51.732 GiB), with matching output hashes. Host available memory stayed above 49.30 GiB and swap use remained zero. Earlier short diagnostics also completed 512- and 1024-token prompts with eight new tokens. The tiny cap probe verified that an oversized allocation raises an allocator error.

The shutdown was not reproduced. The user had to power the machine on again and recognized behavior from previous OOM incidents; memory exhaustion remains a hypothesis. There is no saved OOM record tied to the original shutdown. Stable retained allocation in these tests argues against a progressive leak over these repetitions, but does not rule out earlier transient pressure, driver behavior or longer-session problems. The cap did not reject any allocation in these inference runs, so it has not been shown to be the reason they succeeded. Both experiment containers were stopped afterward.

## Loading and memory

These are fresh-process model/tokenizer load measurements after explicit CUDA initialization. Filesystem/NAS cache state was uncontrolled: **none is guaranteed to be a cold-disk measurement**. Load time varies substantially between runs and is not a sound before/after optimization comparison without further control.

| Run | Load time (s) | CUDA allocation peak during load (GiB) | Process-lifetime RSS high-water mark at load end (GiB) |
| --- | ---: | ---: | ---: |
| 9B original | 109.54 | 18.57 | 36.24 |
| 27B original | 44.30 | 52.47 | 16.61 |
| 27B short diagnostic | 142.71 | 52.47 | 16.61 |
| 27B full-length diagnostic | 41.98 | 52.47 | 16.61 |

CPU and GPU share physical memory on this machine. Do not sum independent high-water marks or add CUDA-global free memory to host available memory. The 27B Q8_0 file expands to BF16 for execution; this engine currently does not execute quantized matrix multiplication. See the [architecture walkthrough](ARCHITECTURE.md#how-weights-become-a-ready-model).

## What profiling supports

The saved 9B PyTorch profile used a warmed 256-token prompt and eight new tokens. It is a separate instrumented workload, excluded from baseline wall timings. Of 873.81 ms summed ATen operator device self-time, `aten::mm` accounts for 790.09 ms (90.42%). Multiplication and copies each account for approximately 2.4%. CUDA tensors use ATen/library kernels; no custom kernels were added.

This identifies matrix multiplication as the dominant operator family in that request. It does not establish memory-bandwidth saturation, GPU utilization, which projection is most expensive, or what a particular replacement kernel would save. CPU/device annotated native ranges appear twice in this profiler version; do not add those duplicate rows. Operator allocation fields describe profiled events, not host capacity or request peak memory. No completed 27B operator profile exists. Further PyTorch Profiler runs remain deferred while the shutdown cause is unresolved.

## Selected first optimization: remove the unused final decode

`QwenCausalLM.generate` selects and appends an output token, then calls `decode` even when that token exhausts `max_new_tokens`. Its resulting logits and updated request cache are discarded. Source inspection confirms this control flow; CUDA-event diagnostics independently measured the last decode call.

| Evidence | Discarded final decode | Share of observed generation CUDA interval |
| --- | ---: | ---: |
| 9B original, 256 prompt / 128 output | 92.6 ms | 0.77% |
| 27B original, 256 prompt / 128 output | 238.8 ms | 0.77% |
| 27B capped, 1024 prompt / 128 output | 232.5 ms | 0.74% |
| 27B short, 1024 prompt / 8 output | 231.0 ms | 6.20% |

The first change should stop after selecting the final requested token, without computing another next-token distribution. Only the local Qwen model may implement it, confined to `QwenCausalLM.generate` in `spark_engine/qwen_model.py`. It must preserve the public signature, exact output IDs, prompt prefix, EOS behavior, output dtype/device and established zero-token behavior. It must not change loaders, layer arithmetic, public prefill/decode APIs, caches returned by those APIs, precision or numerical tolerances.

This is a small, directly measured reduction in unnecessary whole-model work. At 128 output tokens, the observed final decode consumes roughly 0.7–0.8% of the generation interval. At eight output tokens it represents a larger share. These are potential savings estimated from the existing event run, not measured optimization results. Removing this work should not improve time to first token or the steady-state decode rate; it should reduce complete-reply latency. Major throughput improvements need a subsequent, separately measured task.

The [local-model task and acceptance plan](OPTIMIZATION_TASK.md) defines implementation scope, correctness gates, measurement repair and before/after comparison. That task has now been executed through correctness validation, but performance acceptance is incomplete after a second host reboot. In particular, the current event observer uses decode entry as its token-selection marker; that assumption must be corrected before measuring the optimized loop, or the last selected token would disappear from its accounting.

## Evidence and reproducibility

- [Benchmark harness and metric definitions](BENCHMARKS.md); [architecture](ARCHITECTURE.md).
- [Original interrupted report](benchmarks/results/2026-11-14-interrupted/README.md), [9B JSON](benchmarks/results/2026-11-14-interrupted/9b.json), [27B partial JSON](benchmarks/results/2026-11-14-interrupted/27b-partial.json).
- [Recovery report](benchmarks/results/2026-11-14-memory-diagnostics/README.md), [full-length JSON](benchmarks/results/2026-11-14-memory-diagnostics/27b-full-length.json), [diagnostic summary](benchmarks/results/2026-11-14-memory-diagnostics/summary.json).

Original workers recorded clean benchmark commit `19afb7987d601b4a5035c82eb4104753f97684a8`. Recovery workers recorded the same parent with a dirty working tree; exact source hashes are embedded in their JSON. Native inference hashes match across protocols and the previously validated engine. Harness/recovery evidence was checkpointed locally at `8d7da0e`. Golden fixtures and the full acceptance harness remain external to the repository. Original long-context, profiler and production-safety limitations remain in effect.

## Current hold

Further GPU experiments are halted after the second host shutdown. The caps and external watchdog did not prevent it. Review the [partial optimization report](benchmarks/results/2026-11-14-final-decode/README.md) and [options](OPTIMIZATION_OPTIONS.md) before choosing further work. No blanket speedup or host-stability fix is claimed.

## Later small trials with clean shutdown

Separate one-model trials used 256 raw prompt tokens, 8 output tokens, 1 warmup and 3 timed repetitions, with explicit model release after timing and verified normal process/container exits. The 9B median was 0.809 s; the 27B median was 2.098 s. These are short post-change cleanup measurements, not the missing longer-workload baseline or a new before/after comparison. Both had matching outputs, no swap growth, and same-boot completion. See [9B evidence](benchmarks/results/2026-11-14-clean-shutdown/README.md) and [27B evidence](benchmarks/results/2026-11-14-clean-shutdown-27b/README.md). Earlier shutdowns remain unexplained and no further optimization has been selected.

## Latest larger-workload result

The isolated 27B 256-prompt/128-output stage passed three timed repetitions (median 29.799 s) and verified clean teardown. The fresh 1024-prompt/128-output stage completed warmup and one 31.800 s timed reply, then the host powered off spontaneously just after the next repetition began. The user confirms original NVIDIA adapter/cable. One model, no profiler, clean stage boundaries and recovered headroom did not prevent the failure. Larger work is held; no complete baseline or blanket speedup is claimed. See [third-crash evidence](benchmarks/results/2026-11-14-larger-27b-third-crash/README.md).

## Post-update heavy-workload repeat

The actual 27B 1024-prompt/128-output trial passed three timed repetitions plus warmup, separate CUDA-event observation and verified clean teardown after the user applied the Dashboard update. Median 31.417 s / 4.074 output tokens/s including prefill; no swap growth or reboot. Native implementation remained unchanged. Firmware and kernel both changed; one successful trial does not prove the earlier power-off fixed or establish an optimization speedup. See [complete evidence and limits](benchmarks/results/2026-11-14-post-update-27b-heavy/README.md).

## Same-loaded-model comparison completed

One27B model,256prompt,4 alternating pairs per output length:8outputs improved2.338→2.112s(9.66% lower latency);128outputs29.808→29.573s(0.79% lower). Every pair favored current source and all output hashes matched; current128-output throughput4.328tok/s including prefill. No reboot/swap growth; clean teardown verified. This resolves the earlier slowdown concern for this tested case, not all contexts/models. [Full paired evidence](benchmarks/results/2026-11-14-paired-27b/README.md).
