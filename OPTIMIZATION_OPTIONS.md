# Options to review before further work

No new optimization is selected. GPU experiments are held after two host shutdowns. The current local-Qwen change passes protected correctness, but [performance validation is incomplete](benchmarks/results/2026-11-14-final-decode/README.md). Short token-limited replies improved in saved measurements; longer results were mixed/slower. The pending paired diagnostic did not run.

A later [small 9B clean-shutdown trial](benchmarks/results/2026-11-14-clean-shutdown/README.md) passed with normal worker/container exits and recovered host available memory. OS cache grew by about 17 GiB and CUDA-global free stayed lower; review that physical-memory headroom before another large-model run. This has not resolved the 27B shutdowns.

## Reliability comes first

| Option | What it would establish | Limitation / decision needed |
| --- | --- | --- |
| Review existing crash evidence and accessible OS/NVIDIA logs | Preserve both incidents, compare their stages, look for a driver, memory, power or thermal record | Current logs do not establish a cause; some persistent crash storage requires additional access |
| Inspect physical shutdown conditions and available hardware diagnostics | Distinguish power-off, freeze, reboot and possible thermal/power issues | Requires the operator's observations; telemetry before the second crash did not capture the actual failure |
| Design one narrowly bounded reproduction after reviewing evidence | Isolate context length, repeated model loads, native runtime or a known reference backend | A new GPU run can still shut down the host; allocator caps and a polling watchdog have already proved insufficient |
| Keep or revert the small local-Qwen patch while investigating | Choose the source checkpoint used for later diagnostics | The original engine also suffered a shutdown; reverting is not a demonstrated stability fix |

The default action is to preserve evidence and keep heavy work stopped. No driver, swap, power, unrelated service or engine changes are part of this options review.

## Performance choices once a diagnostic plan is agreed

The choices improve different user-visible things. These are alternatives, not an automatic task queue. Expected benefits below are hypotheses unless explicitly measured.

| Choice | Evidence we have | Likely benefit | Effort / risk |
| --- | --- | --- | --- |
| Reduce 9B loader initialization and staging | Fresh-process loads took roughly 108–161 seconds in saved runs; native construction allocates CPU FP32 parameters before moving to BF16 and overwriting them; one recorded load RSS high-water mark was 36.24 GiB | Faster startup and potentially lower host memory peaks; no expected improvement to steady decode | Small to medium scope; must preserve every loaded weight, missing-weight errors, dtype/device and full-logit parity. Separate initialization, transfer and I/O timing first |
| Reuse the delta-network QKV projection during prefill | `QwenGatedDeltaNet.prefill` calls the same projection twice; the source is explicit, but its isolated time has not been measured | Lower prompt-processing time; potentially fewer temporaries | Small to medium scope; preserve the exact convolution/recurrent cache values and ownership |
| Avoid retaining large backing storage through convolution-cache views | The last-kernel-width convolution state can be a view into a full-prompt projection | Potentially lower request memory near prefill; actual storage retention and peak effect need measurement | Small to medium scope; verify storage size/lifetime, aliasing and all cached-decoding checks before changing it |
| Produce only the needed vocabulary logits during generation prefill | Public prefill currently produces logits for every prompt position, while greedy generation selects from only the final position | Lower first-token time and temporary memory, especially for long prompts | Medium scope; existing public full-logit prefill behavior is tested and must remain available unchanged |
| Reuse compatible token caches between chat turns | Each turn currently formats and prefills the full history again | Lower latency for long multi-turn conversations; does not directly improve steady per-token decoding | Medium to high scope; requires prefix-token identity, reset/cancel/edit semantics, cache ownership and memory budgeting; current text API is intentionally stateless |
| Investigate quantized GPU execution | 27B Q8_0 storage expands to BF16; matrix multiplication accounted for about 90.4% of operator GPU self-time in the saved 9B profile | Potentially improve model capacity and decode throughput | High scope; kernels/layout/numerical reference work required. Neither bandwidth saturation nor a promised speedup has been established |
| Reduce attention KV copying / repeated KV heads | Source performs duplicate concatenations; all `aten::cat` together contributed only about 0.24% of GPU operator self-time in the saved 9B 256-prompt/8-output profile | Probably small on that workload; could matter at longer contexts | Small to medium scope, with cache-layout/aliasing risks; measure at the intended context before prioritizing |

For startup delay or host memory, the loader is the strongest measured lead. For long-conversation latency, prefix/cache reuse addresses a different problem. For steady decode throughput, quantized execution is a larger research project; the operator profile alone does not select a kernel implementation.

Any chosen engine implementation remains local Qwen's job. Before implementing another change, agree on the target metric and a narrow task, capture its baseline, retain the existing numerical gates, and choose containment appropriate to the repeated host failures. The next action should be selected by the user after this review.
