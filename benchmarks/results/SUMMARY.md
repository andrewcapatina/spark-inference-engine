# Performance summary across runs

The latest [same-loaded-model comparison](2026-11-14-paired-27b/README.md) found 9.66% lower latency for8 outputs and0.79% for128 outputs on27B with256 prompt tokens; all four pairs at each length favored the current loop. The earlier separate-process slowdown did not reproduce.

The only native optimization so far removes the unused decode after the final requested output token. Local Qwen authored it (`18894a8`); all 56 Qwen gates and 15 Llama cases passed with unchanged numerical tolerances. It removes real work, but the saved timings do not establish a general throughput improvement.

## Closest before/after comparison

Same measurement harness and workload on each side; separate processes, uncontrolled clocks/system load/cache history. Values are medians of three uninstrumented repetitions, excluding load; positive reduction means faster.

| Model | Prompt / output tokens | Before (s) | After (s) | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| 9B | 256 / 8 | 0.894 | 0.826 | +7.58% |
| 9B | 1024 / 8 | 1.325 | 1.259 | +4.95% |
| 9B | 256 / 128 | 11.648 | 11.805 | -1.34% |
| 9B | 1024 / 128 | 12.170 | 12.462 | -2.40% |
| 27B | 256 / 8 | 2.366 | 2.179 | +7.89% |
| 27B | 1024 / 8 | 3.772 | 3.579 | +5.13% |
| 27B | 256 / 128 | 30.154 | 30.652 | -1.65% |

Eight-output cases were 4.95–7.89% faster. The completed 128-output cases were 1.34–2.40% slower. Because only one whole-model decode was eliminated, the expected saving was roughly 0.7–0.8% for 128 outputs, small enough to be obscured by cross-run variation. The change does not accelerate individual decode steps or prefill. Early-EOS replies already skipped the final decode. A same-loaded-model alternating comparison was prepared but never executed; the original matrix remains seven of eight complete.

## Reliability runs and their comparison limits

| Run | Workload | Result |
| --- | --- | --- |
| Original baseline | 9B, 128 outputs at 32/256/1024 prompts | 11.838 / 11.975 / 12.611 s; complete |
| Original baseline | 27B, 128 outputs at 32/256 prompts | 30.777 / 31.121 s; 1024-prompt stage interrupted by first power-off |
| Capped recovery, before optimization | 27B 1024/128 | 31.619 s median, completed |
| Optimization comparison | Seven before/after cases above | Short replies improved; long replies slower; second power-off interrupted eighth case |
| Clean-shutdown checks, optimized | 9B and 27B 256/8 | 0.809 / 2.098 s medians; clean teardown |
| Larger isolated trial, optimized | 27B 256/128, then 1024/128 | 29.799 s median for 256; 31.800 s single sample for 1024, then third power-off |
| Post-update stack, optimized | 27B 1024/128 | 31.417 s median; three repetitions, events and clean teardown passed |

These recovery runs changed process isolation, cache history, cleanup and finally firmware/kernel. They are separate observations, not additional matched optimization pairs. The latest heavy result is about 0.64% lower latency than the earlier pre-optimization capped recovery, or 2.18% lower than the dedicated pre-change 32.116 s case. Neither difference isolates the engine change or firmware benefit. No sustained stability conclusion follows from one passing heavy trial; the same heavy workload had also passed once before the update.

## What we gained and what remains

We have a working real-model baseline (9B about 10–11 output tokens/s; 27B about 4–4.3 including prefill), a correctness-preserving reduction in wasted work with a measured short-reply benefit, durable diagnostics, one-model isolation and verified clean teardown. We have not demonstrated a broad long-reply speedup, reduced model-memory footprint, a loading speedup, or the cause/fix of the three power-offs. Loading varied roughly 42–144 s for 27B with uncontrolled disk/NAS cache state and is excluded from generation timings.

The bounded alternating comparison has now completed; see the linked latest result. It supports a small long-reply benefit for the tested27B workload, without demonstrating universal throughput improvement. No subsequent optimization has been selected.

Sources: [matched comparison](2026-11-14-final-decode/README.md), [original baseline](2026-11-14-interrupted/README.md), [capped recovery](2026-11-14-memory-diagnostics/README.md), [third power-off](2026-11-14-larger-27b-third-crash/README.md), [post-update result](2026-11-14-post-update-27b-heavy/README.md).
