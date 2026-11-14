# Final-decode optimization: correctness passed, performance incomplete

Local Qwen authored the entire replacement `QwenCausalLM.generate` method. Its exact response was retained and mechanically placed after signature/scope review. Native implementation commit: `18894a8`; supporting harness/provenance commit: `a820bc3`. Codex authored only the harness, tests, diagnostics and documentation.

## Correctness

All 56 unique existing Qwen gates and all 15 Llama acceptance cases passed. This includes actual 9B full-logit parity at unchanged rtol=0.05/atol=0.05, exact greedy outputs and actual 9B/27B conversation checks. The new checkpoint-free repository test covers six token-budget/EOS cases; focused CUDA fixtures verify both call counts and observer accounting. Actual timed outputs below also match across source versions. This is correctness evidence, not a host-stability guarantee.

An initial aggregate regression process was deliberately stopped when cumulative host swap crossed 1 GiB, after 32 observed passes and with about 45 GiB available. Its partial logs remain preserved. All protected gates were then rerun using a fresh process per real checkpoint, retaining head-layout/import interaction checks. The recovered watchdog uses a stricter 32 GiB available-memory floor and a 1 GiB *new swap growth* budget per command. GPU allocator limits remained unchanged. No numerical test or tolerance was weakened.

## Partial matched measurements

The measurement code is the same on both sides (schema 2, source hashes recorded), with batch1 CUDA BF16, exact raw-token prompts, EOS disabled, one warmup, three timed repetitions and separate CUDA-event observation. Model-specific allocator caps are 24 GiB for 9B and 56 GiB for 27B. Loading and text preprocessing are excluded. GPU clocks, external system load and filesystem cache state were uncontrolled. Before/after host-watchdog policies differ as described above; neither interrupted a completed timing case.

Positive reduction means lower latency; negative reduction means the optimized version was slower in these separate-process runs. All seven saved cases have matching before/after output hashes and show N-1 decode calls for N returned tokens after the change.

| Model | Prompt | Output | Before median (ms) | After median (ms) | Latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 9b | 256 | 8 | 894.16 | 826.34 | +7.58% |
| 9b | 1024 | 8 | 1324.59 | 1259.04 | +4.95% |
| 9b | 256 | 128 | 11648.26 | 11804.84 | -1.34% |
| 9b | 1024 | 128 | 12170.48 | 12462.49 | -2.40% |
| 27b | 256 | 8 | 2365.61 | 2179.02 | +7.89% |
| 27b | 1024 | 8 | 3772.29 | 3578.93 | +5.13% |
| 27b | 256 | 128 | 30154.30 | 30652.21 | -1.65% |

The eight-token cases improve by approximately 5–8%. The saved 128-token cases are slower, so a blanket throughput/speedup claim is unsupported. An alternating same-loaded-model 9B diagnostic was prepared to investigate run-to-run conditions, but **never executed** because the host failed first. Early-EOS replies already skipped the unused final decode; this optimization targets token-limit termination.

## Second host shutdown

The host rebooted during the post-change 27B 1024-prompt/128-output timed repetition0. The1024 warmup completed, but no timed 1024 sample completed. The prior 256 case was durably saved. All original benchmark/phase logs are preserved.

The last phase snapshot was `before_sample` at 21:15:01.629645UTC. The external monitor continued until 21:15:03.359636UTC, about 1.7 seconds later. Its last host available memory was 59.26 GiB. The phase snapshot reported 14.25 GiB CUDA-global free, 50.106 GiB allocated and 52.504 GiB reserved, with zero process swap. The completed 1024 warmup allocation peak was 51.732 GiB, below the 56 GiB PyTorch cap. The external watchdog recorded no new swap growth. Shared CPU/GPU memory views must not be added.

The next boot started 21:20:18UTC, boot ID `c7a27d76-0d0b-44bf-a860-709525b7582e`. Docker reports agent exit255/OOMKilled=false. No tied OOM/GPU-fault or orderly shutdown record was found in the persisted previous-boot kernel journal. These facts do not rule out an abrupt driver, memory, hardware, power or thermal failure. A fast unrecorded spike remains possible; the measurements do not establish the root cause. This crash occurred without PyTorch Profiler, so profiling is not necessary to reproduce the shutdown.

The allocator cap and polling watchdog did not prevent a host failure. Their limitations must not be presented as a whole-system safety guarantee. Both experiment containers and the original controller are stopped. A `GPU_WORKLOAD_HOLD` and controller `STOP` marker prevent automatic experiment restarts. **No heavy retry or paired diagnostic was started after this reboot.**

The local code still contains the correctness-tested Qwen patch. Performance acceptance remains incomplete, and host stability remains unresolved. The pre-change engine also experienced a shutdown; neither keeping nor reverting this small optimization is a proven stability fix.

## Evidence and next decision

[Partial comparison](partial-comparison.json), [correctness audit](correctness.json), [authorship/placement provenance](provenance.json), [crash snapshots](second-host-crash.json), [phase log](second-crash-phases.jsonl), [host-monitor tail](second-crash-host-tail.jsonl). Full host/kernel logs remain under the experiment run directory named in the provenance. Raw worker output is unchanged; its saved `benchmarking` status is wrapped explicitly as interrupted, not promoted to complete.

Review reliability evidence and [the options](../../../OPTIMIZATION_OPTIONS.md) with the user before further GPU work or another optimization. The original interrupted baseline remains incomplete and no 27B operator profile is claimed.
