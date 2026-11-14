# Bounded 27B memory diagnostics — 2026-11-14

Both diagnostics completed on the actual downloaded 27B GGUF with native CUDA BF16 inference. The host remained in the same boot throughout. The tiny prerequisite probe verified a 16 MiB PyTorch allocator cap: a 1 MiB allocation succeeded and a 32 MiB request raised OutOfMemoryError.

The workload uses batch 1, exact raw-token prompt lengths, greedy generation with EOS disabled, one warmup, and separate CUDA-event observation. Profiling is disabled. Model loading is included in monitoring but excluded from generation timing. Both processes enforce a 56 GiB PyTorch allocator cap and check for at least 32 GiB host available memory between stages. A separate host process samples memory every 0.5 seconds, fsyncs its log, and stops this experiment container below 24 GiB available or above 1 GiB swap use. These are containment measures, not a guarantee against a fast host/driver failure.

| Diagnostic | Prompt tokens | Output tokens | Timed repetitions | Generation wall time | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 512 | 8 | 1 | 2.713 s | 50.99 GiB |
| Short | 1024 | 8 | 1 | 3.727 s | 51.73 GiB |
| Full length | 1024 | 128 | 3 | 31.619 s median | 51.73 GiB |

The full-length median throughput including prefill is 4.048 output tokens/s. All three timed repetitions have identical retained GPU allocation (50.106 GiB), reservation (52.477 GiB), and allocation peak (51.732 GiB). Process RSS changed by approximately 8 KiB across those repetitions. Repeated output hashes match; both short cases and the full-length case also match their separately timing-observed outputs. A single short sample does not establish repeatability by itself.

The external monitor recorded minimum host available memory of 51.49 GiB in the short run and 49.30 GiB in the full-length run, with zero swap use in both. CUDA and host memory views share physical memory and must not be added. CUDA load peak was 52.47 GiB. No watchdog or allocation-cap rejection occurred during inference. No matching NVRM/Xid/OOM/thermal/error/fail kernel messages were found in the diagnostic window.

These measurements show no growing leak over the tested repetitions and do not reproduce the shutdown. They do not prove its cause or establish that the cap prevented it. Compared with the interrupted experiment, this used a fresh host boot, no profiler, a memory cap, and different prior allocation/cache history. The original 27B profiler remains unvalidated. Do not combine these results with the original uncapped runs as a single unchanged protocol, or infer long-session safety from them.

Native inference hashes match the validated implementation. The result files record exact source hashes and a dirty working tree on parent commit `19afb7987d601b4a5035c82eb4104753f97684a8`, because the diagnostic harness was edited after that commit. The worker exited 0 before explicit container cleanup. Docker reports container exit137/OOMKilled=false because its init required SIGKILL after the stop timeout; this is not a failed inference or an OOM record. Both experiment containers are stopped.

Evidence: [summary](summary.json), [short results](27b-short.json), [full-length results](27b-full-length.json), [short phase log](27b-short-phases.jsonl), [full-length phase log](27b-full-length-phases.jsonl). Raw host monitoring remains in the experiment run directories named in the summary. See the [original interrupted run](../2026-11-14-interrupted/README.md) and [methodology](../../../BENCHMARKS.md).

Next: use the capped, non-profiler path for further work; retain the profiler path as a separate unresolved runtime investigation. Broader context lengths and long conversation stress tests remain unvalidated.
