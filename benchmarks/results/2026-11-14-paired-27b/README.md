# Same-loaded-model old/current comparison

The current local-Qwen final-decode change was faster in all four pairs at both output lengths. The earlier separate-process long-output slowdown was not reproduced. This supports retaining the existing patch for the tested workload; it is a small improvement on long replies. No engine source was edited.

One actual27B GGUF model was loaded once in CUDA BF16. The old method was extracted verbatim from `18894a8^`, checked by source hash, and temporarily bound to that same instance by the benchmark. The current method stayed on the class and was restored after every old-version invocation. No second model or reference backend was loaded. Existing engine files retain their pre-run hashes.

256 identical raw prompt tokens, EOS disabled, batch1, one warmup per version at each length, four timed pairs per output length. Pair order AB/BA/AB/BA counters systematic ordering/time drift; it does not lock clocks or establish all-workload performance. CUDA synchronized wall times exclude load, prompt tokenization, output hashing and telemetry. No profiler/event observer inside timings.

| Output tokens | Before median | Current median | Lower latency | Current output tok/s including prefill |
| --- | ---: | ---: | ---: | ---: |
| 8 | 2.3379 s | 2.1121 s | 9.66% | 3.788 |
| 128 | 29.8078 s | 29.5734 s | 0.79% | 4.328 |

The saving was approximately0.22–0.24s per response, consistent with eliminating one full-model decode. The percentage benefit shrinks on longer responses because the saving is roughly fixed; it did not reverse into a penalty at128 outputs. Every version/repetition produced exactly the same full output digest at its token length. This is not a new full-logit reference validation; unchanged numerical gates were passed previously.

All16 timed generations plus4 warmups completed. Allocator cap56GiB, host floor32GiB, new-swap guard256MiB retained. No swap growth or reboot; worker0; fresh CUDA tracked allocation/reservation0/0; NVML empty; both containers stopped and controller inactive. In-worker release left about8.13MiB allocated and22MiB reserved, followed by full process exit. Host available memory recovered within4GiB. Thermal/host raw telemetry is retained in the run directory. No trial-window OOM/Xid/thermal fault recorded. This second successful post-update workload does not diagnose the previous power-offs or establish long-session stability.

An initial attempt failed before model loading because the harness path was wrong; it is retained separately and excluded from timings. The path was corrected to the existing `/workspace/acceptance` mount, with no engine change.

[Summary and every paired delta](summary.json), [raw measurements](paired-results.json), [source manifest](paired-manifest.json), [trial/cleanup status](trial-status.json), [initial harness failure](initial-harness-failure.json). Raw baseline text and fsynced generation/host/thermal logs remain under the experiment run and `benchmark_runs` directories, outside Git. Scripts here are snapshots of the controller/acceptance harness.

Next: discuss options that target steady decode throughput. No further engine implementation or workload selected.
