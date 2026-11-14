# Interrupted GB10 benchmark — 2026-11-14

The machine rebooted during the 27B benchmark. The 9B matrix/profile completed; the 27B 32- and 256-token cases completed, while its 1024-token case and profile did not. This is an incomplete baseline, not a completed performance report.

## Preserved measurements

Both models used native CUDA BF16 inference at benchmark source commit `19afb7987d601b4a5035c82eb4104753f97684a8`. Workload: batch 1, fixed 128 output tokens, EOS disabled, one warmup and three timed repetitions. OS/NAS cache state was uncontrolled.

| Model | Prompt tokens | Median output tokens/s, including prefill | Median generation wall time (s) |
| --- | ---: | ---: | ---: |
| 9B | 32 | 10.81 | 11.838 |
| 9B | 256 | 10.69 | 11.975 |
| 9B | 1024 | 10.15 | 12.611 |
| 27B (partial) | 32 | 4.16 | 30.777 |
| 27B (partial) | 256 | 4.11 | 31.121 |

All saved cases have matching repeated output hashes and matching timing-observed output. The full 9B profile attributes approximately 90.4% of summed ATen operator device self-time to `aten::mm`; that is not a measured memory-bandwidth or utilization result. No 27B operator profile was completed.

## Memory evidence before the interruption

Across all six saved 27B repetitions, end-of-run CUDA allocation stayed near 50.11 GiB and CUDA reservation at 52.51 GiB. Process RSS was about 2.87 GiB and process swap was zero. The final saved snapshot reported 14.35 GiB CUDA-global free memory and 59.65 GiB system available memory. These shared-memory views must not be added.

The recorded repetitions do not show progressive retained-allocation growth. They cannot rule out a transient allocation spike or a GPU/driver failure in the unrecorded 1024-token phase. CUDA phase peaks rose with prompt length, as expected; the repeated measurements at each saved length had identical allocated/reserved peaks.

## Crash evidence and limits

- A new host boot was observed starting 2026-11-14 11:33:30 PST. The last durable 27B result file is timestamped 11:23:05 PST and contains the completed 256-token case. The next planned case was 1024 tokens; its exact failing operation is not recorded.
- Docker reports the agent stopped, exit 255 and `OOMKilled=false`. That flag is not proof against a host/GPU memory failure during a hard reset.
- No OOM-kill entry tied to this interruption was found in the persisted previous-boot kernel journal. An earlier 06:42 OOM event is outside this benchmark window.
- Driver 580.178.04 logged `refcntRequestReference_IMPL` messages with status `0x00000056` at 11:01:51, 11:08:06 and 11:16:58. The tiny native profiler probe ran at 11:08:05 and the 9B profiling worker finished at 11:17:07. This timing makes profiling a path to isolate, not a proven cause. The 27B profiler had not begun.
- NVIDIA defines `0x00000056` as `NV_ERR_NOT_SUPPORTED`, not the out-of-memory status. See the [versioned status definitions](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/580.178.04/src/common/sdk/nvidia/inc/nvstatuscodes.h).
- No orderly shutdown marker was found in the filtered journal. Persisted kernel crash storage could not be inspected without additional root access. The user reports the machine shut down and had to be powered on again, matching previous suspected OOM incidents. This strengthens memory exhaustion as a working hypothesis; it does not establish the failing allocation or exclude another cause.

## Recovery boundary

The heavy benchmark has not been restarted. Preserve this run and record durable per-stage/per-sample memory telemetry before another attempt. A controlled follow-up should isolate the 27B 1024-token case in a fresh process without profiling, with a conservative process allocator limit and short output length. Any such result must be identified as a separate diagnostic protocol. No driver, swap or unrelated service settings were changed.

See [methodology](../../../BENCHMARKS.md), [complete 9B data](9b.json), and [partial 27B snapshot](27b-partial.json). Raw traces/logs remain in the ignored benchmark directory and experiment run directory.
