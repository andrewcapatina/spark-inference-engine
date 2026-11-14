# Short 27B trial with verified clean shutdown

The user explicitly requested this 27B-only trial after the successful [9B cleanup trial](../2026-11-14-clean-shutdown/README.md). It completed without a host reboot. One native 27B GGUF model ran in a fresh container process boundary; the coding/model server stayed stopped, with no other GPU compute process present before loading. The default crash hold remained active outside this one-shot trial.

## Workload and measurements

Batch 1 CUDA BF16, 256 raw prompt tokens, 8 generated tokens, EOS disabled, one warmup, three timed repetitions and one separate CUDA-event diagnostic. No profiler. The allocator cap was 56 GiB; external guards required 32 GiB available host memory and at most 256 MiB new swap growth. Native inference source is unchanged from local-Qwen implementation `18894a8`.

| Metric | Observed |
| --- | ---: |
| Fresh-process model/tokenizer load | 141.408 s |
| Median generation wall time | 2.098 s |
| Generation range | 2.098–2.103 s |
| Output tokens/s, including prefill | 3.812 |
| First token, CUDA-event diagnostic | 508.37 ms |
| Prefill tokens/s, CUDA diagnostic | 503.67 |
| Useful decode tokens/s, CUDA diagnostic | 4.392 |
| Peak allocated CUDA memory, including load | 52.47 GiB |
| Peak allocated CUDA memory during measured generation | 50.62 GiB |
| Minimum host available memory during monitoring | 49.55 GiB |
| New swap usage | 0 bytes |

Timed output hashes match each other, the event diagnostic, and the earlier optimized 256-prompt/eight-output case. Eight selected tokens required seven useful decode calls. Loading/tokenization and the subsequent release checks are excluded from generation timing. The load is not guaranteed cold-disk: filesystem/NAS caches were uncontrolled. This is a short post-change result, not a before/after optimization comparison.

## Verified teardown

Explicit release inside the worker left 8.125 MiB tracked allocation and 20 MiB reservation. After worker exit 0, a fresh CUDA probe reported zero tracked allocation/reservation, and NVML compute-process inventories were empty. The agent stopped in 0.164 seconds with exit 0, pid0, OOMKilled=false. The coding/model container remained stopped; boot ID was unchanged. Native source hashes match the pre-trial snapshot.

Host available memory recovered from 115.40 to 114.63 GiB, within the 4 GiB recovery bound. Host cached memory grew from 20.12 to 47.93 GiB; CUDA-global free changed from 93.20 to 64.67 GiB. Those are shared-memory views, not capacities to add. A clean model/process/container shutdown does not flush OS file cache. No global cache flush, swap change, driver change, package installation or unrelated service action was performed.

## Limits and evidence

Both models have now passed small 256-prompt/eight-output cleanup trials in separate processes. The prior shutdown cause is still unproven. This trial did not exercise the 1024-prompt/128-output workload that previously failed, and does not complete the earlier interrupted performance comparison or validate long-session stability. No new optimization or further workload was automatically selected.

[Summary](summary.json), [measurements](27b.json), [trial status](trial-status.json), [plan](trial-plan.json), [last process inventory](process-inventory.txt). Complete durable host/phase logs remain in `/home/cap/spark-inference-overnight-20261113/runs/clean-shutdown-27b-20261114T231647Z`. The benchmark source checkpoint was `28d2baf` and the worker recorded repository checkpoint `60ebd27`; exact source hashes are in the measurements. See [methodology](../../../BENCHMARKS.md), [options](../../../OPTIMIZATION_OPTIONS.md) and [previous shutdown evidence](../2026-11-14-final-decode/README.md).
