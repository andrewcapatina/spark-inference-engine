# Small 9B benchmark with verified clean shutdown

The explicitly authorized bounded trial passed. It loaded one actual 9B Safetensors model in a fresh signal-aware container, ran a 256-token raw prompt with 8 generated tokens, and verified teardown. The coding/model server stayed stopped; no 27B inference or profiler ran. The general crash hold stayed active throughout. This is a small post-change cleanup diagnostic, not a completed two-model baseline or proof that the previous host failure is fixed.

## What changed in the harness

The original agent used `sleep infinity` as PID1 and required SIGKILL after Docker's stop timeout. It was preserved as a stopped backup. The replacement uses the exact same image, environment, mounts, user, GPU requests, security and limits, with an init process and signal-aware keepalive. Its empty CPU-only lifecycle stopped in 0.112 seconds with exit 0.

The benchmark's optional `--clean-release` deletes the model/text adapter after timing, runs garbage collection, synchronizes CUDA, empties the allocator cache, and records release memory. It changes no native implementation or numerical assertions. The driver verifies model-worker exit, GPU process inventories, a fresh CUDA probe, container exit and host-memory recovery. One-shot execution holds the controller/benchmark locks and does not resume the old queue.

The first attempt was stopped by an erroneous process-count guard: it counted `timeout ... python3 ...` and its single Python child as two workers. Its inventory/logs were retained. The parser was fixed to inspect the actual executable; four checks cover the observed inventory, a genuine second worker, timeout-only and renamed-thread cases. The same small trial was retried in a separate directory. This was a harness failure, not an engine regression or shutdown.

## Measured results

- Load: 163.037 seconds in a fresh process; filesystem/NAS cache state uncontrolled.
- Generation: 806.39–810.72ms across three timed repetitions; median 808.82 ms, 9.89 output tokens/s including prefill.
- One warmup and a separate CUDA-event diagnostic; eight selected tokens required seven useful decode calls. All hashes match, including the previous optimized 256-prompt/eight-output case.
- Minimum host available memory: 77.54 GiB; no swap growth.
- Explicit model release left 8.125 MiB tracked allocation and 40 MiB allocator reservation in the still-running worker. These are residual runtime allocations, not a claim of exactly zero memory at that stage.
- After worker exit, a fresh CUDA probe reported zero tracked allocation and reservation. NVML compute-process inventories were empty before/after model loading.
- Worker exit 0. Agent stop 0.165 seconds, container exit 0, pid0, OOMKilled=false. The original backup and model-server containers also remain stopped. Boot ID remained unchanged.

## Memory after shutdown

Host available memory recovered from 115.35 GiB before to 114.59 GiB afterward, within 0.76 GiB of the starting level. However, host file cache grew from 3.33 GiB to 20.11 GiB while CUDA-global free fell from 109.96 GiB to 92.33 GiB. These changes are consistent with checkpoint reads remaining in OS page cache on shared physical memory; per-file residency was not established. The optional `fincore` utility was unavailable and no packages were installed.

No live model remained in the experiment containers, but a clean process/container exit does not flush filesystem caches or prove all driver resources have returned to their starting state. Host available and CUDA-global free memory are different views of the same physical capacity and must not be added. No global cache flush, swap change, driver change or unrelated service action was performed.

The previous 27B shutdown remains unexplained. A later 27B trial needs its own conservative physical-memory/headroom review; success here is not permission to resume the old mixed workload automatically. No additional optimization was selected or implemented.

## Evidence

[Summary](summary.json), [9B measurements](9b.json), [trial status](trial-status.json), [container replacement](container-replacement.json), [empty lifecycle](empty-lifecycle.json), [guard recovery](retry-origin.json), [final process inventory](process-inventory.txt). Full per-phase memory logs and the preserved failed attempt remain in the experiment directories named in the status/recovery records. Container-inspect data containing environment values is private outside Git.

Harness source checkpoint: `28d2baf`; native local-Qwen implementation remains `18894a8`. See [methodology](../../../BENCHMARKS.md), [earlier shutdown evidence](../2026-11-14-final-decode/README.md) and [options](../../../OPTIMIZATION_OPTIONS.md).
