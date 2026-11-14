# Larger 27B trial: 256/ 128 passed, 1024/ 128 powered the host off

The user confirmed that the system **powered off by itself**, using the original NVIDIA adapter and cable. This was a host failure, not a Python exception or an orderly container stop. The root cause is unproven. No further GPU workload or retry was started after the new boot.

## What completed and what did not

- **256 prompt / 128 output:** one warmup, three timed repetitions and a separate CUDA-event diagnostic completed. Median generation was 29.799s (4.295 output tokens/s including prefill). Outputs matched. Worker exit 0, container exit 0, model release/GPU-process inventory/fresh CUDA probe/host-memory recovery all passed before proceeding.
- **1024 prompt / 128 output:** a fresh worker/container boundary loaded 27B only. Warmup completed, then timed repetition 0 completed in 31.800s with the same output hash as the prior native 1024/ 128 case. The machine powered off shortly after repetition 1 began. One completed sample is not a three-repetition median, completed case, or stable-workload validation. No event diagnostic or clean release completed for this stage.

The interrupted worker stdout contains NUL-filled/truncated output. The fsynced phase log is more informative: it records `after_sample` for repetition 0 and `before_sample` for repetition 1. Original raw worker data was not repaired or promoted to complete.

## Conditions established before the failure

Only one experiment Python model worker was recorded, and the coding-model server remained stopped. The controller was inactive and experiment locks were held. The 256 stage completed its clean shutdown before the 1024 stage started. Profiling was disabled. CUDA BF16, 56GiB PyTorch allocator cap, 32GiB host available-memory floor and 256MiB new-swap growth guard were retained. Native implementation and numerical expectations were unchanged.

Before these stages, file-scoped `POSIX_FADV_DONTNEED` advice released cached pages associated with five inactive 9B checkpoint files, opened read-only. CUDA-global free rose from 65.41 to 82.11GiB, while host cached memory fell by approximately 16.71 GiB. File size/mtime stayed unchanged. No global cache flush, driver/swap change, installation or unrelated-service action occurred. The existing 80 GiB CUDA preflight gate was met before both model loads. Cache advice is advisory; measured headroom was checked rather than assuming success. See the [Linux interface definition](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html).

This changed cache history and isolated each prompt length in its own process. These results must not be treated as an unchanged continuation of the original before/after optimization comparison.

## Last durable observations

- Host telemetry at 2026-11-14T23:32:31.302847+00:00: **59.07GiB available**, zero swap use/growth.
- Before the failed repetition: **26.29GiB CUDA-global free**, 50.106GiB allocated and 52.477GiB reserved. The prior completed sample peaked at 51.732GiB allocation, below the 56 GiB cap.
- GPU health at 2026-11-14T23:32:30.840222+00:00: **79°C**, 84.45W reported GPU power, SM clock 2392MHz. The preceding sample was 73°C. These are periodic samples, not proof of the absolute temperature/power peak or a thermal/power protection trip. CPU/VRM/adapter temperatures at failure were not recorded.
- New boot ID:`761dc008-6038-4238-b04f-95058c87ce87`. Docker observed agent exit 255/OOMKilled=false after reboot; this is not a verified model-worker exit code. No tied OOM/GPU-fault or orderly shutdown record was found in persisted kernel messages. Missing logs do not exclude a sudden unrecorded failure.

Host available memory and CUDA-global free are shared-memory views and must not be added. No ordinary progressive retained-allocation growth was observed, but an unrecorded transient or non-PyTorch/driver allocation cannot be ruled out. The 79°C sample by itself does not establish overheating. Likewise, original power hardware does not exclude a faulty supply, connector, cooling, firmware or driver problem.

The failure recurred despite clean teardown, one-model isolation and recovered headroom. Those measures are therefore not sufficient to declare the larger workload reliable, and the recorded data does not justify diagnosing OOM or blaming the small final-decode optimization. An earlier engine revision also suffered host shutdowns.

## Current state and next decision

Both experiment containers and the original controller are stopped. Global `GPU_WORKLOAD_HOLD`, controller `STOP` and a new `LARGE_WORKLOAD_HOLD` prevent automatic repeats. Larger trial entry points now fail fast while the latter hold exists. No new optimization is selected; the local-Qwen native source remains unchanged at `18894a8`.

The next step is system-level diagnostic collection and review. The installed `/usr/bin/nvidia-bug-report.sh` supports `--safe-mode`, but requires root according to its own privilege check. The field-diagnostic package is not installed; it was not downloaded or run. NVIDIA's [support guide](https://docs.nvidia.com/dgx/dgx-spark/support.html) describes hardware support and separate field diagnostics; those can require software/boot changes and are not part of this experiment's authorized automatic actions. The [known-issues page](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) identifies non-supplied adapters as a possible shutdown cause, but the user reports original hardware here.

## Evidence

[Summary](summary.json), [completed 256 measurements](256-prompt-128-output.json), [256 teardown](256-clean-shutdown.json), [partial 1024 data](1024-prompt-128-output-partial.json), [durable 1024 phase log](1024-phases.jsonl), [GPU health](gpu-health.jsonl), [host tail](1024-host-tail.jsonl), [single-worker inventory](1024-process-inventory.txt), [cache/headroom experiment](cache-headroom.json), [crash record](third-host-crash.json). Full raw/kernel logs remain at `/home/cap/spark-inference-overnight-20261113/runs/clean-27b-large-20261114T232636Z`. No benchmark median was fabricated for the interrupted case.
