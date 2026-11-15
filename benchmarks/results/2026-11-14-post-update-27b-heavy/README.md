# Post-update real 27B heavy trial

The explicitly requested repeat of the workload that previously powered off the host **passed**, including cleanup. This is actual downloaded 27B GGUF inference, not a tiny-model test. No native engine implementation changed; its hashes match local-Qwen implementation `18894a8`.

One native model, batch 1 CUDA BF16, 1024 raw prompt tokens and 128 outputs with EOS disabled, one warmup, three timed repetitions and a separate CUDA-event diagnostic. No profiler or coding server. Allocator cap 56 GiB, host floor 32 GiB, new-swap budget 256 MiB. No lighter inference trial preceded this run. The general crash and legacy-runner holds remain active outside this one-shot authorization.

| Measurement | Result |
| --- | ---: |
| Fresh-process load, caches uncontrolled | 144.053 s |
| Median generation including prefill | 31.417 s |
| Timed range | 31.409–31.479 s |
| Output tokens/s including prefill | 4.074 |
| First token, separate CUDA diagnostic | 1911.95 ms |
| Prompt tokens/s, CUDA diagnostic | 535.61 |
| Useful decode tokens/s, CUDA diagnostic | 4.300 |
| Peak tracked CUDA allocation including load | 52.47 GiB |
| Minimum host available memory | 46.56 GiB |
| New swap growth | 0 bytes |
| Highest sampled GPU temperature | 81.0 °C |
| Highest sampled ACPI thermal zone | 88.4 °C |

All three timed outputs and the diagnostic have identical hashes, also matching the saved pre-update native output. Each response selected 128 tokens with 127 useful decodes. No fresh reference-model/numerical regression was run; the earlier unchanged correctness gates remain prior evidence.

Explicit release left 8.125 MiB allocated and 20 MiB reserved inside the worker. Worker exit 0; fresh CUDA probe allocation/reservation 0/0; NVML compute inventories empty. Container stopped in 0.114 s with exit 0, pid 0 and OOMKilled=false. Both experiment containers and the controller were confirmed stopped. Host available memory recovered 115.35 → 115.19 GiB. Boot unchanged; no trial-window OOM, Xid or thermal fault in kernel journal. Host and CUDA memory views share physical memory and must not be added.

Read-only post-update inventory verified EC `0x03000d07` (3.13.7), SoC `0x02009e0f` (2.158.15), PD `0x00000519` (0.5.25), kernel `7.0.0-1025-nvidia`, driver `580.178.04`. The user applied the Dashboard update and reboot; the supervisor did not install firmware. Both firmware and kernel changed, so this is a post-update-stack observation, not an isolated firmware experiment. One successful trial cannot establish long-session stability, identify the prior shutdown cause, or prove a speedup. A heavy capped trial had also passed once before the update.

[Summary](summary.json), [measurements](27b.json), [trial status](trial-status.json), [plan](trial-plan.json), [one-shot runner](run_post_update_heavy_trial.py), [read-only sampler](monitor_post_update_trial.py). Full fsynced phase/host/thermal telemetry remains at `/home/cap/spark-inference-overnight-20261113/runs/post-firmware-27b-heavy-20261115T000929Z` and the measurements' raw-output directory. Private firmware/device inventories remain outside Git.

Next: review this reliability result and the existing optimization options before selecting further work. No additional workload or optimization was automatically launched.
