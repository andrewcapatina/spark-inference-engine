# Current optimization work

The selected target remains prompt-processing latency. The user selected three storage/work changes in order: QKV reuse, compact convolution history, then attention-cache allocation/copy improvements. Engine implementation belongs to local Qwen.

**QKV reuse is validated at6dd8d9f.** One raw projection serves both convolution and its history cache. Exact real27B outputs/caches/128generated IDs are preserved;28tiny cases,40-case quality screen and56Qwen/15Llama gates pass. Same-model prefill medians fall297.5→275.5ms (256tokens) and1227.0→1173.3ms (1024tokens). Sustained generation remains essentially unchanged. [Full evidence](benchmarks/results/2026-11-14-qkv-reuse/README.md).

| Next work | Evidence / boundary |
| --- | --- |
|Compact convolution history — quarantined|Memory reduction confirmed (960→3.75MiB at1024), but one initial generation mismatch remains unexplained despite exact replays. Default code unchanged. [Evidence](benchmarks/results/2026-11-14-conv-history/README.md).|
|Attention-cache copies and allocation|Decode concatenates growing K/V arrays, then repeats concatenation for returned caches. Remove duplicate work, then test preallocation while preserving prefix values, separate requests and cache-branch ownership.|
|Fused normalization / FP8 preparation|Earlier profiles show meaningful cost; future independent numerical and whole-model checks required. Not currently selected for implementation.|

The [compiler experiments](benchmarks/results/2026-11-14-gdn-prefill/README.md) are rejected and rolled back. FlashInfer is faster in isolation but fails final-state bounds; ordinary compilation passes per-operation bounds but drifts through the full model. A C++ wrapper gives no material gain. Exact graph replay has modest/inconsistent operation gains. No thresholds were loosened and no compiler backend remains enabled.

The earlier [profiling options](benchmarks/results/2026-11-14-qkv-reuse/previous-options.md) retain component shares and unselected alternatives. Timing, memory and diagnostic milestones must remain distinct from real-checkpoint acceptance.
