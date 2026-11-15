# Reuse attention KV concatenations

Local Qwen task022 changes only `QwenAttention.decode`: save the first, unrepeated K/V concatenations and return them as the cache, instead of concatenating the same tensors again. Group expansion remains temporary attention work. The supervisor wrote the harness and placed the retained local-model source verbatim. Accepted QKV reuse remains present; convolution-history compaction stays quarantined. No preallocation is included here.

## Measurements

One resident native27B NVFP4 model alternates original/reuse/reuse/original. Two warm uninstrumented 128-output timing samples per method and prompt exclude model loading. Small throughput differences do not establish a meaningful speedup.

| Prompt tokens | Total 128-output tokens/s, old → reuse | Fewer CUDA kernels per 16-output profile | Fewer cumulative cat allocation requests |
| --- | --- | --- | --- |
| 256 | 9.049 → 9.066 | 480 | 247.5 MiB |
| 1024 | 8.249 → 8.299 | 480 | 977.5 MiB |

The profile loses exactly480 decode concatenations and480 kernels at each length. All decode `aten::cat` calls fall2655→2175; the remaining calls include other engine work. Allocation-request totals are cumulative, not peak memory or measured memory bandwidth. Prefill is unchanged by this method; its small timing variations are noise. Prefill peak allocation is identical in the paired runs. This is a small cleanup with reduced redundant work, not a substantial performance optimization.

## Validation

- 110 tiny dense/mixed-FP8 numerical cases pass exactly, including batch1/2, grouped and ungrouped attention, noncontiguous hidden inputs, three decode steps and branching. Prefixes and retained caches remain unchanged; KV-axis concatenations fall4→2.
- Real27B256/1024 prompts preserve entire prefill logits/cache values exactly, and all128 generated IDs agree across all alternating runs. Failure diagnostics retain expected and actual token IDs before raising.
- Frozen quality screen40/40, all generated IDs and446 token losses exactly match accepted QKV baseline. Weighted perplexity remains8.946730. The fixed vLLM-reference screen also passes. This remains a small screen; broad quantized reference parity is not claimed.
- All56 unique Qwen gates and15 Llama cases pass, including separate real9B/27B numeric and conversation processes. Original fixture hashes and numerical tolerances are unchanged.
- Profile inputs and outputs match baseline/control/paired-generation prefixes. All CUDA kernels are attributed. Profile overhead is excluded from the 128-output throughput measurements.
- Native workers release CUDA allocations/reservations to0/0; containers exit cleanly with no OOM, host boot unchanged and bounded swap growth recorded in cleanup evidence. One large model at a time, service MainPID0 and global holds retained. Cleanup evidence is retained separately.

## Reproduction and artifacts

Use the installed pinned runtime image `sha256:fa7334538384d67c72284edc081c78b1149a36d53aba1fb3c5f7d1d534154ec6` (Torch2.13+cu130), `unsloth/Qwen3.8-27B-NVFP4` revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`, and this exact source manifest. Protected original-model tests use their preserved runtime. Main run: `/home/cap/spark-inference-overnight-20261113/runs/attention-kv-reuse-20261115T073543Z`.

With no experiment CLI/controller running:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_attention_kv_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_attention_kv_prepared.py ab
python3 /home/cap/spark-inference-overnight-20261113/finish_attention_kv_validation.py
```

Result JSON, source provenance and exact harness copies are included here; weights and large raw traces stay outside Git. Next experiment: per-request capacity-backed attention cache with preserved snapshots and branch ownership, evaluated independently before adoption.
