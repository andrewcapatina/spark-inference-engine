# Identical-input NVFP4 numerical diagnostic

The user authorized normalized-logit and teacher-forcing comparisons after the256-prompt free-generation discrepancy. No engine implementation changed. All engine source hashes match the preceding benchmark. No numerical tolerance was loosened or fitted to these observations.

## Finding

The first free-generation disagreement is output token38 (zero-based37). All preceding prompt/output token IDs are identical. Native generation chooses3296; vLLM cached generation chooses6165. The latter reproduced exactly in the new reference run, including all128 output IDs.

At that SAME prefix, both engines choose3296 when starting from a fresh prefill. The native cached path also chooses3296. Thus the disagreement exists between vLLM's own cached and fresh-prefix paths. This is evidence of execution-path sensitivity, not proof that either runtime implements the intended quantized computation correctly. Kernel, GDN recurrence/chunking, cache state and quantized arithmetic differences remain possible causes; none is isolated here.

| Execution at the identical first-divergence prefix | Preferred token |
| --- | ---: |
| Native incremental cache |3296|
| Native fresh prefill |3296|
| vLLM fresh prefill |3296|
| vLLM incremental generation |6165|

Native cached probability for3296 is99.71%; vLLM cached probability for6165 is75.22%. This cannot simply be dismissed as a near-tie on the measured paths. The cached reference logprob margin between6165 and3296 is1.25.

At this prefix, we captured all248,320 normalized vocabulary scores. Native cached versus vLLM fresh-prefill total variation is0.006774 (about0.68% probability mass), KL(native||reference)0.006293, top20 overlap11/20. Native fresh-prefill versus vLLM fresh-prefill total variation is0.006612. Native cached versus its own fresh-prefill is0.001157. Tail logprob errors are much larger: native cached/reference mean absolute2.223, max9.088. These are measured diagnostics, not a newly defined passing threshold.

## Teacher-forcing scope

The forced history is the earlier vLLM BF16KV128-token output. Native prefills the256-token prompt then incrementally consumes those exact teacher tokens. vLLM scores prompt+teacher tokens in a causal full-prefix prompt-scoring request; it does NOT forcibly replay those tokens through its incremental cache. We explicitly retain this execution-path difference. A separate fresh-prefix request at the first divergence supplies the full-vocabulary comparison above.

Top choices agree at125/128 forced positions. Disagreements at zero-based17,92,94 are retained with scores. Some disagreements have substantial confidence differences, so this is NOT declared full real-checkpoint numerical parity. The first free-generation divergence at37 is not one of those prompt-scoring disagreements. Mean target NLL is0.09865 native and0.12031 reference prompt scoring on this repetitive synthetic sequence; this is not a language-quality or general perplexity benchmark.

The full-vocabulary arrays contain FP32 normalized log probabilities: native BF16 output logits followed by FP32 log_softmax, and vLLM's returned logprobs. Raw logits are not compared because normalization discards a shared additive offset. Identical token history does not imply identical numerical state when different kernels and cache paths are used.

## Recovery and containment

The first reference diagnostic failed before weight loading because the supervisor harness lacked a multiprocessing main guard. Its script/log/status were preserved; the harness was repaired and rerun. The first CPU analysis also targeted a root-owned container output directory; its output destination was corrected to the host-owned run directory. Neither incident was an engine failure.

Native and reference models ran sequentially, never simultaneously. Native tracked CUDA release0/0, no surviving model tensor weakrefs; both stages worker/container exit0, PID0/OOMfalse, no new swap and recovered host available memory within4GiB. Same boot, experiment service inactive; all four experiment containers stopped afterward. Original numerical regression fixtures remain unchanged;56Qwen/15Llama gates are the prior passed checkpoint, not rerun for this harness-only diagnostic.

Runtime remains the pinned existing Torch2.13/CUDA13 image with vLLM0.32.1.dev0+g5df63a720.d20261019 and FlashInfer0.7.0.post1. Same Unsloth NVFP4 revision, BF16 reference KV, FLASH_ATTN2, eager execution, no MTP/graphs. Host guards32GiB available and256MiB new swap; native allocator cap32GiB. This is not a new performance benchmark.

## Reproduce and next accuracy work

Diagnostic controller: run_teacher_diagnostic.py; worker: acceptance/teacher_forced_quant.py; CPU analysis: analyze_teacher_diagnostic.py. Exact launch arguments and source hashes are retained in stage-status JSONs. Archived source has .py.txt extensions to prevent accidental GPU test collection. Input IDs and compact top20 outputs are included; full arrays and logs remain in the controller directories and their hashes appear in local-array-manifest.json. No weights are committed.

Before claiming numerical correctness or making another performance change, isolate the three same-history disagreements and the reference cache/fresh-prefill sensitivity with layer-level captures and an independent floating-point/dequantized reference. Quantized operator tests and preserved BF16 regression gates still apply. Representative chat/task quality evaluation remains separate and pending. There is insufficient evidence here to justify an engine patch.
