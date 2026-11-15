# Native NVFP4 preliminary accuracy screen

**Passed all criteria fixed before outputs.** No engine implementation changed.

| Measure | Native engine | vLLM reference |
| --- | ---: | ---: |
| Fixed-answer tasks correct |40/40|40/40|
| Truncated answers |0|0|
| Token-weighted prose perplexity, lower is better |8.946730|9.073222|
| Prose tokens scored |446|446|

Native/reference perplexity ratio0.986059: about1.4% lower for native on these four documents. This is not evidence of a general quality advantage. There were no native-only, reference-only or shared task failures. Both stages completed and shut down cleanly.

## Frozen protocol

The [40-case suite and four original prose documents](../../accuracy/nvfp4-smoke-v1.json), grader and six grader tests were committed in **a836fab** before any model outputs. Suite SHA256:a06956fa688f340743e16342bd5e12fc33c4ff788dfae557fc6e1e144b54cc28. Criteria were minimum90% native task accuracy, zero native-only task failures, token-weighted perplexity ratio<=1.05, and no length-truncated task replies. None was adjusted after seeing outputs.

Categories:8 arithmetic,6 logic,6 extraction,6 formatting,6 code-reading,4 factual and4 conversation-context tasks. Conversation cases use fixed transcripts, not generated earlier replies. The catalog extraction prompt is the longest at972 tokens. Each chat request is rendered with the same checkpoint tokenizer, enable_thinking=False, greedy decoding, and64-output-token limit. Both engines receive identical submitted token IDs; comparison checks every task's IDs and every corpus sequence. The prose documents are scored as raw completion sequences with the first unconditioned token excluded, using all446 remaining next-token likelihoods. Perplexity is exp(total negative log likelihood / total tokens), not an average of document perplexities.

The same downloaded Unsloth Qwen3.8-27B-NVFP4 revision f0b7c9e722f5565102fff8481c99e4d86ae099c7 is used for both. Reference runtime is the existing pinned vLLM0.32.1.dev0+g5df63a720.d20261019/Torch2.13 image, explicit BF16KV, eager execution, one sequence, prefix caching off,512MiB KV budget, no MTP or CUDA graphs. Its attention backend is FLASH_ATTN2, as in the prior matched-cache diagnostic. This reference differs from the earlier FP8KV throughput baseline; these results are not a new performance benchmark.

## Scoring and controls

[accuracy_scoring.py](../../../tools/accuracy_scoring.py) validates whole numeric answers, CSV item order, and JSON values/types while ignoring JSON key order. It separates semantic correctness from format compliance. Text facts ignore case and trailing periods; uppercase tasks require exact case. Formatting tasks must also satisfy their requested raw format, including no Markdown fence for raw JSON. Extra explanations are not silently reduced to a guessed answer.

Six portable grader tests pass. Five additional CPU controls check token weighting, paired regressions despite equal total accuracy, rejection of different prompt IDs, truncation failures and the likelihood-ratio criterion. Final evidence verification regrades every saved reply and independently recomputes document sums and perplexities from per-token NLLs. Raw replies, submitted IDs, scores and per-token NLLs are preserved here.

Per-document native/reference perplexities: workshop8.5265/8.7613; software10.9084/11.0252; journey8.3809/8.2360; garden8.1557/8.4518. The small aggregate advantage is not uniform across documents.

## Meaning and limitations

This is a small original screening set, not a standard benchmark, statistical noninferiority study, broad task-quality certification, or evidence for longer conversation stability. It supports the current basic text/chat path on these inputs. The [independent numerical reference discrepancy](../2026-11-14-nvfp4-layer-reference/README.md) remains real and is not reclassified as numerical parity because this screen passed. The separate official-HF graph control with matching projection primitives remains bit-identical at its tested prefix.

Existing56Qwen/15Llama numerical regressions remain the previously passed engine checkpoint, with original tests and engine hashes unchanged. They were not rerun for this harness-only evaluation. No implementation patch or performance optimization was selected.

## Containment and reproduction

Native then reference ran sequentially in spark-prototype-vllm-nvfp4; the coding model and other experiment containers stayed stopped. Native allocator cap32GiB; host floor32GiB and new-swap guard256MiB. Parent controllers acquired both experiment locks and refused duplicate controllers. Both worker/container exits0, PID0/OOMfalse, same boot, no new swap, recovered host available memory within4GiB. Native model weak references are gone and tracked CUDA release is0/0; post-reference probe is0/0 with no compute processes. These finite runs do not explain or rule out earlier host poweroffs.

All four experiment containers were stopped afterward; legacy service inactive MainPID0 and global holds preserved. Exact commands, source hashes and boot/cleanup evidence are in stage-status JSON. Archived harness sources use .py.txt to avoid accidental test collection. No weights are committed.

Durable native run: /home/cap/spark-inference-overnight-20261113/runs/accuracy-native-20261115T035947Z. Reference run: /home/cap/spark-inference-overnight-20261113/runs/accuracy-reference-20261115T040352Z. Worker: acceptance/evaluate_quant_accuracy.py; parent: run_accuracy_evaluation.py; CPU paired comparison: compare_accuracy_evaluation.py.

With this preliminary screen passed, the next performance work should profile the current native NVFP4 path and present measured alternatives before choosing an optimization. Broader standardized/held-out quality evaluation remains worthwhile before stronger accuracy claims.
