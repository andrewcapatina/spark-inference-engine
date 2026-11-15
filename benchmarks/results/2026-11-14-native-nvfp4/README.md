# Native NVFP4 update: evidence and validation scope

All native implementation was authored by the local Qwen model. The supervisor authored only prompts, containment, exact text placement, independent reference fixtures, measurement and documentation. Accepted source checkpoints: `c491ef7` NVFP4 operator, `c2d74ce` FP8 operator, `c207950` strict streaming loader, `56c6f08` public dispatch and `ac4cbc0` chat API compatibility. Exact responses, rejected candidates and placement/source hashes are preserved here.

The engine uses its existing Qwen architecture and generation methods. NVFP4 GEMMs call FlashInfer/CUTLASS primitive kernels; FP8 GEMMs call PyTorch scaled_mm. It does not delegate model loading or inference to vLLM, Transformers or Ollama. The runtime image contains those packages, but native inference is the project code. Loading builds the model on meta, streams individual Safetensors tensors, and retains packed U8/FP8 matrices. Vision/MTP are skipped; KV cache remains BF16, explicitly ignoring source FP8 KV calibration.

## Passed scopes

- Independent tiny CUDA NVFP4/FP8 matrices, nontrivial scales, address-map/layout, multidimensional inputs and packed storage checks. Numerical tolerances unchanged.
- Nine CPU reader checks, including original exception preservation, unknown/unconsumed/duplicate keys, path traversal and ignored-only supported metadata.
- Sequential tiny reference/native two-layer hybrid model: forward, prefill and decode max absolute error0; exact4-token generation; missing/unexpected weight rejection. The reference model is released before loading the candidate.
- Actual downloaded27B checkpoint: 168NVFP4 and233FP8 linears,22,586,157,376 resident tensor bytes (about21.04GiB), exact first8 generated tokens matching the vLLM reference at256/1024 prompts. Two chat turns answer4 then7. This is not full-logit or broad quality validation.

## Recovered blockers

Source review rejected missing shape validation, invalid scalar-tensor handling, FP8 signed-max scaling, incorrect meta construction/import paths and an invalid ordinary-linear assignment branch. A real tiny CUDA failure caught FP8 activation strides; local Qwen repaired it. All rejected source was retained before replacement.

The first real run reached exact raw generation but chat failed because the new Transformers runtime defaults to BatchEncoding. Local Qwen added only return_dict=False; an AST guard proved other method behavior unchanged. CPU fixtures on old/new versions and the real two-turn conversation passed afterward.

The second worker completed all functional checks, but its parent cleanup guard rejected72MiB left after model release. A tiny no-model control reproduced exactly72MiB: cuBLAS clear reduced it to40MiB, and clearing idle FlashInfer workspace cache reduced it to0/0. This is runtime scratch storage, not evidence of retained model weights. The benchmark harness now checks weak references to model tensors, explicitly releases idle library workspace after timing, and retains the original16MiB allocation/64MiB reservation bounds. Eleven portable measurement tests pass. Corrected real-model benchmark teardown subsequently passed0/0 with no surviving model tensor references.

## Completed follow-up

All56 Qwen gates and15 Llama cases passed unchanged. Native benchmark completed9.17/8.35tok/s with clean teardown. Full128-token reference matching passes at1024 prompts but differs at256 prompts even with BF16KV reference; real-checkpoint numerical parity remains partial. See [final benchmark and accuracy scope](../2026-11-14-native-nvfp4-benchmark/README.md).

Reference backend: [actual vLLM measurements](../2026-11-14-nvfp4-vllm/README.md), about11.45/11.09tok/s at256/1024 prompts with128outputs. That runtime auto-selected FP8 KV; the native path deliberately uses BF16 KV. The former native BF16 measurements use different checkpoint weights and an older PyTorch runtime, so cross-stack numbers do not isolate a pure quantization effect.

Full durable logs: `/home/cap/spark-inference-overnight-20261113/runs/nvfp4-native-20261115T005339Z`. No model weights or bulk traces are committed.
