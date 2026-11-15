# Actual NVFP4 vLLM reference

Downloaded Unsloth Qwen3.8-27B-NVFP4 revision f0b7c9e722f5565102fff8481c99e4d86ae099c7 ran in the installed GB10 vLLM image fa7334538384d67c72284edc081c78b1149a36d53aba1fb3c5f7d1d534154ec6. Offline, one model, batch1, BF16 activations, mixed FP8/NVFP4 weights, FlashInfer/CUTLASS kernels. No speculative decoding, prefix reuse, compilation or CUDA graphs. KV budget512MiB, context2048; memory utilization0.40.

Both prompt lengths used one warmup and three timed128-output repetitions with EOS disabled. Wall timings include vLLM request/response overhead and prefill, excluding load/tokenization. Median throughput: 256prompt: 11.453tok/s (11.176s), 1024prompt: 11.093tok/s (11.539s). All timed outputs match within each case. Loading/JIT/autotuning is reported separately, not guaranteed cold-disk.

This is a reference backend, not native engine support. Current native BF16 uses different DavidAU checkpoint weights and an older PyTorch runtime; observed speed ratios do not isolate quantization. Worker0, fresh CUDA allocation/reservation0/0, NVMLempty, container0, hostavailable recovered within4GiB, boot unchanged. Native/coding containers stayed stopped. Global holds retained outside explicit trials. Full telemetry/logs: /home/cap/spark-inference-overnight-20261113/runs/nvfp4-reference-20261115T004643Z.

Next authorized phase: local-Qwen-only native packed NVFP4 and FP8 operators, independent numerical fixtures, then loader/integration and existing BF16 regression.
