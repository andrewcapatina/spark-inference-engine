# Overnight inference-engine experiment

Run date: November 13, 2026. Stop by 8:00 AM America/Los_Angeles.
Coding model: the downloaded DavidAU Qwen3.8 27B GGUF, Q8_0 variant, served locally.

Build a small working inference engine. Start with one model and one request.
Make correctness easy to assess before pursuing performance. This is a bounded
experiment: a partial, tested implementation is useful; unsupported claims are not.

## Environment

- Worktree: `/workspace` in an isolated OpenClaw container.
- Python 3.11, PyTorch 2.9.1, Transformers 4.57.3, Safetensors 0.6.2,
  NumPy 2.2.6, and pytest 8.4.2 are supplied.
- Small reference checkpoint: `/workspace/reference/smollm2` (read-only), a snapshot of
  HuggingFaceTB/SmolLM2-135M-Instruct. Record the exact revision from its manifest.
- Test on CUDA. The coding model uses the Spark GPU in a separate server.
- Network and other host projects are unavailable to the agent.

## Milestones, in order

Follow ENGINE_CONTRACT.md for exact interfaces and the controller's current
stage in run_feedback/latest.txt. Repair reported failures before advancing.

1. Create a minimal Python package, README, CLI, and meaningful smoke tests.
   Document how to run from a checkout without installing more dependencies.
2. Implement a configurable Llama-style decoder: embeddings, RMSNorm, RoPE,
   grouped-query causal attention, gated feed-forward layers, and output head.
   Use PyTorch operations, but own the forward pass. First test tiny randomized
   configurations against Transformers' Llama reference.
3. Load the supplied checkpoint's config and Safetensors weights with correct
   tensor mapping, tied embeddings if applicable, and its tokenizer. Compare
   logits on fixed input IDs with the reference using stated tolerances.
4. Implement greedy token generation with EOS and length limits. Provide a CLI
   for a short prompt and document observed output from the real checkpoint.
5. Add explicit KV cache state, prompt prefill, and incremental decode. Compare
   cached and uncached results; test positions, reset, and separate requests.
6. Add a small benchmark reporting prompt and decode timing, environment, input
   lengths, and clear CUDA-only limitations. Optimize only after profiling.
7. If all core checks pass, document the adapter work needed for the downloaded
   Qwen hybrid models. Treat full Qwen support, GGUF quantized kernels, vision,
   MTP, custom CUDA, multiple requests, and multiple Sparks as later work.

## Evidence required

Keep tests small enough to run repeatedly. Prefer fixed seeds and numerical
tolerances to full generated-text equality. Include a real-checkpoint comparison
when the loader is ready. Record failures rather than loosening tests to hide
them. Run tests with a bounded CUDA thread count (for example OMP_NUM_THREADS=4).
After each turn write PROGRESS.md with what exists, exact checks and outcomes,
known gaps, and the next task. Finish with a concise report in that file.
