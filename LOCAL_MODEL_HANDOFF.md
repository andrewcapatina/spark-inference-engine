# Engine development handoff

The user authorized Codex to implement the current GDN chunk batching work
directly, as an exception to the earlier local-model-only authoring rule.
Do not infer that model weights were trained or updated: this file supplies
project context for future coding prompts.

## Accepted baseline before this experiment

- Engine `3139a9b`: defer BF16 value widening until the existing FP32-beta
  product. FP32 output/state and original arithmetic are preserved.
- Report `f8df2cf`; previous accepted gated normalization engine `ea70e47`.
- QKV reuse, FP8 preparation and both normalization tails are accepted.
- State-tail fusion is quarantined after an unexplained generation mismatch.
  Paired RHS/solve fusion and four solver layouts were not worth adopting.
  They are retained in reports but excluded from active engine code.

## Completed experiment

The rejected candidate batched only independent chunk-local transforms in
`qwen_ops.gated_delta_chunk`.
The carried-state scan, its order and its FP32 expressions stayed sequential.
Preparation used the same lower-unit triangular solves, products, cumulative
decays and masks as the accepted implementation. It batched at most eight chunks,
subject to a conservative 256 MiB temporary-workspace estimate; unsupported
modes and large per-chunk workspaces retained per-chunk execution.

Status: **not adopted; accepted engine 3139a9b restored byte for byte**.
The authoritative run pointer is
`/home/cap/spark-inference-overnight-20261113/feedback/prefill-batch-main-run.txt`.
Its manifest records source hashes, author authorization and frozen baseline.

The initial broad batch path changed triangular-solve rounding for small
right-hand-side widths. Restricting width alone was insufficient: small head
counts also changed solver arithmetic. A specialization for single-request,
48-value-head, 128-wide, 64-token chunks passed exact checks, including actual
convolution output layouts. All other shapes retained the original path.

That specialization passed 304 direct cases, 44 dense/quantized blocks,
20 tiny models, two gradient cases and 11 CPU/provided-state-dtype cases.
However, grouping 2/4/8/16 independent chunks was respectively about
7%/13%/21%/25% slower at 1,024 tokens in the convolution-layout sweep.
Head-major batching caused repeated implicit copies; explicit chunk-major
packing reduced but did not remove the regression. No real-checkpoint timing,
generation or new quality result was claimed for this rejected candidate.
Do not describe the tiny/component tests as real-checkpoint validation.

Complete candidate sources and earlier revisions were retained before exact
rollback. All four containers stopped exit 0, no GPU jobs, same boot and no new
swap; service inactive and STOP/GPU/LARGE holds preserved. The report is
[GDN chunk batching](benchmarks/results/2026-11-16-gdn-chunk-batching/README.md).
Do not retry plain Torch batching without a concrete plan to avoid packing,
extra materialization and changed vendor-solver arithmetic. A kernel that
operates on the existing layout is a separate, unimplemented option.

## Validation and containment

Keep existing numerical tolerances and fixtures unchanged. Require exact
component/tiny-model outputs, state ownership and continuation; then entire
real 27B prefill/next-decode logits/caches and two repeated 128-token generations
per backend and prompt length. Preserve the frozen 40-case quality suite and
446 token losses, plus all 56 Qwen and 15 Llama gates. A primitive speedup is
not a real-checkpoint speedup. Compare warm direct prefill at 256/1,024 tokens;
do not call that sustained generation throughput.

One large model at a time. Keep the unattended service inactive and STOP,
GPU_WORKLOAD_HOLD and LARGE_WORKLOAD_HOLD present. Use existing guarded Docker
controllers, memory/swap limits and clean teardown. No package installation,
downloads, subagents, unrelated services or serving-engine substitutions.
Retain complete rejected source before rollback. Update this handoff with the
actual final outcome and report; never label an incomplete test as passing.
