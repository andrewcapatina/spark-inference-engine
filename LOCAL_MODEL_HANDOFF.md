# Engine development handoff

Current accepted engine: `da80ee6` (MLP activation fusion), following `f1e760f`
(GDN key preparation). Current report:
benchmarks/results/2026-11-16-mlp-prefill-activation/README.md.
Local Qwen authored all MLP implementation: helper059 and forward061, copied
verbatim after source review. Codex only supervised harness/tests/prompts/placement.
Single-request CUDA BF16 inference prefill T>=16 uses fused SiLU then multiply
with explicit BF16 intermediate rounding. Decode/CPU/grad paths keep the original
expression. Missing Triton fallback is narrow; other errors propagate.

The comparison evaluated preparation reuse and activation separately. All56NVFP4
gate/up input scales match;8FP8 pairs can share prep. Reuse remains prototype-only.
Activation was chosen for larger/simple dominantNVFP4 benefits, no module cache.
Real27B warm prefill:256.0387→251.1717ms at256tokens (save4.8670ms,1.9009%);
1002.5413→985.9550ms at1024 (save16.5863ms,1.6544%);8/8ABBAblocks positive each.
These are prompt-processing gains, not output-token throughput measurements.

Validation: all65536BF16gate patterns x9multipliers on both Torch2.13 and2.11
runtimes,12real-weight layer cases,48integration cases,gradient/import/error
fallbacks; full real27B prefill+nextdecode logits/allcache and8x128IDs exact;
frozen40answers/446NLL exact;56Qwen/15Llama protected checks passed including
original9B/27B numerical and conversation tests. Profiler:64fused launches replace
128pointwise launches, other majorcounts unchanged; peak allocated memory unchanged.
All4containers stopped, GPU idle, service MainPID0, sameboot, zero new swap, holds
retained. Main pointer: feedback/mlp-main-run.txt. Earlier055/056/060drafts were
retained/rejected; local model repairs supplied the selected source.

Historical generation outlier remains unexplained; no recurrence in this run.
Future failures retain logits and use the recorded commit/source peelback plan,
including uncommitted prototypes. Passing repeats do not establish a root-cause fix.
Earlier completed/unadopted descriptions below are historical and do not override
this accepted checkpoint. This handoff provides context, not model training.

The user authorized Codex to implement GDN optimization work directly,
including the subsequent fused-key preparation prototype, as an exception to
the earlier local-model-only authoring rule.
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

## Completed follow-up: fused key preparation

Directly approved by the user after the batching rejection. New prototype
qwen_key_prepare.py computes K*beta once and its ordered FP32 product with
caller-supplied eager exp. Separate output layouts preserve original Torch
pointwise dimension order; original two solves, matrices and scan stay frozen.
Status: **not adopted; all 16 accepted engine files restored exactly**.
Authoritative pointer: feedback/prefill-keys-main-run.txt in the controller.

The GPU-only replay diagnostic measured about4.1us versus9.8us for the three
original Torch products. Repeated checks made the public wrapper slower;
a once-validated per-prefill plan reduced eager call cost to about12.3us
versus13.2us for Torch. Native-layout component timings improved, but expanded
whole-model A/B with16samples/backend gave only0.43% at256tokens and0.46% at1024.
One short-prompt timing block regressed; the initial short-prompt trial also
regressed0.26%. The added kernel/plan complexity was not worth adopting.

Both successful real A/B repeats preserve full prefill/next-decode/cache and
2x128generatedIDs/backend/length. The initial real trial failed at generated
token6: its first baseline output was an outlier, while the candidate matched
older accepted benchmarks. A fresh same-priming diagnostic produced6identical
alternating baseline/candidate sequences matching that historical reference;
original-v-original and original-v-candidate127-step teacher pairs were exact.
**The cause remains unresolved.** The baseline ran after candidate warmups, so
this does not rule out an allocation/runtime interaction with the candidate.
Do not discard the failure or claim baseline reliability is fixed.

The A/B harness now checks prompt identity and every generation against
`acceptance/prefill-canonical-generation-v1.json` in the controller, derived
from the already accepted3139a9b benchmark/profile. It rejects drift from either
backend rather than treating the first generation as an unquestioned oracle.
Those golden IDs are for harness validation only: never read/hardcode them in
engine implementation. Existing numerical tests/tolerances were not weakened.

Review also found constexpr strides causing recompilation as conversation
length grows. A final runtime-stride address variant reuses one binary per
layout across128/256/512/1024tokens. It passes101standalone cases,4multi-chunk
plan cases,304direct/44block/20tiny/2gradient/11CPU-state/3broadcast-fallback
checks. It did **not** receive another real-checkpoint performance/quality run;
the design was already rejected for its small gain. Both the real-tested
constant-stride version and this final cache-correct prototype are retained.

No new frozen40-quality or protected-original-model suite was run for the
unadopted kernel. The restored baseline keeps its prior validation through
byte-identical restoration. All containers stopped cleanly, no new swap,
same boot, GPU idle, service inactive, STOP/GPU/LARGE holds retained.
See [fused key preparation](benchmarks/results/2026-11-16-gdn-key-preparation/README.md).
Before more performance work, prioritize investigating generation drift and
reducing repeated checkpoint loading in the validation harness. A cheaper
kernel launcher or broader fusion is an unimplemented option, not a proven gain.
