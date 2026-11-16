# GDN key preparation: investigation and readdition

Status: retained and checkpointed as engine `f1e760f`. All numerical gates
passed. The historical outlier's root cause remains unresolved.

The user authorized direct Codex investigation and restoration of the retained
runtime-stride candidate, accepting small correct gains. Numerical fixtures and
tolerances remain unchanged. Main run:
`/home/cap/spark-inference-overnight-20261113/runs/gdn-outlier-readd-20261116T145716Z`.
The [earlier trial report](../2026-11-16-gdn-key-preparation/README.md) is historical.

## Outlier investigation

The first constant-stride prototype trial produced an outlying baseline reply:
token index 5 was 78326 instead of accepted token 9989. Candidate warmups preceded
that baseline generation, so this did not independently establish a baseline bug.
The failing logits were not captured. Repeated success does not identify or repair
the original cause.

This investigation physically removed the candidate for its baseline diagnostics:

- 18 native 27B replies across unprimed, prefill-primed and allocator-perturbed
  phases matched all 128 accepted IDs.
- Baseline A/A repeated A/B-style priming at 256/1,024 tokens. Entire prefill and
  next-decode logits/caches matched, as did eight additional 128-token replies.
- Four further replies matched all 128 IDs. Every step's 248,320 logits was
  bitwise identical across repeats and finite; GPU and CPU argmax agreed.
  Diagnostic CPU copies add synchronization and can mask timing-dependent faults.
- 1,728 fixed-input real-weight BF16 gate, FP8 head and NVFP4 MLP projection
  repeats were bitwise identical. All 6,144 large-vocabulary tied-argmax cases
  selected the correct index.
- Installed CUDA initcheck reported FP8 output-copy reads, then made no further
  progress for about three minutes. The supervisor terminated only that target;
  the controller stopped its container. Its log and exit 71 are retained.
  This is an inconclusive diagnostic, not a passing test or a confirmed engine bug.
- Three FP8 head output-buffer tests initialized the destination with NaN,
  +123 or -123. Every padded output value was overwritten, all outputs were
  finite, and the valid row matched the normal path exactly. This test found
  no unwritten output values.

NVIDIA documents possible initcheck false positives for unsupported tensormap and
TCGEN05 global-memory instructions in its [known limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations).
That limitation alone does not prove these warnings false. No speculative engine
repair was applied.

At the historical divergence step, the current canonical run's best logit was
26.75, the runner-up 19.375, and historical wrong token 78326 scored -7.25. The
current run has no near tie; this cannot establish what happened in the old run.

## Restored implementation

Only `spark_engine/qwen_ops.py` and the new `qwen_key_prepare.py` were restored
from the hash-verified retained runtime-stride tree. Every other engine file
matches accepted engine `3139a9b`.

A Triton kernel computes FP32 `weighted = K * beta`, then the separately rounded
FP32 `decayed = weighted * eager_exp`. It returns separate outputs in the original
pointwise layouts. Original eager exponentials, right-hand sides, two triangular
solves, GEMMs and sequential carried-state arithmetic remain intact. Metadata is
validated once per prefill; each chunk gets fresh output buffers. Runtime strides
avoid prompt-length-specific kernel compilation. One key binary served both
actual-model prompt lengths.

## Numerical validation

Component tests passed 101 standalone cases, four multi-chunk plans, 304 direct
cases, 44 model blocks, 20 tiny models, two gradient cases, 11 CPU/state cases and
three broadcast fallbacks. Missing-Triton fallback and error propagation passed.
These are component/tiny-model checks, distinct from real-checkpoint validation.

One resident native NVFP4 27B model passed element-exact full prefill logits and
all cache tensors, plus next-decode logits and all caches, at both prompt lengths.
All eight 128-token replies (two per backend/length) matched pinned accepted IDs.
Captured logits were finite and GPU/CPU argmax agreed. Diagnostic GPU copies were
inserted after argmax only during correctness generation, without per-token CPU
synchronization. They do not affect the primary timed prefill measurements.

The frozen quality suite passed 40/40 with no truncation. Generated IDs and all
446 token losses were exactly unchanged from accepted `3139a9b`; perplexity was
8.946730084578427. The frozen vLLM-reference comparison screen also passed. This
small screening suite is not a general language-quality certification.

All 56 Qwen and 15 Llama protected gates passed, including original 9B Safetensors
and 27B GGUF numerical and conversation checks. Large checkpoints ran in separate
fresh processes. The acceptance sources match their frozen hashes.

## Measured prefill performance

Uninstrumented warm direct prefill, 16 samples per backend/length, paired ABBA:

| Prompt tokens | Baseline ms | Fused keys ms | Saved ms | Less time |
|---|---:|---:|---:|---:|
| 256 | 260.4538 | 257.9591 | 2.4947 | 0.9578% |
| 1,024 | 1,011.0956 | 1,002.8385 | 8.2570 | 0.8166% |

Seven of eight blocks improved at each length; one regressed at each length.
Block variation is retained. Per the user's preference, no all-positive timing
filter overrides the exact numerical gates. These are small measured gains,
without a confidence interval, sustained generation-throughput claim or evidence
for longer prompts. Do not add these percentages to earlier experiment gains.

Median peak allocated bytes were unchanged: 23,239,909,888 at 256 tokens and
24,465,904,128 at 1,024 tokens.

Prefill-only profiles generated one token and made zero decode calls. GDN kernel
counts fell 8,448→8,064 at 256 tokens (-384), and 30,912→29,376 at 1,024 (-1,536),
with 192/768 fused-key launches. Normalization, FP8 and NVFP4 GEMM counts stayed
unchanged. Inclusive instrumented GDN GPU time was 83.61/345.14 ms versus
76.26/340.73 ms in the historical baseline profile. These separate instrumented
runs are not a paired speed comparison; no isolated GDN timing gain is claimed.
The primary speed result is the uninstrumented warm A/B above.

## Historical replay, containment and checkpoint

The final isolated replay of the exact retained constant-stride source and original
16-prefill warmup sequence passed full prefill/next-decode outputs and caches,
plus all four uninstrumented 128-token replies against accepted IDs. Its timing is
diagnostic only and is not used as current-engine performance evidence. No old
prototype was restored into the active engine.

Across this investigation, 30 candidate-absent baseline replies, eight current A/B
replies and four historical-source replies were canonical. The source of the old
failure remains unknown; it cannot be labeled fixed or attributed to the baseline.
New A/B correctness generations retain logits on nonfinite values, GPU/CPU argmax
disagreement or accepted-ID divergence. If it recurs, use that first failing trace
to locate the earliest differing operation and peel back source snapshots. The
user explicitly suggested commit peelback. `commit-peelback-plan.json` records
accepted revisions from 3139a9b back through gated/ordinary normalization, FP8
preparation, KV/QKV reuse and initial native quantized dispatch. The earlier bad
prototypes were uncommitted, so Git commits alone do not cover all relevant states.
There is currently no reproducible bad endpoint for a meaningful bisect.

All four experiment containers stopped with exit 0 and OOMKilled false; GPU compute
inventory is empty and the user service has MainPID 0. Every stage stayed on the
same boot; all telemetry stayed above the 32 GiB host floor with zero new swap.
STOP, GPU_WORKLOAD_HOLD and LARGE_WORKLOAD_HOLD remain in place. The inconclusive
sanitizer run remains recorded as failed, not silently converted to a pass.

Engine checkpoint: `f1e760f9d4ba6f5d0db72513f1f2fbab5b049f01`. Only the two intended engine files
changed. Diagnostic/verification harness sources and compact evidence are archived
beside this report. Bulk logits, profiler traces and the complete sanitizer log
remain in the experiment run directories rather than Git.
