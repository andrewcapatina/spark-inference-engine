# MLP prefill: compare options, then fuse the activation

Status: retained and checkpointed as engine `da80ee6`. All numerical,
quality, profiling and original-checkpoint protected gates passed. Baseline was
accepted engine f1e760f.

Main run: `/home/cap/spark-inference-overnight-20261113/runs/mlp-prefill-study-20261116T175739Z`.
The user authorized comparing input-preparation reuse with activation fusion
before choosing a change. All implementation was authored by local Qwen. Codex
wrote the harness, reference checks, prompts, source-placement guards and reports.
Rejected raw drafts and responses are retained; Python bodies were placed verbatim,
with only Markdown delimiters removed.

## Comparison and selection

CPU-only header/scalar inspection found equal input-global scales in all 56
NVFP4 gate/up pairs. The other eight pairs use FP8 preparation independent of their
weight scales. This audit establishes eligibility, not whole-model correctness.

Separate local-Qwen prototypes were tested on real layer 0 (NVFP4) and layer 56
(FP8), with hidden width 5,120 and intermediate width 17,408. Two separate GEMMs
were preserved by the reuse prototype; only input preparation was shared.
The activation prototype preserves the BF16 rounding of SiLU before multiplication.

All 65,536 BF16 gate bit patterns passed with nine up multipliers, including
extreme/subnormal values and signed zero. Finite values were bitwise exact;
nonfinite masks matched. Additional activation shapes, 12 real-weight layer
cases, repeated pair calls and CPU/GPU/autograd fallback checks passed.

Isolated full-MLP median savings in milliseconds (16 samples per backend):

| Backend | Rows | Preparation reuse only | Activation only | Both prototypes |
|---|---:|---:|---:|---:|
| NVFP4 | 256 | 0.0096 | 0.0632 | 0.0618 |
| NVFP4 | 1,024 | 0.0131 | 0.3143 | 0.3696 |
| FP8 | 256 | 0.0640 | 0.0319 | 0.1325 |
| FP8 | 1,024 | 0.5735 | 0.6708 | 0.8618 |

Standalone activation-only timing at 256 rows was effectively unchanged;
at 1,024 rows it fell 0.8062→0.4939 ms. These different contexts are retained
rather than selecting only favorable component measurements. Factory validation
was excluded from steady reuse timing: about 0.0458 ms for NVFP4 and 0.0151 ms
for FP8 in this test. Reuse integration would need correct metadata lifetime and
invalidation handling to realize those gains.

Activation was chosen first for larger gains in the dominant NVFP4 block and a
simpler integration without persistent module caches. Preparation reuse remains
prototype-only. Isolated layer gains are not whole-model speedups.

## Scoped engine change

New `qwen_mlp_activation.silu_mul` computes SiLU with FP32 CUDA libdevice exp and
direct division, rounds to BF16, then multiplies up and stores BF16. FP fusion is
explicitly disabled. Fresh output; no input/weight/cache mutation. Other dtypes,
layouts, devices and autograd contexts retain the original Torch expression.

`QwenMLP.forward` selects it only for single-request CUDA BF16 inference prefill
with at least 16 tokens. Original expression and evaluation order remain the
fallback. Missing Triton alone permits fallback; unrelated import and runtime
errors propagate. The integration passed 48 dispatch/output cases, two gradient
cases, two missing-Triton cases, two unrelated-import cases and a runtime-error
propagation check. Every other engine definition remains unchanged.

Raw drafts 055 and 056 were rejected before execution for API/math-order errors;
060 was rejected for message-based exception matching. Focused local-model tasks
057/058/059 repaired standalone candidates, and 061 supplied the accepted method
candidate. Selected engine code is exclusively local tasks 059 and 061.

## Native 27B A/B

One resident model, identical pinned prompts, 16 samples per backend/length:

| Prompt tokens | Baseline prefill ms | Candidate ms | Saved ms | Less time |
|---|---:|---:|---:|---:|
| 256 | 256.0387 | 251.1717 | 4.8670 | 1.9009% |
| 1,024 | 1,002.5413 | 985.9550 | 16.5863 | 1.6544% |

All eight ABBA blocks improved at each length. These warm direct-prefill results
have no confidence interval or sustained output-token throughput claim. Do not
add their percentages to earlier optimization results.

Full prefill logits and every cache tensor, plus next-decode logits/caches,
were element-exact at both lengths. All eight 128-token replies matched pinned
accepted IDs with finite captured logits and GPU/CPU argmax agreement. The
historical generation outlier did not recur; its cause remains unresolved.
Diagnostic copies occur after argmax only during correctness generation, not in
the primary timed prefill measurements.

## Quality, profile and portability

Frozen quality passed 40/40 with no truncation. All generated IDs and all 446
token losses were exactly unchanged from accepted engine f1e760f. Perplexity
remained 8.946730084578427; the frozen vLLM-reference comparison screen passed.
This fixed small suite is not a general language-quality certification.

At both prompt lengths, MLP kernel launches fell 1,160→1,096: 128 separate
SiLU/multiply launches became 64 fused launches. Total prefill launches also fell
by 64. GDN state/key preparation, normalization and FP8/NVFP4 GEMM counts were
unchanged. Profiles generated one token and made zero decode calls. Historical
profile timings are separate instrumented runs; use the paired uninstrumented
A/B above for the speed claim.

The original-model Torch2.11/Triton runtime also passed all65,536BF16 patterns
with all nine multipliers, before any original large checkpoint was loaded.
The generation-budget repository tests and protected small-Qwen group passed.
All 56 Qwen and 15 Llama protected gates passed, including original 9B
Safetensors and 27B GGUF numerical/conversation checks, with one checkpoint per
fresh process. Acceptance source hashes and numerical tolerances were unchanged.
Median peak allocated memory was unchanged: 23,239,909,888 bytes at 256 tokens
and 24,465,904,128 bytes at 1,024 tokens.

All four experiment containers stopped with exit0/OOMKilled false, GPU compute
inventory empty, controller service MainPID0. Same boot throughout, zero new
swap; telemetry stayed above the 32GiB host floor. STOP/GPU/LARGE holds retained.

Engine checkpoint: `da80ee6a80775b42257a71b0bc1ae5178e9fa378`. All other engine files match the baseline.
Local-model drafts/provenance, compact evidence and harness sources accompany
this report; bulk logits/profiler traces remain in experiment run directories.
