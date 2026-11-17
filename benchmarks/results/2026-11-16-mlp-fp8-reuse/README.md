# FP8 gate/up preparation reuse

Status: retained and checkpointed as engine `e54057e` after all gates passed.
Baseline was activation-fusion engine da80ee6. All engine implementation is local
Qwen; Codex supervises prompts, reference checks, placement, containment and reporting.

Main run: `/home/cap/spark-inference-overnight-20261113/runs/mlp-fp8-reuse-20261117T031301Z`.

## Scoped change and source ownership

The eight FP8 MLP gate/up pairs receive the same input. New
`qwen_mlp_fp8_pair._make_pair_projector` prepares row scales and FP8 activations
once, then issues the original two separate `torch._scaled_mm` calls with each
projection's own weights/scales. Reduction, conversion, clamping, division,
padding, output dtype, accumulation and reshape arithmetic are unchanged.
The callback AST matches the previously validated local-Qwen prototype task058.

A fresh factory/callback is used per forward. No persistent module metadata,
prepared tensors, request outputs or weights are cached. The factory declines
subclasses, incompatible shapes/devices, registered per-module forward/pre-hooks
and instance-level forward overrides. The caller requires contiguous input and
exact FP8Linear types under the accepted single-request CUDA BF16 inference
prefill eligibility. Normal module calls remain the fallback. NVFP4, decode,
CPU/autograd and accepted activation fusion are preserved.

Local Qwen task062 authored the guarded helper. Task063's contradictory
ordinary-Linear/FP8Linear guard was retained and rejected before placement;
focused task064 supplied the selected forward method. Bodies were copied verbatim,
with only Markdown delimiters removed and method indentation placed in its class.
An AST guard verifies that no other engine definition changed.

## Isolated correctness

All eight real FP8 MLP layers (56–63) passed pair-output and full-block bitwise
checks at lengths 1,15,16,17,256,1024: 48 cases. The dispatcher was observed to
invoke reuse on eligible inputs. Additional checks passed for forward hooks,
pre-hooks, instance forwards, noncontiguous inputs, batching, decode, subclass
and shape rejection, plus three gradient cases. These layer/tiny checks are
distinct from whole-checkpoint generation validation.

## Whole-model prefill A/B

One resident native NVFP4 27B model; identical pinned prompts, 16 timing samples
per backend/length in interleaved ABBA order:

| Prompt tokens | Baseline ms | Reuse ms | Saved ms | Less time |
|---|---:|---:|---:|---:|
| 256 | 252.6239 | 252.6072 | 0.0167 | 0.0066% |
| 1,024 | 992.2916 | 989.1666 | 3.1250 | 0.3149% |

The short case is effectively unchanged; it is not treated as a demonstrated
speedup. Five of eight short blocks and seven of eight long blocks improved.
Variation and regressions are retained. The long-prompt benefit is modest,
without a confidence interval. These are incremental warm-prefill measurements
above activation fusion, not sustained output-token throughput. Do not add the
percentage to earlier experiments.

Full prefill logits/all cache tensors and next-decode logits/caches were
bitwise exact at both lengths. All eight 128-token replies matched pinned
accepted IDs; captured logits were finite and GPU/CPU argmax agreed. Diagnostic
copies occur after argmax only in correctness generation, outside primary timing.
The historical outlier did not recur; its cause remains unresolved.

## Quality and profiling

Frozen quality passed 40/40 with no truncation. All 40 generated answers and all
446 token losses were element-exact against accepted activation-fusion engine
da80ee6; perplexity remained 8.946730084578427. The frozen vLLM-reference screening
comparison also passed. This is a small fixed suite, not broad language-quality
certification.

Profiling confirms all eight pairs execute shared preparation. The pair scopes
contain 72 launches (one shared preparation plus two GEMMs per pair). MLP launches
fall 1,096→1,040 and total prefill launches fall by 56 at both prompt lengths.
FP8 GEMMs remain 233, NVFP4 GEMMs 168; GDN state/key, normalization and 64 fused
MLP activation launches are unchanged. Profiles generate one token with zero
decode calls. The paired uninstrumented A/B above is the performance measurement;
historical separate instrumented profiles are used to verify operation counts.

The protected small-Qwen group and repository generation-budget tests passed.
All 56 Qwen and 15 Llama protected gates passed, including original real 9B
Safetensors and 27B GGUF numerical/conversation checks, with one checkpoint per
fresh process. Frozen fixture hashes and numerical tolerances were unchanged.
Median peak allocated memory was unchanged within A/B: 23,240,434,176 bytes at
256 tokens and 24,473,309,696 bytes at 1,024 tokens.

All four experiment containers stopped with exit0/OOMKilled false. GPU compute
inventory empty, controller service MainPID0, same boot throughout, zero new
swap, telemetry above the32GiBhost floor. STOP/GPU/LARGE holds retained.

Engine checkpoint: `e54057e09c14aa8e39652b780a1725ac64057db4`. All other engine files match the baseline.
Local-model drafts, rejection/provenance, compact results and harness sources
are archived beside this report. Raw logits/profiler traces remain in experiment
run directories rather than Git. The historical outlier remains unresolved;
new A/B mismatch capture and the earlier commit/source peelback plan still apply.
