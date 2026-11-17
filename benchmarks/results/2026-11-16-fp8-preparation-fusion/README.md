# Fused FP8 activation preparation

Status: retained after strict primitive, real-weight, whole-checkpoint, quality and protected validation. Engine commit: 48d4cc758027f6f5c84d5fd6afa444d4968cdeae.

Main run: `/home/cap/spark-inference-overnight-20261113/runs/fp8-prep-fusion-20261117T061723Z`. Baseline engine `e54057e`, repository checkpoint `275f23f`. Local Qwen wrote every engine/kernel change; Codex handled harnesses, references, prompts, placement and reporting.

## Scoped change

New `qwen_fp8_prepare.prepare_fp8` fuses per-row maximum/scale, division, FP8 conversion and zero padding into one Triton launch. It targets contiguous CUDA BF16 inference inputs with at least 16 rows and width 5,120 or 6,144. The original preparation remains the fallback for other widths/modes/layouts; decode remains on its original path. `FP8Linear.forward` and the accepted shared gate/up projector call the helper. Original separate GEMMs, weights/scales, BF16 output and accumulation flags are unchanged. Fresh outputs are owned by each invocation; no persistent prepared-tensor cache.

The checkpoint has 233 FP8 projections: 161 use width 5,120, 64 use width 6,144, and 8 use width 17,408. Eight shared pairs reduce 233 preparations to 225 groups: 217 eligible fused groups and 8 wide fallbacks. NVFP4 preparation and all state-update math are unchanged.

Reference probing found that eager CUDA FP32 scalar `/448` exactly follows rounded reciprocal multiplication; 9,878 of 32,640 finite BF16 row maxima differ from true RN division. The kernel therefore preserves that multiply, the FP32 clamp and RN division for `x/scale`, with contraction disabled. Explicit NaN propagation preserves the reference scale and FP8 byte bits.

## Warm whole-model prefill

One resident actual 27B NVFP4 checkpoint, frozen 256/1,024-token prompts, accepted baseline versus candidate selected in ABBA order: 16 samples/backend/length, 8 blocks. Both FP8Linear.forward and the shared-pair factory are restored for the baseline. Loading and generation are excluded from primary timings. No confidence interval or output-token speed claim.

| Prompt tokens | Baseline CUDA ms | Fused CUDA ms | Saved ms | Less time | Prompt tok/s before → after | Positive blocks |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 250.906 | 241.712 | 9.195 | 3.66% | 1020.3 → 1059.1 | 8/8 |
| 1024 | 985.020 | 931.077 | 53.943 | 5.48% | 1039.6 → 1099.8 | 8/8 |

Wall timings, all ABBA blocks and median peak allocations are retained in `paired-prefill-audit.json`. Compare these incremental figures only with this run’s accepted baseline; do not add percentages from earlier optimizations.

## Correctness and profile evidence

- Strict preparation prototype: 359 cases; all 32,640 finite nonnegative BF16 maxima at width 5,120, random/boundary/negative-zero/nonfinite cases at both widths, padded rows and raw FP32-scale/FP8-byte comparisons. All pass under both native Torch2.13 and original Torch2.11 runtimes. No tolerance changes.
- Real weights:48 full MLP cases covering all 8 FP8 MLPs at 1/15/16/17/256/1,024 rows; 6 actual 6,144-wide projection cases; import, runtime-error and metadata/mode fallbacks passed.
- Whole actual 27B: entire prefill and next-decode logits/all caches exact; 8 canonical 128-token replies (two/backend/length) exact. Every traced GPU argmax matches CPU argmax with finite logits.
- Frozen quality: 40 generated answers and 446 token NLLs exactly match the accepted native engine; perplexity 8.946730084578427. Existing vLLM screen also passes.
- All 56 Qwen and 15 Llama protected gates pass, including original real 9B Safetensors and 27B GGUF numerical/conversation checks, one checkpoint per fresh process. Fixtures/tolerances unchanged.
- Prefill-only profile selects one output token and executes no decode calls. Exactly 217 fused preparation kernels replace 1,519 launches, removing 1,302 launches/prefill. Both 233 FP8 and 168 NVFP4 GEMMs and activation/norm/GDN state/key kernel counts remain unchanged. Shared-pair scope 72→24 kernels; total MLP scope 1,040→992. Timed prefill measurements above exclude instrumentation.

## Rejections and limitations

Local 065 was rejected before execution for unsafe padded-row loads; 066 passed finite cases but failed exact NaN propagation; 067 used unavailable Triton APIs and failed compilation. Local 068 fixed those issues and passed. Local 070 used a literal optional-backend submodule guard and was rejected before placement; 071 repaired only that guard. Raw responses, prompts and reviews are retained. No numerical checks were weakened.

Historical generation outlier did not recur. Its root cause remains unknown; candidate warmup/allocation interaction was never ruled out. Existing canonical fixtures, failing-logit capture and source-snapshot peelback plan remain in force. This work is not evidence of arbitrary-Hugging-Face-model support or production/long-context validation.

## Ownership and containment

Kernel 068, FP8Linear.forward 069 and pair factory 071 were placed verbatim with only surrounding class indentation. AST checks preserve all original guards, fallback preparation and GEMMs. No supervisor implementation changes. All other 17 original engine files are unchanged. `placement.json`, raw local responses and the 20-file manifest identify exact sources.

All 4 experiment containers stopped with exit 0/OOM false; GPU compute empty; service MainPID0; same boot; STOP/GPU/LARGE holds retained. Host floor 32 GiB, per-worker new-swap ceiling 256 MiB, one large model only; 32 GiB native/8 GiB primitive/56 GiB protected allocator caps. Actual telemetry is in `cleanup-check.json`.

Compact results and harness sources are included here; full traces, tensors, compiler caches and weights remain outside Git. `stage-index.json` links diagnostic, real-checkpoint and rejected runs.
