# Gated normalization: prefill pointwise fusion

Engine **ea70e47**, authored by local Qwen tasks037–040. Previous accepted checkpoint `checkpoint-prefill-norm-20261115` protects engine dd139d0/report56babd9. The supervisor wrote prompts, harness, diagnostics and documentation; all implementation is retained verbatim from reviewed local-model responses, with mechanical method indentation only.

## Warm prefill result

One resident native 27B NVFP4 model alternates baseline/candidate/candidate/baseline twice, four warm samples per method and prompt length. Direct CUDA-event `model.prefill` latency excludes loading, tokenization, imports and compilation. Generation is correctness validation only.

| Prompt tokens | Baseline → accepted prefill | Less processing time | Prompt tokens/sec baseline → accepted |
| --- | --- | --- | --- |
| 256 | 265.07 → 260.17 ms | 1.85% | 965.8 → 984.0 |
| 1024 | 1048.58 → 1017.40 ms | 2.97% | 976.6 → 1006.5 |

Both ABBA blocks, wall-time medians and per-request peak allocations are retained in `paired-prefill-audit.json` and `ab-results.json`. This is a repeated small-workload comparison, not a confidence interval or an all-context/hardware claim. The baseline already includes accepted QKV, FP8 preparation and ordinary normalization improvements; percentages from different experiments must not be added together.

## Exact arithmetic and explicit dispatch

Original `gated_rms_norm` stays unchanged. A new dedicated prefill entry point keeps original eager FP32 conversion, square/mean reduction, reciprocal square root and PyTorch FP32 SiLU. One Triton pointwise kernel then multiplies x by inverse, rounds to BF16, multiplies by the BF16 weight, rounds again to BF16, widens that result to FP32 for the supplied SiLU product and casts the output to BF16. Both intermediate BF16 rounds are mandatory and explicitly tested. Floating-point fusion is disabled; no transcendental/reduction approximation is introduced.

Gated normalization sees flattened rows and cannot infer token phase reliably. `QwenRMSNorm.forward` therefore adds a keyword-only `prefill=False`; only `QwenGatedDeltaNet.prefill`, for sequence lengths at least16, requests the new path. Default forward, short prompts and decode keep their original gated path. The module call and hooks remain intact. The helper checks same-device CUDA contiguous FP32 inputs/inverse/SiLU, BF16 weights, matching shape/width and bounded indexing; the entry point falls back for unsupported metadata/dtypes, CPU and outside inference mode. Missing Triton imports alone fall back; unrelated import errors and helper execution errors propagate.

Median measured peak allocated memory is unchanged at21.644GiB (256tokens) and22.775GiB (1,024tokens); no memory-saving claim is made.

Standalone fresh-cache compilation is recorded separately, including initial process compiler startup. In the full A/B worker, baseline ordinary normalization already initializes Triton before the first gated candidate; those first-request timings are not a controlled cold-start comparison. All reported gains are warm.

## Matched prefill-only profile

Each profile produces exactly one output token and asserts zero decode calls. Baseline is the immediately preceding accepted ordinary-normalization profile, with identical prompt IDs and input-shape observations. Inclusive scopes overlap; do not add parent/child costs.

| Prompt tokens | Gated normalization GPU time baseline → accepted | Gated kernels baseline → accepted |
| --- | --- | --- |
| 256 | 12.05 → 7.79 ms | 576 → 384 |
| 1024 | 78.12 → 51.41 ms | 576 → 384 |

There are48 new gated-tail calls/kernels and192 fewer gated-normalization kernels per prompt. Ordinary normalization remains966 kernels, FP8 GEMMs233, and GDN state-kernel counts are unchanged. All CUDA kernels are attributed. Profile times include profiling overhead and are separate from unprofiled paired latency. Cumulative allocations are not peak memory or measured bandwidth.

## Correctness evidence

- 205 standalone bitwise cases include all65,280 finite BF16 patterns in x and gate inputs, signed zeros, partial blocks, real128-wide shapes up to49,152rows, noncontiguous source conversions, parameter ownership, invalid metadata, inference-mode and empty-input checks. Original eager inverse/SiLU are captured observationally; the harness does not substitute alternate engine arithmetic.
- 41 primitive integration cases, two exact gradient cases including gate gradients, and20 tiny mixed-quantized models pass. Tiny prefill/next-decode logits and all cache tensors, full forward and16generatedIDs match exactly. Actual routing is checked: no new gated helper calls in default forward/decode, threshold15/16 respected, module keyword selects the dedicated entry point.
- Real27B at256/1,024 prompt tokens preserves entire prefill logits/every cache tensor, next-decode logits/cache, and all128generatedIDs exactly. Prefill observes48 new helper calls; single-token decode observes zero.
- Frozen40quality output sequences and all446corpus token losses are exact against the preceding accepted native checkpoint; perplexity stays8.946730. The frozen vLLM-reference quality screen also passes unchanged. This is not expanded independent full-logit quantized-model parity coverage.
- All56 Qwen and15Llama acceptance gates pass with unchanged fixture-source hashes and tolerances, including original9B/27B numeric and conversation processes. These original-runtime checks are distinct from NVFP4 performance measurements.

## Provenance and runtime containment

Task036 was retained/rejected before placement for a missing width guard and runtime width specialization;037 is the corrected standalone helper,038 the guarded entry,039 the explicit module flag and040 the prefill call. All earlier accepted behavior is preserved; previously rejected compiler/GDN/cache candidates stay excluded. No numerical tests were weakened.

Protected tests use the executable project-mounted compiler cache established during the preceding ordinary-normalization experiment. Its read-only-home/noexec-tmp recovery is documented there; this experiment does not silently treat those old failures as new passing tests. All large models run serially. Exact image IDs, memory telemetry, swap/boot/service/container state are archived. Final four containers stopped with exit0/OOMfalse, no GPU compute process, serviceMainPID0, STOP and holds retained; see `cleanup-check.json` for measured swap growth.

Model `unsloth/Qwen3.8-27B-NVFP4`, revision `f0b7c9e722f5565102fff8481c99e4d86ae099c7`. Installed Torch2.13+cu130/Triton3.7.1 for the experiment and original Torch2.11 runtime for protected checks. Main run /home/cap/spark-inference-overnight-20261113/runs/prefill-gated-normalization-20261116T014607Z. Weights, compiler binaries and large raw traces stay outside Git.

Reproduce using the existing installed experiment/runtime, with all other workers stopped:

```bash
python3 /home/cap/spark-inference-overnight-20261113/prepare_prefill_gated_trial.py ab
python3 /home/cap/spark-inference-overnight-20261113/launch_prefill_gated_prepared.py ab
```

Long contexts, other architectures, arbitrary devices, nonfinite-input equivalence and production reliability remain outside this validation. Next options should be selected from the resulting profile rather than extending fusion scope without measurement.
