# Independent layer arithmetic and HF graph isolation

This follows the [teacher-forcing discrepancy](../2026-11-14-nvfp4-teacher-forcing/README.md). All engine implementation remains local-Qwen-authored and byte-identical to the preceding benchmark. Only diagnostic fixtures, containment and documentation changed. No numerical tolerance was relaxed.

## Established at this checkpoint

On the actual downloaded27B checkpoint and a273-token prefix (original256 prompt plus17 teacher tokens), Hugging Face's official graph **with the same native projection operators** matches the native engine's entire248,320-element normalized vocabulary-score vector bit for bit. All64 recorded last-token block outputs match in both local and streamed graph comparisons. All64 isolated cached-block replays also match the native cached outputs exactly. This validates the tested graph/cache behavior under matching primitive arithmetic; it is not independent primitive validation, every-token full-hidden-tensor equality, or broad model-quality evidence.

Separately, independent compressed-tensors activation QDQ plus dequantized FP32 matrix multiplication passed8 tiny cases at the existing rtol0.01/atol0.03125. Real projection probes in blocks1 and43 reproduce the original native captured block outputs exactly and have0 output elements beyond those same bounds over14 invocations (13distinct projections; GDN qkv is called twice). The FP4 codes and local scales match exactly on all6 tested NVFP4 calls; FP8 activation QDQ matches exactly. FP8 fast and regular accumulation return identical BF16 outputs on all8 sampled calls. This provides no evidence that changing the fast-accumulation flag would fix the discrepancy.

## Independent arithmetic still differs end to end

The fully independent numerical path uses the installed compressed-tensors recipe to quantize activations, explicitly unpacks FP4/FP8 weights, performs FP32 dense GEMMs, then casts projection outputs to BF16. Official HF5.19.0 plain-Torch GDN chunk/recurrent and convolution fallbacks and eager attention supply the architecture. Ordinary operations retain their architecture precision; this is not a full-FP32 unquantized model. Weights stream one decoder block at a time; there is no second whole-model allocation. Embedding rows match checkpoint values exactly.

At the examined prediction:

| Path | Preferred token ID | Probability |
| --- | ---: | ---: |
| Native cached |449|99.89%|
| Native fresh prefill |449|99.70%|
| Official HF graph, native projection primitives |449|99.70%|
| Official HF graph, independent QDQ/FP32 projections |264|55.10%|

The independent path assigns42.91% to449. Its final score array reproduced bit for bit in a separate process. Total variation against native fresh-prefill scores is0.56794, so this is a substantial distribution discrepancy, not merely a tied greedy choice. The earlier vLLM full-prefix prompt-scoring path also preferred264 at this teacher position, but this does not establish it as mathematical ground truth.

When each independent block starts from native inputs, median last-token relative L2 error is0.1724%, maximum1.6435% at block43. Isolated independent cache replay has median0.2095%, maximum2.4328% at block51. Propagating the reference's own states across all64 blocks produces final hidden relative L2 error77.75%. These block metrics are diagnostics, not pass thresholds fitted afterward.

The shared-operator graph control removes the final-score discrepancy completely. Together with the projection probes, this isolates the observed sensitivity to the numerical projection path and its interaction with subsequent quantization, rather than identifying a graph/cache wiring error. The sampled probes show tiny GEMM rounding differences with matching activation codes; this is consistent with quantization amplifying arithmetic differences. It does not prove that every projection is correct, that all differences are harmless, or that one runtime has better task accuracy.

## Validation boundaries

- Tiny primitive tests, real projection checks, real one-prefix graph/cache validation, and broader model quality are separate milestones.
- The independent mathematical reference remains discrepant; no full real-checkpoint numerical-parity claim is made.
- Only two blocks received isolated real-projection checks. All64 blocks received local/streamed/reference-cache diagnostics, using last-token output vectors.
- Reference teacher forcing in the prior run used prompt scoring; this run independently replays each block's captured inputs through its cache. These are different comparisons.
- The graph control deliberately reuses local-Qwen projection modules. Fields prefixed independent in its raw JSON reuse the original metric schema; its explicit scope identifies shared primitives. Its completion log marker likewise comes from the common diagnostic harness and is not a primitive-independence claim.
- Native graph-control head projection uses all273 positions to match native prefill GEMM shape. The independent FP32 path projects only the selected last position; this is another projection arithmetic/shape difference, not a timing comparison.
- Existing56Qwen/15Llama regressions remain the prior passed engine checkpoint. They were not rerun for this harness-only work; engine source hashes were verified unchanged.

## Recovery and containment

The first independent reference completed64 blocks and saved the final score array, then failed its cleanup guard because the fixture had not released idle cuBLAS workspaces. Its failed worker/parent snapshots and source remain retained. A repeat with explicit workspace release produced the identical score array and allocated/reserved0/0, without changing original16MiB/64MiB cleanup bounds. The exact residue in the first attempt was not durably recorded. Do not count that failed attempt as a successful cleanup run.

Native capture, both real-block probes, corrected independent reference and graph control all exited normally with stopped containers, PID0/OOMfalse, same boot, no new swap, and host available memory recovered within4GiB. Native capture, probes and corrected references released tracked CUDA allocation/reservation to0/0. The model server was stopped throughout; only one whole checkpoint was loaded, and reference stages held one block at a time. Independent reference peak CUDA allocation9.51GiB; graph control1.40GiB. These are diagnostic memory measurements, not throughput benchmarks.

All four experiment containers were stopped afterward and the legacy systemd controller remained inactive. Global holds were retained. No new packages, images, model downloads or host changes were required. No engine patch was justified or authored during this investigation.

## Reproducibility and next work

Stage-status JSON contains exact commands, boot and engine hashes. Source snapshots, input IDs, compact metrics and raw result summaries are included here. Full captured tensors and vocabulary arrays remain local under /home/cap/spark-inference-overnight-20261113/feedback/nvfp4-reference; local-capture-manifest.json records their hashes and sizes. They are excluded from Git along with weights. The independent source before adding the graph-control branch is preserved separately. Reference package sources are pinned by the runtime image and recorded source hashes.

The next useful accuracy work is a representative, fixed task/perplexity evaluation against a declared checkpoint/runtime reference, with an explicit acceptable degradation criterion. Numerical identity across different low-precision kernels should not substitute for task accuracy, and operator bounds alone do not prove it. Additional focused tests should target block51 cache sensitivity and a broader range of real activations if a concrete numerical failure appears. Further performance changes should wait for that quality baseline.
