# Independent GDN chunk batching: not adopted

The user explicitly authorized Codex to implement this work directly, as an exception to the earlier local-model-only rule. No local-Qwen authorship is claimed. All 16 accepted engine source files were restored byte for byte to engine **3139a9b** (report f8df2cf) after retaining the full candidate, earlier revisions and patch. No engine optimization from this experiment is active.

## What was implemented

Independent cumulative decays, masked pairwise decays, query/key systems and two original lower-unit triangular solves were prepared in bounded chunk groups. The original FP32 carried-state scan remained sequential, including query scaling, corrections, output and state propagation expressions. A conservative workspace estimate limited preparation to 256 MiB and at most eight chunks; an isolated sweep also tested a 16-chunk/512 MiB estimate. These are planning bounds, not measured process-memory caps; workers also kept the existing allocator/host/swap guards.

The first batch layout put heads before chunks and triggered repeated implicit materialization. A second layout explicitly packed chunk-major input buffers, making each prepared chunk compact for its later scan. Both layouts and the frozen baseline are retained.

## Correctness and rejected variants

Broad batching changed cuBLAS triangular-solve rounding for narrow RHS widths. A diagnostic compared actual intermediate values: cumsums, pairwise decays, RHS products, systems and intra-chunk matrices matched; solver outputs differed by small FP32 amounts, enough to change two BF16 output elements in one direct case. Width-only restriction still failed a three-head dense block's carried state (~1.49e-8 max absolute difference). Those failures remain failures; no tolerance was changed.

The final specialization was BF16 CUDA inference, one request, 48 value heads, key/value width128 and chunk size64, matching the target native27B. Other modes/layouts kept original per-chunk execution. This specialization passed **304 direct cases,44 dense/quantized blocks,20 tiny models,2 gradient cases and11 CPU/provided-state-dtype cases**, with exact outputs, FP32 state, input ownership and continuation. Twelve additional block cases cover native48-head convolution layout and batch2 fallback. The batch-size sweep also requires exact output/state for every tested size.

These are component/synthetic/tiny-model milestones, **not real-checkpoint validation**. No real27B load, generation, quality, protected-original-model suite or whole-model throughput trial was run for this rejected candidate, because its small-tensor performance gate already failed. Restored baseline3139a9b retains its previously completed real-checkpoint gates through byte-identical source restoration.

## Performance gate

Actual tensor layouts were captured from a synthetic native48-head GDN block's convolution/projection pipeline. The table measures the entire GDN chunk function, not just the batched helper and not full-model prefill. Each method has eight warm samples in a symmetric ordering, with loading and projection excluded. Positive changes mean slower. Group sizes are tested at256 and1,024tokens; this is not a confidence interval or hardware-general result.

| Tokens | Method | Baseline ms | Candidate ms | Time change |
| --- | --- | --- | --- | --- |
| 256 | group2 | 1.882 | 1.806 | -4.0% |
| 256 | group4 | 1.882 | 1.857 | -1.3% |
| 256 | group8 | 1.882 | 1.854 | -1.5% |
| 256 | group16 | 1.882 | 1.857 | -1.3% |
| 1024 | group2 | 7.338 | 7.839 | +6.8% |
| 1024 | group4 | 7.338 | 8.287 | +12.9% |
| 1024 | group8 | 7.338 | 8.853 | +20.7% |
| 1024 | group16 | 7.338 | 9.143 | +24.6% |

Every grouped variant is slower at1,024tokens, with regression increasing as the group grows. The final group8 repeat measured7.326→8.762ms at1,024tokens. Profiler-attributed copy operator GPU time was 1.365→2.570ms (parent/child scopes must not be summed); pointwise and matrix work also increased. Fewer solve submissions did not compensate for packing, changed access patterns and additional materialization. The microbenchmark does not justify extrapolating a whole-engine speedup or slowdown percentage.

One profiling harness attempt failed while formatting an absent initial_state as a tensor; that metadata bug was fixed and rerun, separately from implementation numerical failures. Failed trial statuses/logs and successful reruns are archived by run name. No numerical test was weakened.

## Retention, handoff and cleanup

Main run /home/cap/spark-inference-overnight-20261113/runs/prefill-gdn-batch-20261116T071709Z. Full16-file rejected tree: /home/cap/Projects/spark-inference-engine/benchmark_runs/prefill-gdn-batch-20261116T071709Z/full-engine-before-rollback. Complete baseline and earlier source variants remain under the ignored benchmark_runs tree; the final source and binary-capable diff are copied into this report. Large raw traces remain outside Git.

All four experiment containers stopped exit0/OOMfalse, serviceMainPID0, GPU idle, same boot, zero new swap across11 bounded worker stages. STOP/GPU/LARGE holds remain. Initial inspection found the manual-chat agent container running but idle; it was stopped after checking its processes, before experiments. No active chat was interrupted.

LOCAL_MODEL_HANDOFF.md records authorization, accepted/rejected work and checks. QWEN_TASK.md points future local tasks to that file. The raw local-author request harness now also includes the bounded handoff automatically in future prompts and records its hash; this is context, not model training. No local model was loaded for this directly authored experiment.

Do not retry plain Torch batching without a measured plan to eliminate packing and preserve vendor-solver arithmetic. A kernel operating on the existing tensor layout is a distinct, unimplemented option. Another independent iteration-speed option is sharing one native checkpoint load across benchmark/quality/profile phases; it has not been implemented here.

To reproduce the candidate, use its retained source in an isolated checkout/container mount with the existing installed runtime and guarded harness. The active project deliberately contains the restored baseline, so running the archived worker against it would not reproduce candidate timings. Weights and runtime images are unchanged; no downloads, package installation, unrelated services or serving-engine substitutions were used.
