# GDN compiler experiments — rejected and rolled back

No compiler optimization from this package was accepted. The complete local-Qwen implementation and rejected revisions are retained under the controller's ignored experiment artifacts. Exact original engine bytes were restored and verified against112d03d before the subsequent QKV-reuse work. No numerical thresholds were loosened.

| Candidate | Isolated result | Correctness / decision |
| --- | --- | --- |
| Installed FlashInfer GDN | Actual first-layer256 input:1.714ms native versus0.350ms adapter,4.90x; includes normalization/gate/state-layout conversions |604/786432 final-state elements exceed original .01/.002 bounds, maxerror0.08617; rejected. Visible output passes its .01/.00390625 bounds. |
| TorchInductor, unchanged native formula | Helper17lengths passes output/state/continuation,5graphs,23.60s cold warmup; isolated256/1024 speedups1.56x/1.44x |Real27B full-logit comparison fails before timing. Not accepted despite operation-level passes. |
| Eager-numerics compiler options |Actual inputs still differ in279 BF16 output elements; individual fixed bounds pass |Does not restore element-exact native arithmetic. |
| C++ host wrapper |Same actual input about1.11ms, ordinary compiled wrapper about1.10ms |No material gain; triangular solve uses installed proxy-executor fallback. Not a complete C++ engine. |
| Installed PyTorch CUDA-graph replay |Exact output/state on actual first layer and5synthetic sizes; output/cache ownership checks pass;256~1.13x,1024~1.03x,512~0.96x |Diagnostic only, modest/inconsistent operation speedup; no engine integration. |

The real-model control repeats exactly through all64 layers, recurrent states and63,569,920 logits. With the compiler candidate, small first-layer differences grow through the model: first hidden-bound misses occur at layer2. At the final logits,57,131,367 elements exceed .01/.03125, maxdifference16.125, RMS1.142.249/256 teacher-forced top1 choices match; matching the last token alone would miss this failure. This is measured propagation, not a proposed error-correction feature. Baseline repeatability distinguishes it from runtime nondeterminism.

All48 GDN operations pass the fixed per-operation bounds when each receives the same baseline real activation. Max outputerror0.000122 and maxstateerror1.31e-5. Copying those inputs to contiguous storage and using the original operation is element-exact, so input contiguity alone does not explain the failure.

Original protected tests were never weakened. Because the candidate failed the new real-checkpoint gate, its subsequent throughput/quality/profile acceptance sequence was stopped; no whole-engine speedup is claimed. All completed diagnostics released CUDA allocations/reservations to0/0, retained the host boot, used no new swap, and stopped the experiment containers. One resident large model at a time; global GPU holds and the inactive legacy service were preserved.

Only local Qwen authored engine implementation. Supervisor code consists of diagnostics, fixtures, controllers and documentation. Raw traces, responses and activation tensors remain local, outside Git. Main controller run: `gdn-prefill-20261115T045703Z`; associated run names are recorded in provenance and JSON files.

The user selected QKV projection reuse next. That separate change preserves existing arithmetic and cache layout; its tests and performance must be reported independently.

PyTorch explains the generated [C++ host wrapper](https://docs.pytorch.org/tutorials/unstable/inductor_cpp_wrapper_tutorial.html). Installed runtime/source and measured artifacts are authoritative for this experiment.
