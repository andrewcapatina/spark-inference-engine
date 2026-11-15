# Compact convolution history — quarantined

The local-Qwen candidate021 adds only `.clone()` after selecting the last convolution-history window in GDN prefill. It preserves the accepted QKV reuse. **This candidate is not enabled in the default engine.** Project source was restored exactly to validated QKV checkpoint6dd8d9f after the initial strict continuation failure.

28tiny dense/mixed-quantized cases pass element-exact output/cache/continuation checks and compact backing-storage checks. The first real256prompt benchmark then passed entire prefill logits/cache comparisons but failed the128generated-ID comparison. Its log and full candidate were retained before rollback. That first harness did not retain the differing IDs; subsequent strict failures now save both sequences before stopping.

The failure did not reproduce in two focused real256prompt diagnostic runs: all four free128-token generations match in each, then all module outputs/logits and every cache tensor match through128teacher-forced steps for both baseline/baseline and baseline/candidate. A final uninstrumented strict replay, importing exact retained candidate SHA667663c34ebf5ae9a5dde118a24d5e31e7d968ac328656329ba6ef48460a1bba from an isolated readonly harness package, passes the original entire-logit/cache/128-ID assertions at256 and1024. No tolerance or numerical check was weakened.

Those successful replays do not explain the initial failure. There was no engine repair between them. The candidate stays quarantined rather than treating rerun success as a fix; its quality/protected/profile acceptance sequence was not completed.

## Diagnostic measurements, not an accepted optimization

| Prompt | Convolution backing storage before → after | Prefill CUDA median before → after | Peak allocated tensor bytes before → after |
| --- | --- | --- | --- |
|256|240MiB →3.75MiB|279.09→275.53ms|23240434176→22992183808|
|1024|960MiB →3.75MiB|1154.32→1177.36ms|24457449984→23439151616|

This confirms the memory-retention mechanism, including roughly0.95GiB lower measured peak at1024. It does not establish a consistent latency improvement. Paired128-output throughput is9.085→9.069 and8.276→8.252tok/s, essentially unchanged. The baseline here already includes QKV reuse; these results must not be combined arithmetically with earlier speedups.

All successful diagnostics use one resident model and release CUDA allocations/reservations to0/0 with clean container teardown, same boot and no new swap. Failed preflights remain recorded separately: readonly file-scoped advice to inactive checkpoint ranges restored headroom; the56GiB CUDA floor,32GiB host floor and256MiB new-swap guard were unchanged. No global cache flush, package installation or unrelated service changes occurred.

Main run: `gdn-conv-history-20261115T062259Z`. Initial failure: `conv-ab-20261115T063606Z`. Later run names are identified in the JSON files and provenance. Raw tensors/source snapshots remain under the controller/ignored benchmark artifacts. Only local Qwen authored engine code; supervisor changes were fixtures, diagnostics, scheduling, prompts and exact candidate placement/rollback.

The validated default remains [QKV reuse](../2026-11-14-qkv-reuse/README.md). The next independent authorized experiment removes duplicate attention-cache concatenations before evaluating preallocation.
