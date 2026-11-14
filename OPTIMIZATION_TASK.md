# Local Qwen task: skip the final unused decode

Status: local-Qwen implementation checkpointed at `18894a8`; all 56 unique Qwen gates, 15 Llama cases and focused control-flow checks pass. Matched performance comparison is pending.
Selection evidence: [performance report](PERFORMANCE.md#selected-first-optimization-remove-the-unused-final-decode).

## Problem and scope

Native `QwenCausalLM.generate` currently performs `max_new_tokens` decode calls when EOS is disabled, although the first token is already selected from prefill logits. The final call computes logits that are discarded. The CUDA-event observer measured that call at approximately 93 ms for 9B and 233–239 ms for 27B.

Only the local Qwen model may author the implementation. Its sole implementation scope is `QwenCausalLM.generate` in `spark_engine/qwen_model.py`, with a small function-only patch. Preserve the function signature and surrounding source. Do not edit acceptance fixtures, tests, tolerances, layers, loaders, text adapters or Llama code. Do not introduce another inference backend.

Stop generation after appending the last requested output token. Preserve exact returned IDs, prompt prefix, EOS handling, device, dtype and existing behavior for zero requested tokens. Public prefill/decode interfaces remain unchanged. This task does not introduce persistent conversation caches, sampling or numerical changes.

Expected successful control-flow property for a positive output budget N with EOS disabled: one prefill and exactly N-1 decode calls. For early EOS at output position K, return that EOS token and perform no decode afterward. A one-token request must return the token chosen from prefill without calling decode. Do not provide a supervisor-written replacement function to the model.

## Before implementation

1. Preserve the current native function, full source hash and Git checkpoint. Retain every candidate, its local-model output and diagnostics before any rollback.
2. Repair benchmark accounting in the supervisor-owned harness before using it for before/after claims. The current observer records each token-selection marker at decode entry and assumes a final unused decode exists. After this change, both assumptions break for the final token and for N=1. Derive actual output count from the returned IDs; account for the final selection, report the true decode-call count and label the unused final decode absent. Do not silently report N-1 selected tokens or relabel the last useful decode as unused.
3. Check event accounting on deterministic tiny fixtures for the old and proposed call patterns, including one-token output and early EOS. Check observer restoration on success and failure. Keep full returned-output digests as the equality check. Document any revised final-selection marker semantics and use the same revised harness on both engine versions.
4. Collect a fresh capped, non-profiler baseline with that harness. Historical measurements select the task, but cannot replace a matched before/after baseline. Record both source hashes, settings and boot ID.

## Implementation and protected correctness

Give the local model only the scoped function task. Review candidate scope before placement. Any code repair must go back to local Qwen; the supervisor may repair the harness only.

The supervisor may add a focused behavioral test using deterministic fake prefill/decode outputs. Assert exact generated sequences and decode call counts for N=1, several tokens with EOS disabled, early EOS and EOS selected on the final allowed token. Include the established zero-token output behavior. The test should establish absence of unused computation as well as preserved results.

Run the unchanged tiny generation/EOS tests, the complete protected Qwen acceptance set and the 15 Llama baseline cases. Preserve the existing real-9B full-logit tolerances (rtol=0.05, atol=0.05), exact-reference greedy IDs, and actual 9B/27B conversation checks. Passing a tiny control-flow test is not real-checkpoint validation. No numerical gate may be weakened or replaced by a performance check.

Use one model process at a time. Release the local coding model before loading actual inference checkpoints for verification; do not overlap two large model copies on shared physical memory. Check controller/container state and acquire the experiment locks before starting. Keep profiling disabled in this task.

## Measure and accept

Compare the unchanged baseline and candidate with the same capped, non-profiler harness, prompt IDs, token budgets and settings. Use short replies (8 output tokens) and the existing longer workload (128); use representative 256- and 1024-token prompts, with one warmup and three measured repetitions. Start with one checkpoint and proceed to the other only after correctness and memory checks pass. Use durable per-stage/sample logs and the verified allocator cap; retain the external host watchdog. Select per-model caps with headroom rather than assuming file size equals runtime memory.

Require exact before/after generated IDs for every comparison, no extra decode after the final token, and unchanged protected acceptance behavior. Report actual wall-time distributions and peak allocated/reserved memory. Do not claim a speedup from the source diff alone or subtract a historical event duration from a new wall-time sample. Small 128-token differences may be within run-to-run variation; report that honestly.

The expected benefit is lower complete-reply latency, especially for short replies. First-token latency and steady-state useful-decode rate should remain comparable. Potential savings estimated from historical event runs are approximately 0.7–0.8% for 128-token replies; this is not a major throughput optimization.

If correctness regresses, retain the rejected patch and logs, restore the checkpoint, and send the smallest actual failure back to local Qwen. If a memory guard triggers, classify it as a runtime/containment failure before changing inference arithmetic. A host shutdown requires another recovery review rather than an automatic repeat.

## Completion evidence

Record the local model's candidate identity, accepted commit, preserved regression results, actual-checkpoint before/after output digests, revised harness version, timing distributions, memory peaks and any unresolved limitations. Update `PROGRESS.md` and experiment status/checkpoints. Keep the original interrupted benchmark and recovery diagnostic reports intact.
