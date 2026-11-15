# Native 27B NVFP4 benchmark and accuracy scope

Follow-up: [teacher-forcing and identical-prefix scores](../2026-11-14-nvfp4-teacher-forcing/README.md) show reference cache/fresh-prefix sensitivity,125/128 same-history top choices matching, and remaining numerical discrepancies.

Local Qwen authored the engine implementation. The supervisor supplied task boundaries, independent fixtures, harness recovery, measurements and documentation. Implementation/provenance: [native update](../2026-11-14-native-nvfp4/README.md).

Batch1, 128 new tokens, EOS disabled, one warmup and three timed repetitions per prompt length; load excluded. Separate CUDA-event diagnostics do not contribute to wall medians. One large checkpoint at a time, 32GiB allocator cap and host floor, 256MiB new-swap guard. No profiler, CUDA graphs or MTP.

| Prompt tokens | Native output tok/s | Median wall seconds | Previous BF16 output tok/s | vLLM NVFP4 output tok/s |
| ---: | ---: | ---: | ---: | ---: |
|256|9.172|13.956|4.328|11.453|
|1024|8.346|15.336|4.074|11.093|

Native is about2.12x/2.05x the previous BF16 results. Different checkpoint weights and Torch runtime prevent attributing these ratios purely to quantization. vLLM also uses different attention/GDN kernels and FP8 KV. Timings are raw synthetic repeated-token workloads, not chat quality tests or production guarantees.

Packed resident model tensors:22,586,157,376bytes (21.04GiB), versus approximately50.10GiB for the old BF16 model. Peak allocated including inference:21.65GiB/22.78GiB. Fresh-process model/tokenizer load164.187s; filesystem/NAS caches uncontrolled, not a guaranteed cold load. CUDA-event first-token selection302.5ms/1227.1ms; useful decode9.294/9.008tok/s. CLI output remains buffered until the reply finishes.

## What correctness means here

Independent CUDA operator tests compare quantized operations against unpacked floating-point references at fixed tolerances. A sequential tiny hybrid reference/native test matches forward/prefill/decode with max error0 and exact4-token generation. All56 existing Qwen gates and15 Llama cases passed, including original real checkpoints/conversation, without tolerance changes. Actual quantized27B first8 tokens match vLLM at both prompt lengths; two-turn math chat answers4 then7. These are separate milestones, not interchangeable evidence.

All native timed repetitions and CUDA-event observations have matching output hashes. The1024-token prompt matches all128 vLLM output IDs. The256-token prompt differs later in the128-token output. A separate one-sample diagnostic using explicit BF16 KV also differs. vLLM automatically changed attention from FLASHINFER to FLASH_ATTN2 in that diagnostic, so it does not isolate cache precision. The discrepancy is unresolved; full real-checkpoint numerical parity and broad quality have NOT been established. The diagnostic is not another performance baseline.

Next accuracy work should compare identical-input logits, top-token margins, and teacher-forced decode with a reference, then evaluate representative language tasks. Autoregressive exact-token matching alone is fragile: an early rounding-induced choice changes subsequent context, while an implementation error can produce plausible text. Existing passing regression tests remain required.

## Shutdown and reproducibility

After model release, weak references to all model tensors are gone. Explicit release of idle cuBLAS/FlashInfer workspace restores tracked CUDA allocation/reservation to0/0 with original16MiB/64MiB cleanup bounds preserved. A no-model control diagnosed the prior72MiB residue as library scratch. Worker exit0, container exit0/PID0/OOMfalse; host available memory recovered within4GiB, no swap use, same boot. This finite run does not establish long-session stability or explain earlier poweroffs.

Exact source hashes are in raw results. Repository at benchmark launch:47f3e725896f3417ff8ec162c642a3b6d291126f. Raw git_commit is null because read-only container ownership blocked git; preserved unchanged. A later harness-only per-command safe.directory fix restores metadata without global Git configuration. Archived diagnostic fixtures use .py.txt to avoid accidental test collection.

Try chat with the guarded launcher (existing pinned runtime image, one experiment model, bounded memory):

```bash
cd /home/cap/Projects/spark-inference-engine
python3 tools/chat_nvfp4.py
```

The legacy unattended service/queue remains inactive. All four experiment containers were stopped after validation. Raw durable runs: native-nvfp4-benchmark-20261115T022402Z and matched-kv-reference-20261115T024116Z under /home/cap/spark-inference-overnight-20261113/runs. No weights or bulk traces are committed.

Sampled host minimum available: 71.26GiB; GPU maximum temperature: 79.0C; maximum thermal-zone reading: 88.7C. Telemetry is sampled, not a bound on unobserved transients.
