# Performance baseline methodology

[`tools/benchmark_qwen.py`](tools/benchmark_qwen.py) measures the existing native
Qwen engine. It does not implement or replace generation. Run one model per
process, with other model sessions closed, in the same CUDA environment used
for the CLI.

```bash
python3 tools/benchmark_qwen.py \
  --model /path/to/qwen-model \
  --output-dir benchmark_runs/qwen9b-baseline \
  --cuda-limit-gib 24 --min-available-gib 32
```

Use a GGUF path for the 27B run and a different output directory. An existing
output directory is rejected to preserve previous results. The existing agent
container can run this command from `/workspace`; its local model paths are in
[CLI_USAGE.txt](CLI_USAGE.txt).

## Workload

- Batch size 1, CUDA BF16 weights, four CPU threads, TF32 disabled.
- Exact prompt lengths: 32, 256 and 1024 token IDs by default.
- The tokenizer encodes a fixed public text once; those IDs are repeated and
  sliced to the requested length. Base IDs and input hashes are recorded.
- Exactly 128 new tokens, with EOS stopping disabled. This controls work size;
  the continuations are not a response-quality or chat evaluation.
- One warmup and three uninstrumented measured repetitions per prompt length.
- Model loading happens once per process. OS/NAS caches are uncontrolled;
  **fresh process is not a guarantee of disk-cold loading**. No system caches
  are dropped.

## Metrics and boundaries

| Metric | Meaning |
| --- | --- |
| Model/tokenizer load milliseconds | Production CLI loader, after explicit CUDA initialization, synchronized before completion |
| Baseline wall milliseconds | Unmodified native `generate`, synchronized before/after; IDs are already on CUDA |
| Generated tokens/second | Fixed output count divided by baseline wall time; includes prefill and the actual native loop's work |
| First-token CUDA milliseconds | Separate observed run: CUDA stream time from generation start to the first selected token |
| Prefill CUDA milliseconds / tokens per second | Separate observed run: the existing native prefill call |
| Decode tokens/second, CUDA | Separate observed run: token-selection markers 1 through N; counts N-1 intervals after the first token |
| Final unused decode milliseconds | Existing loop's decode after the last length-limited output token; recorded separately |
| CUDA allocated/reserved peaks | PyTorch allocator peaks reset per measured phase; include model weights |
| VmRSS / VmHWM / VmSwap | Linux process RSS, process-lifetime peak RSS and swap; these are not GPU memory totals |

First-token metrics describe model computation, not the CLI's first visible
text: the CLI prints a completed reply. Tokenization, chat-template rendering,
input transfer and output decoding are outside model-only baseline timing.
CUDA-event observation has its own overhead and is kept separate from the
uninstrumented baseline. Its output hash is compared with the baseline hash.
Repeated baseline hashes are also reported.

CPU and GPU share physical capacity on the development Spark. Do not add their
reported capacities or assume that process RSS includes every CUDA allocation.
Global free-memory snapshots can reflect other processes and file caches.

## Profiling

`--profile` captures a separate warmed request with 256 prompt tokens and 8 new
tokens by default. It records CPU/CUDA activity, input shapes and allocation
events, with labels around existing `prefill` and `decode` calls. Method
observers pass through the original outputs and restore instance attributes
exactly, including when an exception occurs.

`results.json` includes the top operators by self device time and inclusive
stage times. Inclusive stages overlap their child operators and must not be
added to operator self-time totals. Profiled timings and allocation deltas are
diagnostic; they are not uninstrumented latency or peak-memory measurements.

Raw `profile-trace.json`, samples CSV and progress/result JSON go into the
requested directory. `benchmark_runs/` is ignored by Git because traces can be
large. Portable summaries and a findings report can be checked in separately.

Timing/accounting tests:

```bash
python3 -m unittest discover -s tests -p test_benchmark.py -v
```

The harness was also checked on the existing tiny native CUDA Qwen fixture:
baseline, observed and profiled outputs matched; markers counted all native
decode calls, and the observers restored the model's methods. That probe is a
measurement-harness test, not a full-model performance result.

Reference APIs: [CUDA execution/timing semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
and [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).

## Shutdown recovery diagnostics

The interrupted November 14 run is preserved in [the partial report](benchmarks/results/2026-11-14-interrupted/README.md). Subsequent harness revisions flush and fsync results and phase/sample telemetry (`phases.jsonl`). Telemetry runs outside measured generation intervals, but changes the protocol and must be reported separately from the original baseline.

`--cuda-limit-gib` optionally sets the PyTorch caching allocator limit before loading. This does not limit CPU, driver or other non-PyTorch allocations and is not a guarantee against host shutdown. `--min-available-gib` (default 16) rejects work when system available memory is below that threshold at stage boundaries; it cannot catch a fast allocation spike inside a native call. These controls have not yet been GPU validated. See the [PyTorch allocator API](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.set_per_process_memory_fraction.html). The heavy run remains stopped.

## Consolidated findings

[Performance measurements](PERFORMANCE.md) keeps original and recovery protocols separate and selects the first local-Qwen optimization. The example above uses a 9B-specific allocator cap; use the tested 56 GiB cap for the downloaded 27B. A cap does not replace the external host watchdog used in recovery. Profiling is optional and remains deferred for further runs while the shutdown cause is unresolved. The completed original 9B profile remains available as evidence.

## Event accounting, schema 2

Returned output length is authoritative for the number of selected tokens. Decode entry marks each preceding selection. When generation returns without a decode for its final selection (including early EOS or a one-token reply), the final CUDA event recorded after `generate` returns supplies that marker. This includes the small return/marker-submission interval; the marker kind is recorded explicitly. A final unused decode is reported only when there is a decode call for every returned token. One-token output has no decode-throughput interval. The same revised observer must measure both sides of an optimization comparison.
