# Spark inference engine

A small native PyTorch inference engine developed on an NVIDIA DGX Spark.
It loads local model weights and implements the model's forward pass,
request state and greedy token generation in this repository.

The current text paths cover the downloaded Qwen3.5-family 9B Safetensors model
the downloaded Qwen3.5-architecture 27B Q8_0 GGUF model, and the downloaded
Unsloth Qwen3.8-27B mixed NVFP4/FP8 Safetensors checkpoint, plus a separately
validated small Llama baseline. The 27B checkpoint is marketed as Qwen3.8;
its GGUF architecture identifier is `qwen35`.

## Start here

- [Architecture walkthrough](ARCHITECTURE.md): what the implemented system does,
  where to find each component, and what remains a design idea.
- [Performance measurements](PERFORMANCE.md): baseline results, recovery limitations,
  and the selected first optimization.
- [Options review](OPTIMIZATION_OPTIONS.md): reliability work and alternative
  performance directions; the legacy automatic queue remains held outside explicitly authorized trials.
- [CLI usage](CLI_USAGE.txt): chat, completions, commands and the existing local
  Docker setup.
- [Original component map](INFERENCE_ENGINE_DIAGRAM.txt): the preserved design
  roadmap, including future components.
- [Progress history](PROGRESS.md) and [Qwen task boundaries](QWEN_TASK.md).

## Run a local model

From the repository root, in an environment with compatible CUDA-enabled
PyTorch, NumPy, Safetensors, Transformers and Tokenizers installed:

```bash
python3 -m spark_engine --model /path/to/qwen-model --chat
python3 -m spark_engine --model /path/to/qwen35-model.gguf --chat
```

The repository does not include model weights or provision a fresh CUDA
environment. The container commands in [CLI_USAGE.txt](CLI_USAGE.txt) describe
the existing development machine; that container is not created by cloning
this repository.

Chat keeps the model loaded. `/reset` clears history, `/exit` quits, and Ctrl-C
can cancel an unfinished reply. Generation is greedy and replies print after
generation completes. The default chat limit is 128 new tokens; increase it
with `--max-new-tokens`.

Llama raw completion remains available:

```bash
python3 -m spark_engine --model /path/to/llama-model --prompt "Hello"
```

## Native NVFP4 chat on the development Spark

The pinned downloaded checkpoint uses native quantized matrix operations through
FlashInfer/CUTLASS and PyTorch scaled_mm. It needs the validated newer GPU runtime;
the original agent image does not provide all these dependencies. From this repo:

```bash
python3 tools/chat_nvfp4.py
```

This launcher uses the installed pinned GB10 image, verifies the checkpoint
manifest, acquires experiment locks and refuses competing GPU work. It starts
`spark-prototype-quant-agent`, runs the native CLI
with a 32 GiB allocator cap and host/swap guards, then stops its container on exit.
It installs nothing and keeps model/project mounts read-only. The model stays
loaded during chat. Numerical checks and a real two-turn conversation passed;
see [scope and evidence](benchmarks/results/2026-11-14-native-nvfp4/README.md).
NVFP4/FP8 weights remain packed; attention KV cache remains BF16. Vision, MTP and
long-context reliability are outside this validation.

## Validation and scope

The development experiment recorded 56 passing Qwen acceptance gates and 15
passing Llama acceptance cases. These include tiny numerical/component tests
and actual-checkpoint tests; they are not 56 full-model evaluations. CLI work
also passed 16 focused tests and real 9B/27B two-turn chat/reset checks.
See the [evidence and limitations](ARCHITECTURE.md#validation-and-reproducibility).

The portable CLI tests can be run without loading model weights:

```bash
python3 -m unittest discover -s tests -p test_cli.py -v
```

The external CUDA acceptance harness, reference checkpoints and golden outputs
used during development are not bundled in this repository. Vision/video,
MTP/speculative decoding, other quantization formats, non-greedy sampling,
long-context reliability and production performance remain unvalidated.

The local Qwen model wrote the native inference implementation. Codex supervised
the experiment and wrote the CLI with the user's explicit authorization.
