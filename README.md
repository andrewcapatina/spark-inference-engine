# Spark inference engine

A small native PyTorch inference engine developed on an NVIDIA DGX Spark.
It loads local model weights and implements the model's forward pass,
request state and greedy token generation in this repository.

The current text paths cover the downloaded Qwen3.5-family 9B Safetensors model
and the downloaded Qwen3.5-architecture 27B Q8_0 GGUF model, plus a separately
validated small Llama baseline. The 27B checkpoint is marketed as Qwen3.8;
its GGUF architecture identifier is `qwen35`.

## Start here

- [Architecture walkthrough](ARCHITECTURE.md): what the implemented system does,
  where to find each component, and what remains a design idea.
- [Performance measurements](PERFORMANCE.md): baseline results, recovery limitations,
  and the selected first optimization.
- [Options review](OPTIMIZATION_OPTIONS.md): reliability work and alternative
  performance directions; further GPU work is currently held.
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
