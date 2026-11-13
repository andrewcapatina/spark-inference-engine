# Minimum prototype contract

Implement this interface with your own PyTorch forward pass. CPU FP32 is the
first target. Read-only acceptance tests are at `/workspace/acceptance`.

## Stage 1: small verified component

Create `spark_engine/__init__.py` and `spark_engine/ops.py` with:

`rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor`

Normalize the last dimension using root mean square and apply weight. Support
FP32 tensors, noncontiguous inputs, arbitrary leading dimensions, and zeros.
Preserve input values. Implement the math; do not delegate to nn.RMSNorm.

## Stage 2: forward pass and loading

In `spark_engine/model.py`, provide `CausalLM(torch.nn.Module)`:

- `CausalLM(config: dict)` creates a Llama-style decoder in FP32.
- `load_hf_state_dict(state_dict)` maps Transformers Llama weights into it.
- `forward(input_ids)` returns logits shaped `[batch, sequence, vocabulary]`.
- `CausalLM.from_pretrained(directory)` reads local config and Safetensors,
  including tied embeddings, and returns a CPU FP32 model.

Implement GQA, RoPE, RMSNorm, gated SiLU MLPs, residuals and the output head.
Validate both tiny random models and the supplied SmolLM2 checkpoint.

## Stage 3: cache and usable generation

- `prefill(input_ids)` returns `(all_prompt_logits, cache)`.
- `decode(new_token_ids, cache)` returns `(new_token_logits, updated_cache)`.
- Each request owns separate cache state. Cache full-attention keys and values;
  avoid recomputing the whole prefix for each decoded token.
- `generate(input_ids, max_new_tokens, eos_token_id=None)` returns the prompt
  concatenated with generated IDs. Greedy selection; stop after emitting EOS
  when supplied. None disables EOS stopping; zero new tokens returns the input.
- `python3 -m spark_engine --model PATH --prompt TEXT --max-new-tokens N`
  runs offline using the local tokenizer and prints generated text.

Reference path: `/workspace/reference/smollm2`.
Design document: `/workspace/INFERENCE_ENGINE_DIAGRAM.txt`.
Controller feedback: `/workspace/run_feedback/latest.txt`.

The controller rechecks previous stages before advancing. Acceptance tests are
fixed outside your writable project. Add your own tests under `tests/` too.
Full Qwen hybrid support, CUDA performance and multiple Sparks remain later work.
