# Verified baseline and bounded tasks

The forward pass has been checked against Transformers on CUDA: tiny tied and
untied models, a real SmolLM2 checkpoint, batched causal behavior, and RoPE with
nonzero position offsets. Keep all of these passing. Do not change the loader,
RoPE frequency ordering, or GQA expansion to implement generation or the CLI.
Reference implementation: installed transformers/models/llama/modeling_llama.py.

Work through these tasks in order, one named check at a time:

1. `test_attention_kv_cache`: extend ONLY `Attention.forward` with optional
   `past_key_value=None, use_cache=False`. Default behavior still returns a
   tensor. With use_cache=True return `(output, (keys, values))`. Compute RoPE
   on NEW q/k only, then concatenate old/new k/v along sequence dim 2. Store
   UNREPEATED KV tensors `[batch, kv_heads, total_tokens, head_dim]`. Expand
   heads only for attention math. Do not mutate the supplied cache. Existing
   masks are boolean `[new_tokens,total_tokens]`, broadcast once across batch
   and heads. positions is `[batch,new_tokens]`; cos/sin broadcasts as
   `[batch,1,new_tokens,head_dim]`. Run the fixed targeted check and finish.
2. `test_incremental_cache_and_request_isolation`: extend DecoderLayer.forward
   with the same optional cache convention, then add a shared internal runner
   in CausalLM. Cache is a list of layer `(k,v)` pairs owned by the request.
   Position offset = past keys' shape[2]. Causal mask compares absolute new
   query positions with all key positions. prefill returns prompt logits and
   per-layer cache; decode returns NEW-token logits and the updated cache.
   forward(input_ids) must retain its original tensor-returning API. Multiple
   requests must never share mutable cache state. Work may take two short turns;
   preserve the independently passing attention task and forward baseline.
3. `test_greedy_generation_and_stop`: generate uses prefill once, then decode
   with only the newly emitted token. Handle max_new_tokens=0 and EOS exactly
   as ENGINE_CONTRACT.md specifies. Do not change attention or the loader.
4. `test_cli_runs_offline`: add spark_engine/__main__.py using argparse,
   AutoTokenizer.from_pretrained(local_files_only=True), your CausalLM, and
   --device cuda. Move model and token IDs to CUDA, use inference_mode, and
   print decoded generated text. Do not edit model.py for this task.

One focused edit, one targeted check, a brief PROGRESS.md note, then end the turn.
If two attempts produce the same failure, the controller stops for review.
A checkpoint means code was saved; only passing checks count as validated progress.
