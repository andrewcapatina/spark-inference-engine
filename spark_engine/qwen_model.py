"""Qwen3.5 hybrid causal LM. Assembles layers from config; official weight names."""
import torch
import torch.nn as nn
from spark_engine.qwen_layers import QwenAttention, QwenGatedDeltaNet, QwenRMSNorm


class QwenDecoderLayer(nn.Module):
    def __init__(self, config, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.input_layernorm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        if layer_type == "full_attention":
            self.self_attn = QwenAttention(config)
        else:
            self.linear_attn = QwenGatedDeltaNet(config)
        self.post_attention_layernorm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.mlp = QwenMLP(config)

    def forward(self, hidden, position_embeddings=None, attention_mask=None):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "full_attention":
            hidden = self.self_attn(hidden, position_embeddings, attention_mask)
        else:
            hidden = self.linear_attn(hidden)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden

    def prefill(self, hidden, position_embeddings=None, attention_mask=None):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "full_attention":
            hidden, mixer_cache = self.self_attn.prefill(hidden, position_embeddings, attention_mask)
        else:
            hidden, mixer_cache = self.linear_attn.prefill(hidden)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden, mixer_cache

    def decode(self, hidden, position_embeddings, mixer_cache):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        if self.layer_type == "full_attention":
            hidden, mixer_cache = self.self_attn.decode(hidden, position_embeddings, mixer_cache)
        else:
            hidden, mixer_cache = self.linear_attn.decode(hidden, mixer_cache)
        hidden = residual + hidden
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden, mixer_cache


class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        if (x.ndim == 3 and x.shape[0] == 1 and x.shape[1] >= 16
                and x.is_cuda and x.dtype == torch.bfloat16
                and torch.is_inference_mode_enabled()):
            try:
                from spark_engine.qwen_mlp_activation import silu_mul
            except ModuleNotFoundError as e:
                if e.name is None or not (e.name == "triton" or e.name.startswith("triton.")):
                    raise
            else:
                return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenCausalLM(nn.Module):
    @classmethod
    def from_pretrained(cls, directory, dtype=torch.bfloat16, device="cuda"):
        import json, os
        from safetensors.torch import load_file
        with open(os.path.join(directory, "config.json")) as f:
            cfg = json.load(f)
        quant_cfg = cfg.get("quantization_config")
        if quant_cfg:
            from spark_engine.qwen_quantized import load_quantized_qwen
            return load_quantized_qwen(directory, dtype=dtype, device=device)
        text_cfg = cfg.get("text_config", cfg)
        rope = text_cfg.get("rope_parameters", {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25})
        rope_theta = float(rope.get("rope_theta", 10000000.0))
        head_dim = text_cfg.get("head_dim", text_cfg["hidden_size"] // text_cfg["num_attention_heads"])
        engine_cfg = {
            "vocab_size": text_cfg["vocab_size"],
            "hidden_size": text_cfg["hidden_size"],
            "intermediate_size": text_cfg["intermediate_size"],
            "num_hidden_layers": text_cfg["num_hidden_layers"],
            "num_attention_heads": text_cfg["num_attention_heads"],
            "num_key_value_heads": text_cfg.get("num_key_value_heads", text_cfg.get("linear_num_key_heads", text_cfg["num_attention_heads"])),
            "linear_num_key_heads": text_cfg.get("linear_num_key_heads", text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"])),
            "linear_num_value_heads": text_cfg.get("linear_num_value_heads", text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"])),
            "linear_num_heads": text_cfg.get("linear_num_heads", text_cfg.get("num_attention_heads", text_cfg["linear_num_value_heads"])),
            "linear_key_head_dim": text_cfg.get("linear_key_head_dim", head_dim),
            "linear_value_head_dim": text_cfg.get("linear_value_head_dim", head_dim),
            "linear_conv_kernel_dim": text_cfg.get("linear_conv_kernel_dim", 4),
            "head_dim": head_dim,
            "rms_norm_eps": text_cfg["rms_norm_eps"],
            "layer_types": text_cfg.get("layer_types", ["dense"] * text_cfg["num_hidden_layers"]),
            "rope_parameters": rope,
        }
        model = cls(engine_cfg).to(dtype=dtype, device=device)
        index_path = os.path.join(directory, "model.safetensors.index.json")
        engine_keys = set(model.state_dict().keys())
        loaded = set()
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_map = index["weight_map"]
            for shard_name in sorted(set(shard_map.values())):
                shard_path = os.path.join(directory, shard_name)
                shard = load_file(shard_path)
                for name, tensor in shard.items():
                    if name.startswith("model.visual") or name.startswith("model.mtp"):
                        continue
                    if name == "lm_head.weight":
                        target = "lm_head.weight"
                    elif name.startswith("model.language_model."):
                        target = name[21:]
                    elif name.startswith("model."):
                        target = name[6:]
                    else:
                        target = name
                    if target in engine_keys:
                        model.state_dict()[target].copy_(tensor.to(dtype=dtype, device=device))
                        loaded.add(target)
                    elif target == "model.embed_tokens.weight":
                        model.state_dict()["embed_tokens.weight"].copy_(tensor.to(dtype=dtype, device=device))
                        loaded.add("embed_tokens.weight")
                del shard
        else:
            state = load_file(os.path.join(directory, "model.safetensors"))
            for name, tensor in state.items():
                if name == "lm_head.weight":
                    target = "lm_head.weight"
                elif name.startswith("model."):
                    target = name[6:]
                else:
                    target = name
                if target in engine_keys:
                    model.state_dict()[target].copy_(tensor.to(dtype=dtype, device=device))
                    loaded.add(target)
            del state
        missing = engine_keys - loaded
        if missing:
            raise RuntimeError(f"Checkpoint missing {len(missing)} engine weights: {sorted(missing)[:5]}")
        return model

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = nn.ModuleList(
            QwenDecoderLayer(config, config["layer_types"][i])
            for i in range(config["num_hidden_layers"])
        )
        self.norm = QwenRMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)

    def load_hf_state_dict(self, state):
        engine_keys = set(self.state_dict().keys())
        loaded = set()
        for name, tensor in state.items():
            if name == "lm_head.weight":
                target = "lm_head.weight"
            elif name.startswith("model."):
                target = name[6:]
            else:
                target = name
            if target in engine_keys:
                self.state_dict()[target].copy_(tensor.to(self.state_dict()[target].device))
                loaded.add(target)
        missing = engine_keys - loaded
        if missing:
            raise RuntimeError(f"Checkpoint missing {len(missing)} engine weights: {sorted(missing)[:5]}")
        unexpected = set(state.keys()) - {"model." + k if k != "lm_head.weight" else "lm_head.weight" for k in engine_keys}
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {sorted(unexpected)[:5]}")

    def forward(self, ids):
        bsz, seq = ids.shape
        hidden = self.embed_tokens(ids)
        rope = self.config["rope_parameters"]
        head_dim = self.config["head_dim"]
        rotary_dim = int(head_dim * rope["partial_rotary_factor"])
        inv_freq = 1.0 / (rope["rope_theta"] **
                          (torch.arange(0, rotary_dim, 2, device=ids.device, dtype=torch.float) / rotary_dim))
        inv_freq = inv_freq.repeat(2)
        positions = torch.arange(seq, device=ids.device, dtype=torch.float)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        cos = freqs.cos().unsqueeze(0).to(hidden.dtype)
        sin = freqs.sin().unsqueeze(0).to(hidden.dtype)
        position_embeddings = (cos, sin)
        causal_mask = torch.full((seq, seq), float('-inf'), device=ids.device, dtype=hidden.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings, causal_mask)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    def prefill(self, ids):
        bsz, seq = ids.shape
        hidden = self.embed_tokens(ids)
        rope = self.config["rope_parameters"]
        head_dim = self.config["head_dim"]
        rotary_dim = int(head_dim * rope["partial_rotary_factor"])
        inv_freq = 1.0 / (rope["rope_theta"] **
                          (torch.arange(0, rotary_dim, 2, device=ids.device, dtype=torch.float) / rotary_dim))
        inv_freq = inv_freq.repeat(2)
        positions = torch.arange(seq, device=ids.device, dtype=torch.float)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        cos = freqs.cos().unsqueeze(0).to(hidden.dtype)
        sin = freqs.sin().unsqueeze(0).to(hidden.dtype)
        position_embeddings = (cos, sin)
        causal_mask = torch.full((seq, seq), float('-inf'), device=ids.device, dtype=hidden.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        layer_caches = []
        for layer in self.layers:
            hidden, cache = layer.prefill(hidden, position_embeddings, causal_mask)
            layer_caches.append(cache)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits, layer_caches

    def decode(self, ids, layer_caches):
        bsz, seq = ids.shape
        hidden = self.embed_tokens(ids)
        rope = self.config["rope_parameters"]
        head_dim = self.config["head_dim"]
        rotary_dim = int(head_dim * rope["partial_rotary_factor"])
        inv_freq = 1.0 / (rope["rope_theta"] **
                          (torch.arange(0, rotary_dim, 2, device=ids.device, dtype=torch.float) / rotary_dim))
        inv_freq = inv_freq.repeat(2)
        if layer_caches and layer_caches[0] is not None and isinstance(layer_caches[0], dict):
            offset = layer_caches[0]["length"]
        else:
            offset = layer_caches[0][0].shape[2]
        positions = torch.arange(offset, offset + seq, device=ids.device, dtype=torch.float)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        cos = freqs.cos().unsqueeze(0).to(hidden.dtype)
        sin = freqs.sin().unsqueeze(0).to(hidden.dtype)
        position_embeddings = (cos, sin)
        for i, layer in enumerate(self.layers):
            hidden, layer_caches[i] = layer.decode(hidden, position_embeddings, layer_caches[i])
        hidden = self.norm(hidden)
        return self.lm_head(hidden), layer_caches

    def generate(self, ids, max_new_tokens, eos_token_id=None):
        bsz, seq = ids.shape
        generated = ids.clone()
        logits, layer_caches = self.prefill(generated)
        for i in range(max_new_tokens):
            next_logits = logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            if i < max_new_tokens - 1:
                logits, layer_caches = self.decode(next_token, layer_caches)
        return generated
