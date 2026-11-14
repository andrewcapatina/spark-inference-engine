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


class QwenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenCausalLM(nn.Module):
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
                          (torch.arange(0, rotary_dim, 2, device=ids.device, dtype=torch.float) / head_dim))
        inv_freq_full = torch.repeat_interleave(inv_freq, 2, dim=0)
        positions = torch.arange(seq, device=ids.device, dtype=torch.float)
        freqs = positions.unsqueeze(1) * inv_freq_full.unsqueeze(0)
        cos = freqs.cos().unsqueeze(0)
        sin = freqs.sin().unsqueeze(0)
        position_embeddings = (cos, sin)
        for layer in self.layers:
            hidden = layer(hidden, position_embeddings)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)
