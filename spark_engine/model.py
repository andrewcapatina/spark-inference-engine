import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import rms_norm


def _load_safetensors(path):
    """Load safetensors using the installed safetensors library."""
    from safetensors.torch import load_file
    return load_file(path)


def _apply_rope(q, k, positions, head_dim, theta=10000.0):
    """Apply rotary embeddings to query and key tensors."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    # Llama rotates the two halves of each head, so duplicate the frequency
    # vector by concatenation, then broadcast across attention heads.
    angles = torch.cat((freqs, freqs), dim=-1)  # [batch, sequence, head_dim]
    cos = angles.cos().unsqueeze(1).to(q.dtype)
    sin = angles.sin().unsqueeze(1).to(q.dtype)

    def rotate(x):
        x1, x2 = x[..., :head_dim // 2], x[..., head_dim // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q = q * cos + rotate(q) * sin
    k = k * cos + rotate(k) * sin
    return q, k


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = config.get("rope_theta", 10000.0)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, positions, attention_mask=None, past_key_value=None, use_cache=False):
        bsz, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rope(q, k, positions, self.head_dim, self.rope_theta)

        if past_key_value is not None:
            old_k, old_v = past_key_value
            k = torch.cat([old_k, k], dim=2)
            v = torch.cat([old_v, v], dim=2)

        if self.num_kv_groups > 1:
            k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_expanded = k
            v_expanded = v

        scores = torch.matmul(q, k_expanded.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        output = self.o_proj(attn_output)

        if use_cache:
            return output, (k, v)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.up_proj = nn.Linear(config["hidden_size"], config["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(config["intermediate_size"], config["hidden_size"], bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.Parameter(torch.ones(config["hidden_size"]))
        self.post_attention_layernorm = nn.Parameter(torch.ones(config["hidden_size"]))
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-5)

    def forward(self, hidden_states, positions, attention_mask=None):
        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.input_layernorm, self.rms_norm_eps)
        hidden_states = self.self_attn(hidden_states, positions, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = rms_norm(hidden_states, self.post_attention_layernorm, self.rms_norm_eps)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_hidden_layers"]
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-5)

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.num_layers)])
        self.norm = nn.Parameter(torch.ones(self.hidden_size))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        if config.get("tie_word_embeddings", False):
            self.lm_head.weight = self.embed_tokens.weight

    def load_hf_state_dict(self, state_dict):
        mapping = state_dict

        self.embed_tokens.weight.data.copy_(mapping["model.embed_tokens.weight"])
        for i in range(self.num_layers):
            layer = self.layers[i]
            prefix = f"model.layers.{i}."
            layer.input_layernorm.data.copy_(mapping[prefix + "input_layernorm.weight"])
            layer.self_attn.q_proj.weight.data.copy_(mapping[prefix + "self_attn.q_proj.weight"])
            layer.self_attn.k_proj.weight.data.copy_(mapping[prefix + "self_attn.k_proj.weight"])
            layer.self_attn.v_proj.weight.data.copy_(mapping[prefix + "self_attn.v_proj.weight"])
            layer.self_attn.o_proj.weight.data.copy_(mapping[prefix + "self_attn.o_proj.weight"])
            layer.post_attention_layernorm.data.copy_(mapping[prefix + "post_attention_layernorm.weight"])
            layer.mlp.gate_proj.weight.data.copy_(mapping[prefix + "mlp.gate_proj.weight"])
            layer.mlp.up_proj.weight.data.copy_(mapping[prefix + "mlp.up_proj.weight"])
            layer.mlp.down_proj.weight.data.copy_(mapping[prefix + "mlp.down_proj.weight"])
        self.norm.data.copy_(mapping["model.norm.weight"])
        if not self.config.get("tie_word_embeddings", False):
            self.lm_head.weight.data.copy_(mapping["lm_head.weight"])

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        hidden_states = self.embed_tokens(input_ids)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            hidden_states = layer(hidden_states, positions, causal_mask)
        hidden_states = rms_norm(hidden_states, self.norm, self.rms_norm_eps)
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(cls, directory):
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(config)
        weights = _load_safetensors(os.path.join(directory, "model.safetensors"))
        model.load_hf_state_dict(weights)
        return model.eval()
