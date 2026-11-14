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

    def _make_rope(self, seq, device):
        rope = self.config["rope_parameters"]
        head_dim = self.config["head_dim"]
        rotary_dim = int(head_dim * rope["partial_rotary_factor"])
        inv_freq = 1.0 / (rope["rope_theta"] **
                          (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float) / rotary_dim))
        inv_freq = inv_freq.repeat(2)
        positions = torch.arange(seq, device=device, dtype=torch.float)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        return freqs.cos().unsqueeze(0), freqs.sin().unsqueeze(0)

    def forward(self, ids):
        bsz, seq = ids.shape
        hidden = self.embed_tokens(ids)
        cos, sin = self._make_rope(seq, ids.device)
        causal_mask = torch.full((seq, seq), float('-inf'), device=ids.device, dtype=hidden.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        for layer in self.layers:
            hidden = layer(hidden, (cos, sin), causal_mask)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    def prefill(self, ids):
        bsz, seq = ids.shape
        hidden = self.embed_tokens(ids)
        cos, sin = self._make_rope(seq, ids.device)
        causal_mask = torch.full((seq, seq), float('-inf'), device=ids.device, dtype=hidden.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        cache = []
        for layer in self.layers:
            if layer.layer_type == "full_attention":
                qg = layer.self_attn.q_proj(hidden).view(bsz, seq, layer.self_attn.num_heads, layer.self_attn.head_dim * 2)
                q, gate = qg[..., :layer.self_attn.head_dim], qg[..., layer.self_attn.head_dim:]
                k = layer.self_attn.k_proj(hidden).view(bsz, seq, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                v = layer.self_attn.v_proj(hidden).view(bsz, seq, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                q = layer.self_attn.q_norm(q).transpose(1, 2)
                k = layer.self_attn.k_norm(k).transpose(1, 2)
                v = v.transpose(1, 2)
                q, k = apply_partial_rope(q, k, cos, sin)
                if layer.self_attn.num_groups > 1:
                    k = k.repeat_interleave(layer.self_attn.num_groups, dim=1)
                    v = v.repeat_interleave(layer.self_attn.num_groups, dim=1)
                scores = torch.matmul(q, k.transpose(-2, -1)) * layer.self_attn.scaling
                scores = scores + causal_mask
                attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                out = torch.matmul(attn, v).transpose(1, 2)
                out = out * torch.sigmoid(gate)
                out = out.reshape(bsz, seq, -1)
                out = layer.self_attn.o_proj(out)
                cache.append({"type": "attn", "k": k, "v": v})
            else:
                out = layer.linear_attn(hidden)
                cache.append({"type": "linear", "state": layer.linear_attn._last_state})
            hidden = layer.input_layernorm(hidden)
            hidden = hidden + out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits, cache

    def decode(self, new_ids, cache):
        bsz = new_ids.shape[0]
        hidden = self.embed_tokens(new_ids)
        seq = 1
        cos, sin = self._make_rope(seq, new_ids.device)
        new_cache = []
        for i, layer in enumerate(self.layers):
            if layer.layer_type == "full_attention":
                qg = layer.self_attn.q_proj(hidden).view(bsz, seq, layer.self_attn.num_heads, layer.self_attn.head_dim * 2)
                q, gate = qg[..., :layer.self_attn.head_dim], qg[..., layer.self_attn.head_dim:]
                k = layer.self_attn.k_proj(hidden).view(bsz, seq, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                v = layer.self_attn.v_proj(hidden).view(bsz, seq, layer.self_attn.num_kv_heads, layer.self_attn.head_dim)
                q = layer.self_attn.q_norm(q).transpose(1, 2)
                k = layer.self_attn.k_norm(k).transpose(1, 2)
                v = v.transpose(1, 2)
                q, k = apply_partial_rope(q, k, cos, sin)
                prev = cache[i]
                k_full = torch.cat([prev["k"], k], dim=-1)
                v_full = torch.cat([prev["v"], v], dim=-1)
                if layer.self_attn.num_groups > 1:
                    k_full = k_full.repeat_interleave(layer.self_attn.num_groups, dim=1)
                    v_full = v_full.repeat_interleave(layer.self_attn.num_groups, dim=1)
                scores = torch.matmul(q, k_full.transpose(-2, -1)) * layer.self_attn.scaling
                attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                out = torch.matmul(attn, v_full).transpose(1, 2)
                out = out * torch.sigmoid(gate)
                out = out.reshape(bsz, seq, -1)
                out = layer.self_attn.o_proj(out)
                new_cache.append({"type": "attn", "k": k_full, "v": v_full})
            else:
                out = layer.linear_attn(hidden)
                new_cache.append({"type": "linear", "state": layer.linear_attn._last_state})
            hidden = layer.input_layernorm(hidden)
            hidden = hidden + out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits, new_cache
