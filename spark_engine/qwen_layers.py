"""Qwen3.5 attention and linear-attention layers. Same named weights as official."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from spark_engine.qwen_ops import qwen_rms_norm, apply_partial_rope, gated_delta_rule


class QwenRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, z=None):
        if z is None:
            return qwen_rms_norm(x, self.weight, self.eps)
        from spark_engine.qwen_ops import gated_rms_norm
        return gated_rms_norm(x, self.weight, z, self.eps)


class QwenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config["head_dim"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.num_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config["hidden_size"], self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config["hidden_size"], self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config["hidden_size"], bias=False)
        self.q_norm = QwenRMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = QwenRMSNorm(self.head_dim, eps=1e-6)

    def prefill(self, hidden, position_embeddings, attention_mask=None):
        bsz, seq, _ = hidden.shape
        qg = self.q_proj(hidden).view(bsz, seq, self.num_heads, self.head_dim * 2)
        q, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        k = self.k_proj(hidden).view(bsz, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(bsz, seq, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_partial_rope(q, k, cos, sin)
        cache_k, cache_v = k, v
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2)
        out = out * torch.sigmoid(gate)
        out = out.reshape(bsz, seq, -1)
        return self.o_proj(out), (cache_k, cache_v)

    def forward(self, hidden, position_embeddings, attention_mask=None):
        bsz, seq, _ = hidden.shape
        qg = self.q_proj(hidden).view(bsz, seq, self.num_heads, self.head_dim * 2)
        q, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
        k = self.k_proj(hidden).view(bsz, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(bsz, seq, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_partial_rope(q, k, cos, sin)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2)
        out = out * torch.sigmoid(gate)
        out = out.reshape(bsz, seq, -1)
        return self.o_proj(out)


class QwenGatedDeltaNet(nn.Module):
    """Qwen3.5 linear-attention layer with gated delta rule. Official weight names."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_k_heads = config["linear_num_key_heads"]
        self.num_v_heads = config["linear_num_value_heads"]
        self.k_dim = config["linear_key_head_dim"]
        self.v_dim = config["linear_value_head_dim"]
        self.key_dim = self.num_k_heads * self.k_dim
        self.value_dim = self.num_v_heads * self.v_dim
        self.conv_kernel = config["linear_conv_kernel_dim"]
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=self.conv_kernel,
            groups=self.conv_dim, padding=self.conv_kernel - 1, bias=False,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = QwenRMSNorm(self.v_dim, eps=1e-6)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden):
        bsz, seq, _ = hidden.shape
        mixed = self.in_proj_qkv(hidden).transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed)[:, :, :seq]).transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(bsz, seq, self.num_k_heads, self.k_dim)
        k = k.view(bsz, seq, self.num_k_heads, self.k_dim)
        v = v.view(bsz, seq, self.num_v_heads, self.v_dim)
        z = self.in_proj_z(hidden).view(bsz, seq, self.num_v_heads, self.v_dim)
        beta = torch.sigmoid(self.in_proj_b(hidden))
        a = self.in_proj_a(hidden)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        out, _ = gated_delta_rule(q, k, v, g, beta)
        out = out.view(-1, self.v_dim)
        z = z.view(-1, self.v_dim)
        out = self.norm(out, z)
        out = out.view(bsz, seq, -1)
        return self.out_proj(out)

    def prefill(self, hidden):
        bsz, seq, _ = hidden.shape
        mixed = self.in_proj_qkv(hidden).transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed)[:, :, :seq]).transpose(1, 2)
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(bsz, seq, self.num_k_heads, self.k_dim)
        k = k.view(bsz, seq, self.num_k_heads, self.k_dim)
        v = v.view(bsz, seq, self.num_v_heads, self.v_dim)
        z = self.in_proj_z(hidden).view(bsz, seq, self.num_v_heads, self.v_dim)
        beta = torch.sigmoid(self.in_proj_b(hidden))
        a = self.in_proj_a(hidden)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            k = k.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        out, _ = gated_delta_rule(q, k, v, g, beta)
        out = out.view(-1, self.v_dim)
        z = z.view(-1, self.v_dim)
        out = self.norm(out, z)
        out = out.view(bsz, seq, -1)
        out = self.out_proj(out)
        conv_state = torch.zeros(bsz, self.conv_dim, self.conv_kernel, device=hidden.device, dtype=hidden.dtype)
        conv_state[:, :, :seq] = self.in_proj_qkv(hidden).transpose(1, 2)[:, :, :seq]
        recurrent_state = {
            "k": torch.zeros(bsz, self.num_v_heads, self.k_dim, device=hidden.device, dtype=hidden.dtype),
            "v": torch.zeros(bsz, self.num_v_heads, self.v_dim, device=hidden.device, dtype=hidden.dtype),
        }
        cache = {"conv_state": conv_state, "recurrent_state": recurrent_state, "length": seq}
        return out, cache
