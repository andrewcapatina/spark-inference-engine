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

    def forward(self, x):
        return qwen_rms_norm(x, self.weight, self.eps)


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
    """Qwen3.5 linear-attention layer with gated delta rule."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_k_heads = config["linear_num_key_heads"]
        self.num_v_heads = config["linear_num_value_heads"]
        self.k_dim = config["linear_key_head_dim"]
        self.v_dim = config["linear_value_head_dim"]
        self.conv_kernel = config["linear_conv_kernel_dim"]
        self.q_proj = nn.Linear(self.hidden_size, self.num_k_heads * self.k_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_k_heads * self.k_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_v_heads * self.v_dim, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.conv1d = nn.Conv1d(
            self.hidden_size, self.hidden_size, kernel_size=self.conv_kernel,
            groups=self.hidden_size, padding=self.conv_kernel - 1, bias=False,
        )
        self.o_proj = nn.Linear(self.num_v_heads * self.v_dim, self.hidden_size, bias=False)

    def forward(self, hidden):
        bsz, seq, _ = hidden.shape
        conv = F.silu(self.conv1d(hidden.transpose(1, 2))[:, :, :seq]).transpose(1, 2)
        q = self.q_proj(conv).view(bsz, seq, self.num_k_heads, self.k_dim)
        k = self.k_proj(conv).view(bsz, seq, self.num_k_heads, self.k_dim)
        v = self.v_proj(conv).view(bsz, seq, self.num_v_heads, self.v_dim)
        g = self.g_proj(conv)
        beta = torch.sigmoid(self.beta_proj(conv))
        out, _ = gated_delta_rule(q, k, v, g, beta)
        return self.o_proj(out.view(bsz, seq, -1))
