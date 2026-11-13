"""Qwen3.5 low-level operations. CUDA tensors, no reference library calls."""
import torch


def qwen_rms_norm(x, weight, eps=1e-6):
    """Qwen offset-weight RMSNorm: (x * w).to(dtype) with w = 1 + offset."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * (1.0 + weight.float())).to(dtype)


def _rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(q, k, cos, sin):
    """Apply rotary position embedding to the leading rotary width only.

    cos/sin shape: [seq, rotary_dim]. q/k shape: [batch, seq, heads, head_dim].
    Only the first rotary_dim features are rotated; the suffix is preserved.
    """
    cos = cos.unsqueeze(1).unsqueeze(2)  # [seq, 1, 1, rotary_dim]
    sin = sin.unsqueeze(1).unsqueeze(2)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_out = torch.cat([q_embed, q_pass], dim=-1)
    k_out = torch.cat([k_embed, k_pass], dim=-1)
    return q_out, k_out
