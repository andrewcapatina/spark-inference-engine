"""Qwen3.5 low-level operations. CUDA tensors, no reference library calls."""
import torch


def gated_rms_norm(x, weight, z, eps=1e-6):
    """Qwen3.5 gated RMSNorm: norm(x) * weight * silu(z)."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = (x * weight.float()).to(dtype)
    z = torch.nn.functional.silu(z.float())
    return (x * z).to(dtype)


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

    cos/sin shape: [batch, seq, rotary_dim]. q/k shape: [batch, heads, seq, head_dim].
    Only the first rotary_dim features are rotated; the suffix is preserved.
    """
    # cos/sin: [batch, seq, rotary_dim] -> [batch, 1, seq, rotary_dim] for broadcast
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    q_out = torch.cat([q_embed, q_pass], dim=-1)
    k_out = torch.cat([k_embed, k_pass], dim=-1)
    return q_out, k_out


def _l2norm(x, dim=-1, eps=1e-6):
    inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv


def gated_delta_rule(q, k, v, g, beta, initial_state=None):
    """Gated delta rule linear attention (pure PyTorch recurrence).

    Layout: q/k/v [batch,time,heads,feature]; g/beta [batch,time,heads];
    state [batch,heads,key_dim,value_dim]. Transpose to [batch,heads,time,feature]
    for the recurrence, then transpose back.
    """
    dtype = q.dtype
    # [batch,time,heads,feature] -> [batch,heads,time,feature]
    q = q.transpose(1, 2).float()
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()
    # [batch,time,heads] -> [batch,heads,time]
    beta = beta.transpose(1, 2).float()
    g = g.transpose(1, 2).float()
    q = _l2norm(q, dim=-1)
    k = _l2norm(k, dim=-1)
    q = q / (q.shape[-1] ** 0.5)
    bsz, nvh, seq, vdim = v.shape
    if initial_state is None:
        state = torch.zeros((bsz, nvh, k.shape[-1], vdim), dtype=v.dtype, device=v.device)
    else:
        state = initial_state.to(v)
    out = torch.zeros_like(v)
    for i in range(seq):
        decay = g[:, :, i].exp()[..., None, None]
        state = state * decay
        kv_mem = (state * k[:, :, i].unsqueeze(-1)).sum(dim=-2)
        delta = (v[:, :, i] - kv_mem) * beta[:, :, i].unsqueeze(-1)
        state = state + k[:, :, i].unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q[:, :, i].unsqueeze(-1)).sum(dim=-2)
    # [batch,heads,time,feature] -> [batch,time,heads,feature]
    return out.transpose(1, 2).contiguous().to(dtype), state
