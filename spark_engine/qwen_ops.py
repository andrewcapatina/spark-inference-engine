"""Qwen3.5 low-level operations. CUDA tensors, no reference library calls."""
import torch


def gated_rms_norm(x, weight, z, eps=1e-6):
    """Qwen3.5 gated RMSNorm: norm(x) * weight * silu(z)."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight * x.to(dtype)
    z = torch.nn.functional.silu(z.to(torch.float32))
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


def gated_delta_chunk(q, k, v, g, beta, initial_state=None, chunk_size=64):
    """Chunked gated delta rule matching official Qwen3.5 prefill arithmetic.

    Layout: q/k/v [batch,time,heads,feature]; g/beta [batch,time,heads];
    state [batch,heads,key_dim,value_dim]. Pads time to chunk_size, computes
    intra-chunk transforms with FP32 triangular solves, scans chunks, and
    casts only the output back to the input dtype.
    """
    dtype = q.dtype
    bsz, time, nvh, feature = q.shape
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
    # Pad time to multiple of chunk_size
    pad = (chunk_size - (time % chunk_size)) % chunk_size
    if pad > 0:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        beta = torch.nn.functional.pad(beta, (0, pad))
        g = torch.nn.functional.pad(g, (0, pad))
    padded_time = time + pad
    n_chunks = padded_time // chunk_size
    # Official uses cumulative log-decay differences for pairwise decay
    cum_decay = g.cumsum(dim=-1)
    # Intra-chunk computation
    out = torch.zeros_like(v)
    if initial_state is None:
        state = torch.zeros((bsz, nvh, k.shape[-1], v.shape[-1]), dtype=v.dtype, device=v.device)
    else:
        state = initial_state.to(v)
    for c in range(n_chunks):
        start = c * chunk_size
        end = start + chunk_size
        q_c = q[:, :, start:end]
        k_c = k[:, :, start:end]
        v_c = v[:, :, start:end]
        beta_c = beta[:, :, start:end]
        g_c = g[:, :, start:end]
        # Compute cumulative decay independently within each chunk
        cum_decay_c = g_c.cumsum(dim=-1)
        # Pairwise decay: exp(cum_decay[j] - cum_decay[i]) for j <= i
        pairwise_decay = cum_decay_c.unsqueeze(-1) - cum_decay_c.unsqueeze(-2)
        # Causal mask: strictly upper triangle gets -inf
        mask = torch.tril(torch.ones(chunk_size, chunk_size, device=q.device))
        pairwise_decay = pairwise_decay.masked_fill(mask == 0, float('-inf'))
        pairwise_decay = pairwise_decay.exp()
        # Auxiliary system: K^T Q with decay
        ut_system = (k_c * beta_c.unsqueeze(-1)) @ k_c.transpose(-2, -1) * pairwise_decay
        # Intra-chunk attention: Q K^T with decay
        intra_chunk_attn = q_c @ k_c.transpose(-2, -1) * pairwise_decay
        # Solve auxiliary system (unitriangular)
        # v_new = solve(ut_system, v_c * beta_c)
        # k_cumdecay = solve(ut_system, k_c * cum_decay.exp())
        v_beta = v_c * beta_c.unsqueeze(-1)
        decayed_k = k_c * cum_decay_c.exp().unsqueeze(-1)
        # Use torch.linalg.solve_triangular for unitriangular system
        v_new = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
        k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k, upper=False, unitriangular=True)
        # Inter-chunk attention with previous state (apply within-chunk decay to query)
        inter_chunk_attn = (q_c * cum_decay_c.exp().unsqueeze(-1)) @ state
        # Correct intra-chunk values with carried state
        v_corrected = v_new - k_cumdecay @ state
        # Compute output for chunk
        out_c = inter_chunk_attn + intra_chunk_attn @ v_corrected
        out[:, :, start:end] = out_c
        # Update state for next chunk
        # State propagation: state_{end} = state_{start} * exp(-cum_decay[end]) + k_c^T @ v_corrected
        # where k_c is decayed by remaining within-chunk decay
        chunk_decay = cum_decay_c[:, :, -1].exp()
        decayed_k_c = k_c * (cum_decay_c[:, :, -1:] - cum_decay_c).exp().unsqueeze(-1)
        state = state * chunk_decay.unsqueeze(-1).unsqueeze(-1) + decayed_k_c.transpose(-2, -1) @ v_corrected
    # Trim padding
    out = out[:, :, :time]
    # [batch,heads,time,feature] -> [batch,time,heads,feature]
    return out.transpose(1, 2).contiguous().to(dtype), state


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
