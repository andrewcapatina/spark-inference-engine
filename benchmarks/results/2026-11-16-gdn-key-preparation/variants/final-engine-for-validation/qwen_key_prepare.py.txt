"""Exact pointwise GDN key preparation without RHS packing or solve changes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _key_prepare(k, beta, decay, weighted, decayed, total,
                 ks: tl.constexpr, bs: tl.constexpr, es: tl.constexpr,
                 H: tl.constexpr, T: tl.constexpr,
                 D: tl.constexpr, TIME_MAJOR: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    feature = offsets % D
    row = offsets // D
    if TIME_MAJOR:
        head = row % H
        time = row // H % T
    else:
        time = row % T
        head = row // T % H
    batch = row // (H * T)
    key = tl.load(k + batch * ks[0] + head * ks[1] + time * ks[2] + feature,
                  mask, other=0).to(tl.float32)
    b = tl.load(beta + batch * bs[0] + head * bs[1] + time * bs[2], mask, other=0).to(tl.float32)
    e = tl.load(decay + batch * es[0] + head * es[1] + time * es[2], mask, other=0).to(tl.float32)
    w = key * b
    d = w * e
    tl.store(weighted + offsets, w, mask)
    tl.store(decayed + offsets, d, mask)


def _key_layout(k, beta):
    """Validate stable metadata once; never inspect tensor values."""
    assert torch.is_inference_mode_enabled()
    assert k.ndim == 4 and beta.ndim == 3
    assert k.is_cuda and beta.is_cuda and k.device == beta.device
    assert k.dtype == beta.dtype == torch.float32
    batch, heads, time, width = k.shape
    assert heads > 0 and time > 0 and width > 0
    assert beta.shape == k.shape[:-1]
    assert k.stride(-1) == 1
    time_major = k.stride(1) == width and k.stride(2) == heads * width
    head_major = k.stride(2) == width and k.stride(1) >= time * width
    assert time_major or head_major
    assert k.numel() <= 2147483647
    for tensor in (k, beta):
        assert all(s >= 0 for s in tensor.stride())
        assert sum(max(0, size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())) <= 2147483647
    return time_major


def _make_key_preparer(k, beta, chunk_size):
    """Internal inference plan for fixed-size slices of the validated inputs.

    Only the caller's own K/beta slices and freshly computed FP32 decay may be
    supplied. Captures metadata, not the full tensors or mutable output buffers.
    """
    time_major = _key_layout(k, beta)
    batch, heads, time, width = k.shape
    assert 0 < chunk_size <= time and time % chunk_size == 0
    shape = (batch, chunk_size, heads, width) if time_major else (batch, heads, chunk_size, width)
    total = batch * heads * chunk_size * width
    device, ks, bs = k.device, k.stride(), beta.stride()
    launch = _key_prepare[(triton.cdiv(total, 512),)]

    def prepare(k_chunk, beta_chunk, exp_decay):
        weighted = torch.empty(shape, dtype=torch.float32, device=device)
        decayed = torch.empty_like(weighted)
        if total:
            launch(k_chunk, beta_chunk, exp_decay, weighted, decayed, total,
                   ks, bs, exp_decay.stride(), H=heads, T=chunk_size, D=width,
                   TIME_MAJOR=time_major, BLOCK=512, num_warps=4, enable_fp_fusion=False)
        if time_major:
            return weighted.transpose(1, 2), decayed.transpose(1, 2)
        return weighted, decayed

    return prepare


def prepare_keys(k, beta, exp_decay):
    """Checked standalone API preserving original pointwise output layouts."""
    assert exp_decay.ndim == 3 and exp_decay.is_cuda
    assert exp_decay.dtype == torch.float32 and exp_decay.device == k.device
    assert exp_decay.shape == k.shape[:-1]
    assert all(s >= 0 for s in exp_decay.stride())
    assert sum(max(0, n - 1) * s for n, s in zip(exp_decay.shape, exp_decay.stride())) <= 2147483647
    return _make_key_preparer(k, beta, k.shape[-2])(k, beta, exp_decay)
