"""Qwen3.5 low-level operations. CUDA tensors, no reference library calls."""
import torch


def qwen_rms_norm(x, weight, eps=1e-6):
    """Qwen offset-weight RMSNorm: (x * w).to(dtype) with w = 1 + offset."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * (1.0 + weight.float())).to(dtype)
