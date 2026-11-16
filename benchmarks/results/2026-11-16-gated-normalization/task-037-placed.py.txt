import torch
import triton
import triton.language as tl

@triton.jit
def _gated_norm_tail_kernel(x_ptr, inv_ptr, w_ptr, silu_ptr, out_ptr,
                            total, width: tl.constexpr, block: tl.constexpr, wtype: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < total
    row = offs // width
    col = offs % width
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(inv_ptr + row, mask=mask, other=0.0).to(tl.float32)
    normed = (x * inv).to(wtype)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(wtype)
    weighted = (w.to(tl.float32) * normed.to(tl.float32)).to(wtype)
    silu = tl.load(silu_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (weighted.to(tl.float32) * silu).to(wtype)
    tl.store(out_ptr + offs, out, mask=mask)

def gated_norm_tail(x_fp32, inverse_rms, weight, silu_fp32):
    assert torch.is_inference_mode_enabled()
    assert x_fp32.is_cuda and inverse_rms.is_cuda and weight.is_cuda and silu_fp32.is_cuda
    assert x_fp32.device == inverse_rms.device == weight.device == silu_fp32.device
    assert x_fp32.is_contiguous() and inverse_rms.is_contiguous()
    assert weight.is_contiguous() and silu_fp32.is_contiguous()
    assert x_fp32.dtype == torch.float32 and x_fp32.dim() >= 1
    assert inverse_rms.dtype == torch.float32
    assert weight.dtype == torch.bfloat16 and weight.dim() == 1
    assert silu_fp32.dtype == torch.float32 and silu_fp32.shape == x_fp32.shape
    width = x_fp32.shape[-1]
    assert width > 0
    assert weight.numel() == width
    total = x_fp32.numel()
    assert total <= 2147483647
    assert inverse_rms.numel() == total // width
    out = torch.empty_like(x_fp32, dtype=torch.bfloat16)
    if total == 0:
        return out
    block = 1024
    grid = (triton.cdiv(total, block),)
    _gated_norm_tail_kernel[grid](x_fp32, inverse_rms, weight, silu_fp32, out,
                                  total, width, block, tl.bfloat16,
                                  num_warps=4, enable_fp_fusion=False)
    return out
