import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import exp as libdevice_exp

@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    silu = tl.div_rn(gate, 1.0 + libdevice_exp(-gate))
    silu_bf16 = silu.to(tl.bfloat16)
    result = silu_bf16.to(tl.float32) * up
    tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)

def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if not (gate.is_cuda and up.is_cuda and gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16):
        return torch.nn.functional.silu(gate) * up
    if gate.shape != up.shape or not gate.is_contiguous() or not up.is_contiguous():
        return torch.nn.functional.silu(gate) * up
    if not torch.is_inference_mode_enabled():
        return torch.nn.functional.silu(gate) * up
    if gate.device != up.device:
        return torch.nn.functional.silu(gate) * up
    n = gate.numel()
    if n == 0:
        return torch.empty_like(gate)
    out = torch.empty_like(gate)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _silu_mul_kernel[grid](gate, up, out, n, BLOCK=BLOCK, enable_fp_fusion=False)
    return out
