import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_row_kernel(
    x_ptr,
    x_q_ptr,
    scale_a_ptr,
    rows,
    width,
    padded_rows,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = (row < rows) & (cols < width)
    x_row = x_ptr + row * width
    x_q_row = x_q_ptr + row * width
    scale_row = scale_a_ptr + row

    vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    max_abs = tl.max(tl.abs(vals), axis=0)
    scale_work = max_abs * 0.0022321429569274187
    scale_work = tl.maximum(scale_work, 1e-6)

    nan_count = tl.sum((vals != vals).to(tl.int32), axis=0)
    if nan_count > 0:
        nan_bits = tl.full((), 0x7fffffff, dtype=tl.uint32)
        scale_work = nan_bits.to(tl.float32, bitcast=True)

    scaled = tl.div_rn(vals, scale_work)
    q = scaled.to(tl.float8e4nv)
    tl.store(x_q_row + cols, q, mask=cols < width)

    if row < rows:
        tl.store(scale_row, scale_work)
    else:
        tl.store(scale_row, 0.0)


def prepare_fp8(x2d):
    if not torch.is_inference_mode_enabled():
        return None
    if not isinstance(x2d, torch.Tensor):
        return None
    if x2d.dim() != 2:
        return None
    if x2d.dtype != torch.bfloat16:
        return None
    if not x2d.is_cuda:
        return None
    if not x2d.is_contiguous():
        return None
    if x2d.requires_grad:
        return None

    rows, width = x2d.shape
    if rows < 16:
        return None
    if width not in (5120, 6144):
        return None

    padded_rows = ((rows + 15) // 16) * 16
    if padded_rows * width > 2147483647:
        return None

    x_q = torch.empty((padded_rows, width), dtype=torch.float8_e4m3fn, device=x2d.device)
    scale_a = torch.empty((padded_rows, 1), dtype=torch.float32, device=x2d.device)

    BLOCK = triton.next_power_of_2(width)
    grid = (padded_rows,)
    _fp8_row_kernel[grid](
        x2d,
        x_q,
        scale_a,
        rows,
        width,
        padded_rows,
        BLOCK=BLOCK,
        num_warps=8,
        enable_fp_fusion=False,
    )

    return x_q, scale_a
