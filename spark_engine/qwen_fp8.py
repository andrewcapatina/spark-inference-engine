import torch

class FP8Linear(torch.nn.Module):
    def __init__(self, weight, weight_scale):
        super().__init__()
        assert weight.dtype == torch.float8_e4m3fn, "weight must be float8_e4m3fn"
        assert weight.dim() == 2, "weight must be 2D [N,K]"
        N, K = weight.shape
        assert N % 16 == 0 and K % 16 == 0, "N,K must be multiples of 16"
        assert weight_scale.shape == (N, 1), "weight_scale must be [N,1]"
        assert weight_scale.dtype in (torch.bfloat16, torch.float32), "scale dtype"
        assert torch.all(torch.isfinite(weight_scale)), "scale must be finite"
        assert torch.all(weight_scale > 0), "scale must be positive"
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", weight_scale.to(torch.float32))

    def forward(self, x):
            assert x.dtype == torch.bfloat16, "input must be bfloat16"
            assert x.is_cuda, "input must be on CUDA"
            assert x.shape[-1] == self.weight.shape[1], "K mismatch"
            orig_shape = x.shape
            x2d = x.reshape(-1, x.shape[-1])
            M, K = x2d.shape
            prepared = None
            if M >= 16 and torch.is_inference_mode_enabled():
                try:
                    from spark_engine.qwen_fp8_prepare import prepare_fp8
                except ModuleNotFoundError as e:
                    if e.name != "triton" and not (e.name or "").startswith("triton."):
                        raise
                else:
                    prepared = prepare_fp8(x2d)
            if prepared is None:
                scale_a = torch.clamp(torch.amax(x2d.abs(), dim=-1, keepdim=True).float() / 448.0, min=1e-6)
                x_q = (x2d / scale_a).to(torch.float8_e4m3fn)
                if not x_q.is_contiguous():
                    x_q = x_q.contiguous()
                pad = (16 - M % 16) % 16
                if pad:
                    x_q = torch.nn.functional.pad(x_q, (0, 0, 0, pad))
                    scale_a = torch.nn.functional.pad(scale_a, (0, 0, 0, pad))
            else:
                x_q, scale_a = prepared
                pad = (-M) % 16
            scale_b = self.weight_scale.t()
            out = torch._scaled_mm(x_q, self.weight.t(), scale_a, scale_b,
                                   out_dtype=torch.bfloat16, use_fast_accum=True)
            if pad:
                out = out[:M]
            return out.reshape(orig_shape[:-1] + (self.weight.shape[0],))
