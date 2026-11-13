import torch
from spark_engine.ops import rms_norm


def test_basic_shape():
    x = torch.randn(2, 4, 8)
    w = torch.ones(8)
    out = rms_norm(x, w)
    assert out.shape == x.shape
    assert out.dtype == torch.float32


def test_unit_weight_norms_to_one():
    x = torch.randn(3, 10)
    w = torch.ones(10)
    out = rms_norm(x, w)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(3), atol=1e-5)


def test_weight_scales():
    x = torch.randn(2, 5)
    w = torch.full((5,), 2.0)
    out = rms_norm(x, w)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full((2,), 2.0), atol=1e-5)


def test_noncontiguous():
    x = torch.randn(4, 8).t()
    w = torch.ones(4)
    out = rms_norm(x, w)
    assert out.shape == x.shape


def test_zeros():
    x = torch.zeros(2, 6)
    w = torch.ones(6)
    out = rms_norm(x, w)
    assert torch.allclose(out, torch.zeros_like(x))


def test_input_preserved():
    x = torch.randn(2, 4)
    x_copy = x.clone()
    w = torch.ones(4)
    rms_norm(x, w)
    assert torch.equal(x, x_copy)
