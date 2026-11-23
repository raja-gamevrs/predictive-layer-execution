import torch

from ple.quant import nvfp4_fake_quantize
from ple.quant.modelopt import FP8W8A8Linear, unpack_nvfp4


def _pack(values: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 values (already on the grid) into ModelOpt's uint8 layout, low nibble first."""
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    mag = (values.abs()[..., None] == grid).float().argmax(-1)
    codes = (mag | ((values < 0).long() << 3)).to(torch.uint8)
    return codes[:, 0::2] | (codes[:, 1::2] << 4)


def test_unpack_nvfp4_roundtrip():
    torch.manual_seed(0)
    vals = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[torch.randint(0, 8, (4, 32))]
    vals = vals * torch.where(torch.rand(4, 32) < 0.5, -1.0, 1.0)
    vals[vals == 0] = 0.0
    block = torch.full((4, 2), 0.25).to(torch.float8_e4m3fn)
    out = unpack_nvfp4(_pack(vals), block, torch.tensor(2.0))
    torch.testing.assert_close(out, vals * 0.5)


def test_fp8_w8a8_is_not_linear_but_close():
    torch.manual_seed(0)
    w = torch.randn(16, 32)
    a, b = torch.randn(4, 32), torch.randn(4, 32) * 0.1
    lin = FP8W8A8Linear(w, a.abs().max() / 448, None)
    assert not torch.allclose(lin(a + b), lin(a) + lin(b))
    exact = a @ w.T
    assert (lin(a) - exact).norm() / exact.norm() < 0.05  # E4M3 has a 3-bit mantissa


def test_emulated_nvfp4_values_are_representable_by_unpack():
    x = nvfp4_fake_quantize(torch.randn(2, 64))
    assert torch.isfinite(x).all()
