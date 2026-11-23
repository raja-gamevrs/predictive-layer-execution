import torch

from ple.quant import E2M1, NVFP4EmulatedLinear, nvfp4_fake_quantize


def test_values_lie_on_scaled_e2m1_grid_and_quantization_is_idempotent():
    torch.manual_seed(0)
    x = torch.randn(4, 64) * 3
    q = nvfp4_fake_quantize(x)
    torch.testing.assert_close(nvfp4_fake_quantize(q), q)
    for row in q.reshape(-1, 16):
        scale = row.abs().max() / 6
        if scale > 0:
            mags = (row.abs() / scale).unique()
            assert all(torch.isclose(m, E2M1).any() for m in mags)


def test_relative_error_is_plausible_for_fp4():
    torch.manual_seed(0)
    x = torch.randn(256, 256)
    err = (nvfp4_fake_quantize(x) - x).norm() / x.norm()
    assert 0.05 < err < 0.2


def test_weight_only_is_linear_in_input_and_w4a4_is_not():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32, bias=False)
    a, b = torch.randn(5, 64), torch.randn(5, 64)
    w4a16 = NVFP4EmulatedLinear(lin, quantize_activations=False)
    torch.testing.assert_close(w4a16(a + b), w4a16(a) + w4a16(b), rtol=1e-5, atol=1e-5)
    w4a4 = NVFP4EmulatedLinear(lin, quantize_activations=True)
    assert not torch.allclose(w4a4(a + b), w4a4(a) + w4a4(b), rtol=1e-3, atol=1e-3)
