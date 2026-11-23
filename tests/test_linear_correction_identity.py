import pytest
import torch

from ple.correction import full_correction, low_rank_correction, speculate


@pytest.mark.parametrize("dtype,rtol,atol", [(torch.float32, 1e-5, 1e-5), (torch.bfloat16, 1.6e-2, 5e-2)])
def test_full_correction_matches_exact_projection(dtype, rtol, atol):
    torch.manual_seed(0)
    d_in, d_out, tokens = 256, 384, 17
    weight = torch.randn(d_out, d_in, dtype=dtype) / d_in**0.5
    bias = torch.randn(d_out, dtype=dtype)
    h = torch.randn(tokens, d_in, dtype=dtype)
    h_hat = h + 0.1 * torch.randn_like(h)

    exact = torch.nn.functional.linear(h, weight, bias)
    corrected = full_correction(weight, speculate(weight, h_hat, bias), h, h_hat)

    torch.testing.assert_close(corrected, exact, rtol=rtol, atol=atol)


def test_full_rank_low_rank_correction_is_exact():
    torch.manual_seed(0)
    d = 64
    weight = torch.randn(d, d, dtype=torch.float64)
    u, s, vh = torch.linalg.svd(weight)
    weight_u, weight_v = u * s, vh
    h = torch.randn(5, d, dtype=torch.float64)
    h_hat = h + torch.randn_like(h)

    corrected = low_rank_correction(weight_u, weight_v, speculate(weight, h_hat), h, h_hat)

    torch.testing.assert_close(corrected, torch.nn.functional.linear(h, weight))
