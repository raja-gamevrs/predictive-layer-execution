import torch

from ple.correction import basis_correction, channel_topk_correction, speculate, token_selective_correction


def _setup(tokens=12, d=48, d_out=40):
    torch.manual_seed(0)
    w = torch.randn(d_out, d, dtype=torch.float64)
    h = torch.randn(tokens, d, dtype=torch.float64)
    h_hat = h + 0.3 * torch.randn_like(h)
    return w, h, h_hat, h - h_hat, speculate(w, h_hat), torch.nn.functional.linear(h, w)


def test_full_basis_is_exact_and_empty_basis_is_uncorrected():
    w, h, h_hat, e, y_hat, exact = _setup()
    full = torch.linalg.qr(torch.randn(48, 48, dtype=torch.float64)).Q
    torch.testing.assert_close(basis_correction(w @ full, full, y_hat, e), exact)
    empty = full[:, :0]
    torch.testing.assert_close(basis_correction(w @ empty, empty, y_hat, e), y_hat)


def test_basis_correction_exact_for_errors_in_span():
    w, h, h_hat, _, _, exact = _setup()
    basis = torch.linalg.qr(torch.randn(48, 5, dtype=torch.float64)).Q
    h_hat = h - torch.randn(12, 5, dtype=torch.float64) @ basis.T
    torch.testing.assert_close(basis_correction(w @ basis, basis, speculate(w, h_hat), h - h_hat), exact)


def test_channel_topk_all_channels_is_exact():
    w, *_, e, y_hat, exact = _setup()
    torch.testing.assert_close(channel_topk_correction(w, y_hat, e, k=48), exact)


def test_channel_topk_error_does_not_grow_with_k():
    w, *_, e, y_hat, exact = _setup()
    errs = [(channel_topk_correction(w, y_hat, e, k) - exact).norm().item() for k in (1, 8, 24, 48)]
    assert errs[-1] < 1e-9 and errs[0] > errs[-1]


def test_token_selective_fixes_exactly_the_chosen_rows():
    w, *_, e, y_hat, exact = _setup()
    out = token_selective_correction(w, y_hat, e, fraction=0.25)
    worst = set(e.norm(dim=-1).topk(3).indices.tolist())
    for i in range(12):
        torch.testing.assert_close(out[i], exact[i] if i in worst else y_hat[i])
    torch.testing.assert_close(token_selective_correction(w, y_hat, e, 1.0), exact)
    torch.testing.assert_close(token_selective_correction(w, y_hat, e, 0.0), y_hat)
