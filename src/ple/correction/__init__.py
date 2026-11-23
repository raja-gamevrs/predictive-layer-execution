"""Correction of speculative projections: y = W ĥ + W e, plus structured approximations of W e."""

import torch


def speculate(weight: torch.Tensor, h_hat: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Speculative projection ŷ = ĥ Wᵀ + b, using the predicted activation."""
    return torch.nn.functional.linear(h_hat, weight, bias)


def full_correction(weight: torch.Tensor, y_hat: torch.Tensor, h: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """Exact reconciliation y = ŷ + W(h − ĥ). Costs as much as a projection; the reference for structured corrections."""
    return y_hat + torch.nn.functional.linear(h - h_hat, weight)


def basis_correction(w_basis: torch.Tensor, basis: torch.Tensor, y_hat: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """y ≈ ŷ + (W U)(Uᵀ e) for a fixed orthonormal basis U (d, r) with W U precomputed (d_out, r).

    Cost per token r·(d + d_out) MACs instead of d·d_out. Exact only for errors inside span(U).
    """
    return y_hat + torch.nn.functional.linear(e @ basis, w_basis)


def channel_topk_correction(weight: torch.Tensor, y_hat: torch.Tensor, e: torch.Tensor, k: int,
                            col_norms: torch.Tensor | None = None) -> torch.Tensor:
    """y ≈ ŷ + W[:, S] e_S, with S the k input channels per token scoring highest on |e_i|·‖W[:, i]‖.

    Cost per token k·d_out MACs plus a gather. Computed here with a masked dense product for analysis.
    """
    score = e.abs() if col_norms is None else e.abs() * col_norms
    idx = score.topk(k, dim=-1).indices
    sparse = torch.zeros_like(e).scatter_(-1, idx, e.gather(-1, idx))
    return y_hat + torch.nn.functional.linear(sparse, weight)


def token_selective_correction(weight: torch.Tensor, y_hat: torch.Tensor, e: torch.Tensor, fraction: float) -> torch.Tensor:
    """Exact correction for the `fraction` of tokens with the largest ‖e‖; the rest keep ŷ. Cost: fraction of a projection."""
    n = max(1, round(fraction * e.shape[0])) if fraction > 0 else 0
    out = y_hat.clone()
    if n:
        rows = e.norm(dim=-1).topk(n).indices
        out[rows] += torch.nn.functional.linear(e[rows], weight)
    return out


def low_rank_correction(
    weight_u: torch.Tensor, weight_v: torch.Tensor, y_hat: torch.Tensor, h: torch.Tensor, h_hat: torch.Tensor
) -> torch.Tensor:
    """Approximate reconciliation y ≈ ŷ + U(Vᵀ e) with W ≈ U Vᵀ of rank r. Never exact unless r is full rank."""
    e = h - h_hat
    return y_hat + torch.nn.functional.linear(torch.nn.functional.linear(e, weight_v), weight_u)
