"""Correction of speculative projections: y = W ĥ + W e, plus structured approximations of W e."""

import torch


def speculate(weight: torch.Tensor, h_hat: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Speculative projection ŷ = ĥ Wᵀ + b, using the predicted activation."""
    return torch.nn.functional.linear(h_hat, weight, bias)


def full_correction(weight: torch.Tensor, y_hat: torch.Tensor, h: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    """Exact reconciliation y = ŷ + W(h − ĥ). Costs as much as a projection; the reference for structured corrections."""
    return y_hat + torch.nn.functional.linear(h - h_hat, weight)


def low_rank_correction(
    weight_u: torch.Tensor, weight_v: torch.Tensor, y_hat: torch.Tensor, h: torch.Tensor, h_hat: torch.Tensor
) -> torch.Tensor:
    """Approximate reconciliation y ≈ ŷ + U(Vᵀ e) with W ≈ U Vᵀ of rank r. Never exact unless r is full rank."""
    e = h - h_hat
    return y_hat + torch.nn.functional.linear(torch.nn.functional.linear(e, weight_v), weight_u)
