"""Per-layer ridge regression h_l → Δ_l = h_(l+1) − h_l, fitted from streamed activation traces."""

import torch


class RidgeAccumulator:
    """Accumulates the normal equations in FP32 so traces never need to be held in memory."""

    def __init__(self, dim: int, device: torch.device):
        self.xtx = torch.zeros(dim + 1, dim + 1, dtype=torch.float64, device=device)
        self.xty = torch.zeros(dim + 1, dim, dtype=torch.float64, device=device)
        self.count = 0

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # Per-batch products in FP32 (consumer GPUs are slow at FP64); the running sums stay FP64.
        x1 = torch.cat([x.float(), torch.ones(x.shape[0], 1, device=x.device)], dim=1)
        self.xtx += (x1.T @ x1).double()
        self.xty += (x1.T @ y.float()).double()
        self.count += x.shape[0]

    def solve(self, ridge: float = 1e-3) -> "LinearDeltaPredictor":
        d = self.xtx.shape[0] - 1
        lam = ridge * torch.trace(self.xtx[:d, :d]) / d
        reg = torch.eye(d + 1, dtype=torch.float64, device=self.xtx.device) * lam
        reg[d, d] = 0  # do not shrink the bias
        coef = torch.linalg.solve(self.xtx + reg, self.xty)
        return LinearDeltaPredictor(coef[:d].float(), coef[d].float())


class LinearDeltaPredictor:
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.weight, self.bias = weight, bias  # weight: (d, d) applied as h @ weight

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        """ĥ_(l+1) = h_l + Δ̂_l, computed in the stored weight's dtype (FP32 unless converted with `to`)."""
        w = self.weight
        return h + (h.to(w.dtype) @ w + self.bias.to(w.dtype)).to(h.dtype)

    def to(self, dtype: torch.dtype) -> "LinearDeltaPredictor":
        return LinearDeltaPredictor(self.weight.to(dtype), self.bias.to(dtype))

    def truncated(self, rank: int) -> "LinearDeltaPredictor":
        """Best rank-`rank` approximation of the weight (SVD truncation), same bias."""
        u, s, vh = torch.linalg.svd(self.weight)
        return LinearDeltaPredictor((u[:, :rank] * s[:rank]) @ vh[:rank], self.bias)
