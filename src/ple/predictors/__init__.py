"""Zero-parameter next-state predictors over the residual stream (docs/techniques.md, Approach A).

Each takes the states of preceding layers, newest last, and returns ĥ_(l+1).
"""

import torch


def identity(history: list[torch.Tensor]) -> torch.Tensor:
    """ĥ_(l+1) = h_l."""
    return history[-1]


def residual_extrapolation(history: list[torch.Tensor]) -> torch.Tensor:
    """ĥ_(l+1) = h_l + (h_l − h_(l-1)): the next update repeats the previous one."""
    return velocity(history, alpha=1.0, beta=0.0)


def velocity(history: list[torch.Tensor], alpha: float = 1.0, beta: float = 0.0) -> torch.Tensor:
    """ĥ_(l+1) = h_l + α v_l + β v_(l-1), with v_l = h_l − h_(l-1). Needs 2 states, or 3 when β ≠ 0."""
    needed = 3 if beta else 2
    if len(history) < needed:
        raise ValueError(f"velocity with beta={beta} needs {needed} states, got {len(history)}")
    h = history[-1]
    out = h + alpha * (h - history[-2])
    if beta:
        out = out + beta * (history[-2] - history[-3])
    return out
