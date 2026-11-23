"""Fit per-layer predictors and error bases from streamed activation traces."""

from collections.abc import Callable, Iterable

import torch

from ple.predictors.linear import LinearDeltaPredictor, RidgeAccumulator

States = list[torch.Tensor]  # [h_0, ..., h_L] for one sequence, positions on dim 0


def fit_linear(traces: Iterable[States], layers: Iterable[int], dim: int, device) -> dict[int, LinearDeltaPredictor]:
    layers = list(layers)
    acc = {l: RidgeAccumulator(dim, device) for l in layers}
    for hs in traces:
        for l in layers:
            acc[l].add(hs[l].float(), (hs[l + 1] - hs[l]).float())
    return {l: a.solve() for l, a in acc.items()}


def fit_error_bases(traces: Iterable[States], layers: Iterable[int], norms, predictors: dict[str, dict[int, Callable]],
                    dim: int, rank: int, device) -> dict[tuple[int, str], torch.Tensor]:
    """Top-`rank` principal directions (dim, rank) of e = N_(l+1)(h_(l+1)) − N_(l+1)(ĥ_(l+1)) per layer and predictor."""
    layers = list(layers)
    cov = {(l, p): torch.zeros(dim, dim, dtype=torch.float64, device=device) for l in layers for p in predictors}
    with torch.inference_mode():
        for hs in traces:
            for l in layers:
                n_true = norms[l + 1](hs[l + 1]).float()
                for p, pred in predictors.items():
                    e = n_true - norms[l + 1](pred[l](hs[l])).float()
                    cov[(l, p)] += (e.T @ e).double()
    return {k: torch.linalg.eigh(c)[1].flip(1)[:, :rank].float().contiguous() for k, c in cov.items()}
