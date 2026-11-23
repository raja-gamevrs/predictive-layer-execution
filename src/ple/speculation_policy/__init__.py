"""Simulated approximate PLE inside a real forward pass, for output-quality measurement.

At each chosen boundary l, block l+1's attention input projections are replaced by
W·N(ĥ) + correction(W, e), with ĥ predicted from h_l and e = N(h) − N(ĥ). Everything downstream
runs normally on the result, so the effect on the logits is the true end-to-end effect.
Timing here is meaningless: the exact projection input is available and used to form e.
Position 0 (attention-sink token, excluded when fitting predictors) is always kept exact.
"""

from collections.abc import Callable
from contextlib import contextmanager

import torch

Correction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]  # (W, ŷ, e) -> ỹ; e is ≥ FP32


def attention_input_linears(layer: torch.nn.Module, hidden_size: int) -> list[torch.nn.Linear]:
    """Linears inside the attention block that read the normed residual (q/k/v, or fused linear-attention inputs)."""
    out = []
    for name, mod in layer.named_modules():
        leaf = name.split(".")[-1]
        if (isinstance(mod, torch.nn.Linear) and mod.in_features == hidden_size and "attn" in name
                and leaf not in ("o_proj", "out_proj")):
            out.append(mod)
    return out


@contextmanager
def speculate_attention_inputs(layers: torch.nn.ModuleList, boundaries: list[int], predictor: dict[int, Callable],
                               correction_for: Callable[[int], Correction], hidden_size: int):
    """Install hooks for the duration of the context; always removes them."""
    handles, stash = [], {}

    def keep_input(l):
        def hook(_m, args, kwargs):
            stash[l] = args[0] if args else kwargs["hidden_states"]
        return hook

    def replace_output(l):
        correction = correction_for(l)

        def hook(mod, args, out):
            x = args[0]  # exact normed input N(h_(l+1))
            h_hat = predictor[l](stash[l])
            n_hat = layers[l + 1].input_layernorm(h_hat).to(x.dtype)
            y_hat = torch.nn.functional.linear(n_hat, mod.weight, mod.bias)
            wide = torch.promote_types(x.dtype, torch.float32)  # BF16 subtraction would round the difference
            e = x.to(wide) - n_hat.to(wide)
            y = correction(mod.weight, y_hat, e).to(out.dtype)
            y[:, :1] = out[:, :1]
            return y
        return hook

    try:
        for l in boundaries:
            handles.append(layers[l].register_forward_pre_hook(keep_input(l), with_kwargs=True))
            for lin in attention_input_linears(layers[l + 1], hidden_size):
                handles.append(lin.register_forward_hook(replace_output(l)))
        yield
    finally:
        for h in handles:
            h.remove()
