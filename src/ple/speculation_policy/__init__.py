"""Simulated approximate PLE inside a real forward pass, for output-quality measurement.

At each chosen boundary l, block l+1's attention input projections are replaced by
P(N(ĥ)) + correction, with ĥ predicted from h_l and e = N(h) − N(ĥ). P is the projection
module's own forward, so quantized kernels (weight-only, or with activation quantization) behave
as they do in deployment. Everything downstream runs normally on the result, so the effect on the
logits is the true end-to-end effect. Timing here is meaningless: the exact projection input is
available and used to form e. Position 0 (attention-sink token, excluded when fitting predictors)
is always kept exact.
"""

from collections.abc import Callable
from contextlib import contextmanager

import torch

# (module, ŷ, e) -> ỹ. ŷ = module(N(ĥ)); e = N(h) − N(ĥ) in at least FP32.
Correction = Callable[[torch.nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def is_projection(mod: torch.nn.Module) -> bool:
    """A leaf linear-like module: nn.Linear or a quantized replacement exposing in/out features."""
    return (isinstance(getattr(mod, "in_features", None), int) and isinstance(getattr(mod, "out_features", None), int)
            and not any(True for _ in mod.children()))


def attention_input_linears(layer: torch.nn.Module, hidden_size: int) -> list[torch.nn.Module]:
    """Projections inside the attention block that read the normed residual (q/k/v, or fused linear-attention inputs)."""
    out = []
    for name, mod in layer.named_modules():
        leaf = name.split(".")[-1]
        if is_projection(mod) and mod.in_features == hidden_size and "attn" in name and leaf not in ("o_proj", "out_proj"):
            out.append(mod)
    return out


def apply_without_bias(mod: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """mod's linear map without its bias, through mod's own kernel. Hooks are bypassed by calling forward."""
    y = mod.forward(x)
    bias = getattr(mod, "bias", None)
    return y - bias if isinstance(bias, torch.Tensor) else y


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
            y_hat = mod.forward(n_hat)  # speculative projection through the module's own kernel
            wide = torch.promote_types(x.dtype, torch.float32)  # BF16 subtraction would round the difference
            e = x.to(wide) - n_hat.to(wide)
            y = correction(mod, y_hat, e).to(out.dtype)
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
