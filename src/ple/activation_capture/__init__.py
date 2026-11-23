"""Capture the residual stream at every decoder-block boundary with forward hooks.

Hooks are used instead of `output_hidden_states` because the last hidden state there has the
final norm applied, so it is not the raw output of the last block.
"""

import torch


def _first_tensor(x):
    return x[0] if isinstance(x, (tuple, list)) else x


@torch.inference_mode()
def capture_residual_stream(model: torch.nn.Module, layers: torch.nn.ModuleList, **inputs) -> list[torch.Tensor]:
    """Run one exact forward pass; return [h_0, ..., h_L], where h_0 is the input to block 0 and h_(i+1) its output."""
    states: list[torch.Tensor] = []
    handles = [layers[0].register_forward_pre_hook(
        lambda _m, args, kwargs: states.append(_first_tensor(args) if args else kwargs["hidden_states"]),
        with_kwargs=True,
    )]
    for layer in layers:
        handles.append(layer.register_forward_hook(lambda _m, _a, out: states.append(_first_tensor(out))))
    try:
        model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if len(states) != len(layers) + 1:
        raise RuntimeError(f"captured {len(states)} states for {len(layers)} layers")
    return [s.detach() for s in states]
