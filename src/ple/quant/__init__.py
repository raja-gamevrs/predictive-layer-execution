"""NVFP4 numerics, emulated: for quality studies on GPUs/platforms without an NVFP4 kernel.

NVFP4 (as in NVIDIA ModelOpt / Blackwell): values in FP4 E2M1 {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6},
one FP8 E4M3 scale per 16 consecutive elements along the reduction dimension, and one FP32 global
scale per tensor chosen so the largest block scale fits E4M3 (max 448). Dequantised values are
q · s_block · s_global. Rounding is to nearest, ties to the larger magnitude (kernels may round ties
to even; the difference affects only exact midpoints).

The matmul runs in BF16 on the dequantised operands, so results match a real NVFP4 kernel up to
accumulation order, not bit for bit. Speed is not representative.
"""

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX, FP8_MAX, BLOCK = 6.0, 448.0, 16


def nvfp4_fake_quantize(x: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Quantise-dequantise along the last dimension (must be a multiple of `block`); returns x's dtype."""
    if x.shape[-1] % block:
        raise ValueError(f"last dimension {x.shape[-1]} is not a multiple of {block}")
    xf = x.float()
    blocks = xf.reshape(*xf.shape[:-1], -1, block)
    amax = blocks.abs().amax(-1, keepdim=True)
    global_scale = (xf.abs().amax() / (FP4_MAX * FP8_MAX)).clamp_min(torch.finfo(torch.float32).tiny)
    block_scale = (amax / FP4_MAX / global_scale).to(torch.float8_e4m3fn).float()
    scale = block_scale * global_scale
    scaled = torch.where(scale > 0, blocks / scale, torch.zeros_like(blocks))
    grid = E2M1.to(x.device)
    mids = (grid[1:] + grid[:-1]) / 2
    q = grid[torch.bucketize(scaled.abs().clamp_max(FP4_MAX), mids, right=True)] * scaled.sign()
    return (q * scale).reshape(xf.shape).to(x.dtype)


class NVFP4EmulatedLinear(torch.nn.Module):
    """Linear with NVFP4-quantised weights; optionally NVFP4-quantised activations (W4A4)."""

    def __init__(self, linear: torch.nn.Linear, quantize_activations: bool):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.quantize_activations = quantize_activations
        self.weight = torch.nn.Parameter(nvfp4_fake_quantize(linear.weight.data), requires_grad=False)
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quantize_activations:
            x = nvfp4_fake_quantize(x)
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"nvfp4 {'W4A4' if self.quantize_activations else 'W4A16'} (emulated)"


def emulate_nvfp4(layers: torch.nn.ModuleList, quantize_activations: bool) -> int:
    """Replace every nn.Linear inside the decoder blocks (embeddings and lm_head stay BF16). Returns the count."""
    count = 0
    for block in layers:
        for name, mod in list(block.named_modules()):
            if isinstance(mod, torch.nn.Linear) and mod.in_features % BLOCK == 0:
                parent = block.get_submodule(name.rpartition(".")[0]) if "." in name else block
                setattr(parent, name.rpartition(".")[2], NVFP4EmulatedLinear(mod, quantize_activations))
                count += 1
    return count
