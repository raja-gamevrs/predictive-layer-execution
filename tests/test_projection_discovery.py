"""Projection discovery must also find quantized replacements of nn.Linear, and bias handling must be exact."""

import torch

from ple.speculation_policy import apply_without_bias, attention_input_linears, is_projection


class FakeQuantLinear(torch.nn.Module):
    """Stands in for NVFP4Linear / CompressedLinear: not an nn.Linear, but exposes in/out features."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.in_features, self.out_features = d_in, d_out
        self.weight = torch.nn.Parameter(torch.randn(d_out, d_in))
        self.bias = torch.nn.Parameter(torch.randn(d_out))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = FakeQuantLinear(d, d)
        self.self_attn.o_proj = FakeQuantLinear(d, d)
        self.mlp = torch.nn.Linear(d, 2 * d)


def test_finds_quantized_projection_and_skips_output_and_mlp():
    block = Block(8)
    assert attention_input_linears(block, 8) == [block.self_attn.q_proj]
    assert not is_projection(block.self_attn)


def test_apply_without_bias_is_the_linear_part():
    mod = FakeQuantLinear(8, 5)
    x = torch.randn(3, 8)
    torch.testing.assert_close(apply_without_bias(mod, x), x @ mod.weight.T)
