import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sensitivity_atlas import install  # noqa: E402


class Attn(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = torch.nn.Linear(d, d)

    def forward(self, hidden_states, **_):
        return self.proj(hidden_states), None


class Layer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn, self.mlp = Attn(d), torch.nn.Linear(d, d)

    def forward(self, h):
        h = h + self.self_attn(hidden_states=h)[0]
        return h + self.mlp(h)


def test_skip_keeps_position_zero_and_removes_the_op():
    torch.manual_seed(0)
    layer, h = Layer(8), torch.randn(1, 5, 8)
    with torch.no_grad():
        exact = layer(h)
        handle = install(layer, "mlp", "skip", None)
        got = layer(h)
        handle.remove()
        no_mlp = h + layer.self_attn(hidden_states=h)[0]
        torch.testing.assert_close(got[:, 1:], no_mlp[:, 1:])
        torch.testing.assert_close(got[:, 0], exact[:, 0])
        handle = install(layer, "block", "skip", None)
        got = layer(h)
        handle.remove()
        torch.testing.assert_close(got[:, 1:], h[:, 1:])
        torch.testing.assert_close(got[:, 0], exact[:, 0])
        handle = install(layer, "attn", "skip", None)
        got = layer(h)
        handle.remove()
        torch.testing.assert_close(got[:, 1:], (h + layer.mlp(h))[:, 1:])
