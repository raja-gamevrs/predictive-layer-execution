import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_fused_qkv import rms  # noqa: E402


def test_folded_gain_matches_norm_then_project():
    torch.manual_seed(0)
    d, width, t = 64, 48, 5
    norms = [torch.nn.RMSNorm(d, eps=1e-6) for _ in range(3)]
    for n in norms:
        n.weight.data = torch.rand(d) + 0.5
    weights = [torch.randn(width, d) for _ in norms]
    h = torch.randn(t, d)
    ref = [torch.nn.functional.linear(rms(h, n), w) for n, w in zip(norms, weights)]
    fused = torch.cat([w * n.weight[None, :] for n, w in zip(norms, weights)], 0)
    got = torch.nn.functional.linear(rms(h, norms[0], gain=False), fused).split(width, dim=-1)
    for g, r in zip(got, ref):
        torch.testing.assert_close(g, r, rtol=1e-5, atol=1e-5)
