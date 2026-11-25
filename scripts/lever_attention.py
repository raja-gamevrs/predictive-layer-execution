"""L2: decode / verify attention kernels at Qwen3-4B shapes (docs/decode_levers.md).

Fixed shapes so every variant can sit in a CUDA graph: the cache is max_len long and a mask hides
unfilled positions. Variants: SDPA with each available backend; a hand-written GQA attention from
batched matmuls + softmax (q grouped per KV head). Each is checked against the math reference.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def graph_ms(fn, reps=200, rounds=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    out = []
    for _ in range(rounds + 1):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / reps * 1000)
    return statistics.median(out[1:])


def gqa_bmm(q, k, v, mask):
    """q (1, H, T, D), k/v (1, KV, L, D), mask (1, 1, T, L) additive -> (1, H, T, D)."""
    _, H, T, D = q.shape
    KV = k.shape[1]
    g = H // KV
    qg = q.view(KV, g * T, D)                       # heads h = kv*g + j -> rows (j, t)
    s = torch.baddbmm(mask[0].expand(KV, T, -1).repeat(1, g, 1), qg, k[0].transpose(1, 2), alpha=D ** -0.5)
    p = torch.softmax(s.float(), -1).to(q.dtype)
    return torch.bmm(p, v[0]).view(1, H, T, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=656)
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    H, KV, D, L = 32, 8, 128, args.max_len
    torch.manual_seed(0)
    k = torch.randn(1, KV, L, D, device=dev, dtype=dt)
    v = torch.randn(1, KV, L, D, device=dev, dtype=dt)
    rows = []
    for T, ctx in ((1, 513), (1, 640), (3, 515), (5, 517), (7, 519)):
        q = torch.randn(1, H, T, D, device=dev, dtype=dt)
        mask = torch.full((1, 1, T, L), float("-inf"), device=dev, dtype=dt)
        for r in range(T):
            mask[..., r, :ctx - T + r + 1] = 0
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), attn_mask=mask.float(), enable_gqa=True)
        variants = {}
        for be in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH):
            def f(be=be):
                with sdpa_kernel(be):
                    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            variants[f"sdpa {be.name.lower()}"] = f
        variants["gqa bmm (hand-written)"] = lambda: gqa_bmm(q, k, v, mask)
        for name, f in variants.items():
            try:
                out = f()
                err = ((out.float() - ref).norm() / ref.norm()).item()
                us = graph_ms(f)
                rows.append({"T": T, "ctx": ctx, "variant": name, "us_per_layer": us, "rel_err": err})
                print(f"T={T} ctx={ctx} {name:28} {us:8.2f} us/layer  x36 = {us * 36 / 1000:6.3f} ms  rel err {err:.2e}", flush=True)
            except Exception as e:  # backend not available for these inputs
                print(f"T={T} ctx={ctx} {name:28} unavailable: {str(e).splitlines()[0][:100]}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "attention_kernels.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
