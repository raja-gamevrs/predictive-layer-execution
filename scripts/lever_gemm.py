"""L3 / L6: GEMM kernels at every Qwen3-4B linear shape, decode (1 row) and prefill (512 rows)
(docs/decode_levers.md). Times one GEMM per shape in a CUDA graph (36 copies of the weight, as in the
model, so nothing stays in L2). Variants: current BF16 x @ W.t(); BF16 with W pre-transposed;
weight-only INT8 (_weight_int8pack_mm, per-channel scales); weight-only INT4 (_weight_int4pack_mm,
group 128); FP8 E4M3 _scaled_mm with per-tensor scales. Reports µs per layer, weight GB/s (decode) or
TFLOP/s (prefill) and relative error against the BF16 product.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

SHAPES = {"qkv": (6144, 2560), "o": (2560, 4096), "gate/up": (19456, 2560), "down": (2560, 9728), "lm head": (151936, 2560)}


def graph_us(fn, reps=20, rounds=5):
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


def int8_pack(w):
    s = w.float().abs().amax(1).clamp(min=1e-8) / 127
    return (w.float() / s[:, None]).round().clamp(-128, 127).to(torch.int8), s.to(w.dtype)


def int4_pack(w, group=128, inner_k_tiles=8):
    out_f, in_f = w.shape
    wg = w.float().view(out_f, in_f // group, group)
    mx, mn = wg.amax(-1), wg.amin(-1)
    scale = ((mx - mn).clamp(min=1e-8) / 15)
    zero = mn + scale * 8  # tinygemm: w = (q - 8) * scale + zero
    q = ((wg - mn[..., None]) / scale[..., None]).round().clamp(0, 15).to(torch.int32).view(out_f, in_f)
    q8 = (q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8)
    packed = torch.ops.aten._convert_weight_to_int4pack(q8, inner_k_tiles)
    sz = torch.stack([scale, zero], -1).transpose(0, 1).contiguous().to(torch.bfloat16)  # (groups, out, 2)
    deq = ((q.float().view(out_f, in_f // group, group) - 8) * scale[..., None] + zero[..., None]).view(out_f, in_f)
    return packed, sz, deq.to(w.dtype)


def fp8_pack(w):
    s = w.float().abs().amax().clamp(min=1e-12) / torch.finfo(torch.float8_e4m3fn).max
    return (w.float() / s).to(torch.float8_e4m3fn), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=36)
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    torch.manual_seed(0)
    rows = []
    for name, (out_f, in_f) in SHAPES.items():
        n = 1 if name == "lm head" else args.copies
        ws = [torch.randn(out_f, in_f, device=dev, dtype=dt) * 0.02 for _ in range(n)]
        wts = [w.t().contiguous() for w in ws]
        i8 = [int8_pack(w) for w in ws]
        i4 = [int4_pack(w) for w in ws]
        f8 = [fp8_pack(w) for w in ws]
        for M in (1, 512):
            x = torch.randn(M, in_f, device=dev, dtype=dt)
            xp = torch.cat([x, x.new_zeros((-M) % 16, in_f)]) if M % 16 else x
            ref = (x.float() @ ws[0].float().t())
            xs = x.float().abs().amax().clamp(min=1e-12) / torch.finfo(torch.float8_e4m3fn).max
            xq = (xp.float() / xs).to(torch.float8_e4m3fn)
            var = {
                "bf16 x@W.t() (current)": (lambda: [x @ w.t() for w in ws], lambda: x @ ws[0].t()),
                "bf16 x@Wt (pre-transposed)": (lambda: [x @ w for w in wts], lambda: x @ wts[0]),
                "int8 weight-only": (lambda: [torch.ops.aten._weight_int8pack_mm(x, q, s) for q, s in i8],
                                     lambda: torch.ops.aten._weight_int8pack_mm(x, *i8[0])),
                "int4 weight-only g128": (lambda: [torch.ops.aten._weight_int4pack_mm(x, p, 128, sz) for p, sz, _ in i4],
                                          lambda: torch.ops.aten._weight_int4pack_mm(x, i4[0][0], 128, i4[0][1])),
                "fp8 per-tensor (act quantised)": (lambda: [torch._scaled_mm(xq, q.t(), scale_a=xs.reshape(()), scale_b=s.reshape(()), out_dtype=dt) for q, s in f8],
                                                   lambda: torch._scaled_mm(xq, f8[0][0].t(), scale_a=xs.reshape(()), scale_b=f8[0][1].reshape(()), out_dtype=dt)[:M]),
            }
            for vname, (fn, one) in var.items():
                try:
                    got = one().float()
                    err = ((got - ref).norm() / ref.norm()).item()
                    us = graph_us(fn) / n
                    r = {"shape": name, "out": out_f, "in": in_f, "M": M, "variant": vname, "us": us, "rel_err": err}
                    if M == 1:
                        r["bf16_weight_gbs"] = out_f * in_f * 2 / us / 1e3
                    else:
                        r["tflops"] = 2 * M * out_f * in_f / us / 1e6
                    rows.append(r)
                    tail = f"{r['bf16_weight_gbs']:7.0f} GB/s-equiv" if M == 1 else f"{r['tflops']:6.1f} TFLOP/s"
                    print(f"{name:8} M={M:<4} {vname:32} {us:9.2f} us  {tail}  err {err:.1e}", flush=True)
                except Exception as e:
                    print(f"{name:8} M={M:<4} {vname:32} unavailable: {str(e).splitlines()[0][:110]}", flush=True)
        del ws, wts, i8, i4, f8
        torch.cuda.empty_cache()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "gemm_kernels.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
