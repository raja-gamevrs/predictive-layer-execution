"""Where does a Qwen3-4B decode step (and a 512-token prefill) spend its time? (docs/decode_levers.md, step 1)

Each component is captured on its own as a CUDA graph over all 36 layers at real shapes and timed
over back-to-back replays, so it runs under the same launch conditions as inside the full step.
The full step is also traced with torch.profiler to compare summed kernel time with wall time.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from ple.data import wikitext_chunks
from ple.decode import Qwen3Decoder
from ple.model_loader import load, load_config, run_name

PEAK_GBS = 1792.0


def graph_ms(fn, reps=50, rounds=5):
    """Median over `rounds` of the mean ms per replay of a CUDA graph of fn, over `reps` replays."""
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
        out.append(a.elapsed_time(b) / reps)
    return statistics.median(out[1:])


def components(dec: Qwen3Decoder, T: int, ctx: int):
    """Callables, each doing one component for all layers, at T rows with ctx cached positions."""
    d, nh, nkv, hd, n = dec.d, dec.nh, dec.nkv, dec.hd, len(dec.layers)
    dev, dt = dec.dev, dec.dtype
    h = torch.randn(T, d, device=dev, dtype=dt)
    qkv = torch.randn(T, (nh + 2 * nkv) * hd, device=dev, dtype=dt)
    att = torch.randn(T, nh * hd, device=dev, dtype=dt)
    gu = torch.randn(T, 2 * dec.layers[0]["down"].shape[1], device=dev, dtype=dt)
    act = torch.randn(T, dec.layers[0]["down"].shape[1], device=dev, dtype=dt)
    ids = torch.randint(0, dec.embed.shape[0], (T,), device=dev)
    pos = torch.arange(ctx - T, ctx, device=dev)
    mask = torch.full((1, 1, T, dec.max_len), float("-inf"), device=dev, dtype=dt)
    for r in range(T):
        mask[..., r, :ctx - T + r + 1] = 0
    q = torch.randn(1, nh, T, hd, device=dev, dtype=dt)
    L = dec.layers
    qn, kn = L[0]["qn"], L[0]["kn"]

    def rope_norm(i):
        qq, kk, vv = qkv.split([nh * hd, nkv * hd, nkv * hd], dim=-1)
        qq = dec.rms(qq.view(T, nh, hd), L[i]["qn"]).transpose(0, 1)
        kk = dec.rms(kk.view(T, nkv, hd), L[i]["kn"]).transpose(0, 1)
        cos, sin = dec.cos[pos][None], dec.sin[pos][None]
        return qq * cos + dec._rot(qq) * sin, kk * cos + dec._rot(kk) * sin, vv.view(T, nkv, hd).transpose(0, 1)

    kv_k = torch.randn(nkv, T, hd, device=dev, dtype=dt)
    comps = {
        "embed": lambda: dec.embed[ids],
        "norms (ln1, ln2, final)": lambda: [dec.rms(h, L[i][w]) for i in range(n) for w in ("ln1", "ln2")] + [dec.rms(h, dec.final_norm)],
        "qkv GEMM": lambda: [h @ L[i]["qkv"].t() for i in range(n)],
        "q/k norm + RoPE": lambda: [rope_norm(i) for i in range(n)],
        "KV cache write": lambda: [(dec.k_cache[i, 0].index_copy_(1, pos, kv_k), dec.v_cache[i, 0].index_copy_(1, pos, kv_k)) for i in range(n)],
        "attention (masked, max_len)": lambda: [F.scaled_dot_product_attention(q, dec.k_cache[i], dec.v_cache[i], attn_mask=mask, enable_gqa=True) for i in range(n)],
        "attention (valid length, no mask)": lambda: [F.scaled_dot_product_attention(q, dec.k_cache[i][:, :, :ctx], dec.v_cache[i][:, :, :ctx], is_causal=T > 1, enable_gqa=True) for i in range(n)],
        "o GEMM": lambda: [att @ L[i]["o"].t() for i in range(n)],
        "gate/up GEMM": lambda: [h @ L[i]["gu"].t() for i in range(n)],
        "SiLU x up": lambda: [F.silu(a) * b for a, b in [gu.chunk(2, dim=-1)] for _ in range(n)],
        "down GEMM": lambda: [act @ L[i]["down"].t() for i in range(n)],
        "residual adds": lambda: [h + h for _ in range(2 * n)],
        "LM head": lambda: h[-1:] @ dec.lm_head.t() if T > 1 else h @ dec.lm_head.t(),
        "argmax": lambda: (h @ dec.lm_head.t()[:, :1]).argmax(-1),
    }
    W = lambda key: sum(l[key].numel() for l in L) * 2  # noqa: E731
    bytes_ = {"qkv GEMM": W("qkv"), "o GEMM": W("o"), "gate/up GEMM": W("gu"), "down GEMM": W("down"),
              "LM head": dec.lm_head.numel() * 2,
              "attention (masked, max_len)": n * 2 * dec.k_cache[0].numel() * 2,
              "attention (valid length, no mask)": n * 2 * nkv * ctx * hd * 2}
    flops = {"qkv GEMM": 2 * T * W("qkv") / 2, "o GEMM": 2 * T * W("o") / 2, "gate/up GEMM": 2 * T * W("gu") / 2,
             "down GEMM": 2 * T * W("down") / 2}
    return comps, bytes_, flops


KERNEL_GROUPS = [("int4 GEMM", r"int4|tinygemm"), ("BF16 GEMM", r"gemv|gemm|cutlass.*gemm|sm\d+_xmma|Kernel2"),
                 ("attention", r"fmha|attention|cudnn|flash|softmax|bmm"), ("norm", r"rms_norm|norm"),
                 ("KV write / index", r"index|scatter|gather|copy"), ("argmax", r"argmax|reduce"),
                 ("elementwise", r"elementwise|vectorized|unrolled|roll|silu|add|mul")]


def kernel_group(name):
    import re
    for g, pat in KERNEL_GROUPS:
        if re.search(pat, name, re.I):
            return g
    return "other"


def full_step(dec, prompt, w4=False):
    first = int(dec.prefill(prompt, w4=w4)[-1].argmax())
    replay, reset, _ = dec.make_step(prompt.shape[0], first, w4=w4)
    times = []
    for _ in range(6):
        dec.prefill(prompt, w4=w4)
        reset()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(128):
            replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 128)
    # kernel trace of 20 replays
    from torch.profiler import ProfilerActivity, profile
    dec.prefill(prompt, w4=w4)
    reset()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            replay()
        torch.cuda.synchronize()
    kern = [e for e in prof.events() if e.device_type.name == "CUDA"]
    busy_ms = sum(e.device_time for e in kern) / 1000 / 20
    names = {}
    for e in kern:
        names[e.name] = names.get(e.name, 0) + e.device_time / 1000 / 20
    top = sorted(names.items(), key=lambda kv: -kv[1])
    return statistics.median(times[1:]), busy_ms, len(kern) // 20, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--out", default="results/processed/decode_levers")
    ap.add_argument("--kernels-only", action="store_true",
                    help="follow-up C: kernel trace of the L1 + L2 decode step (and the INT4 draft step), grouped")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = Qwen3Decoder(model, args.prompt + 128 + 16)
    prompt = wikitext_chunks(tok, "train", args.prompt, 1, skip=400)[0].to(dec.dev)
    res = {"model": cfg["model"]["id"], "prompt": args.prompt, "peak_gbs": PEAK_GBS}
    if args.kernels_only:
        dec.attn_impl, dec.rope_impl = "fast", "fused"
        out = {"model": cfg["model"]["id"], "prompt": args.prompt, "steps": []}
        with torch.inference_mode():
            dec.quantize_int4()
            for label, w4 in (("L1 + L2 decode step", False), ("INT4 draft step (RTN g128)", True)):
                step_ms, busy, nk, top = full_step(dec, prompt, w4)
                groups = {}
                for name, ms in top:
                    g = kernel_group(name)
                    groups[g] = groups.get(g, 0) + ms
                out["steps"].append({"step": label, "wall_ms": step_ms, "kernel_busy_ms": busy, "kernels": nk,
                                     "groups": groups, "kernels_by_name": top})
                print(f"{label}: {step_ms:.3f} ms wall, {busy:.3f} ms busy, {nk} kernels")
                for g, ms in sorted(groups.items(), key=lambda kv: -kv[1]):
                    print(f"   {ms:7.3f} ms  {g}")
                for name, ms in top[:12]:
                    print(f"      {ms:7.3f} ms  {name[:100]}")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"{run_name(cfg)}_profile_fast.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return
    with torch.inference_mode():
        step_ms, busy, nk, top = full_step(dec, prompt)
        top = top[:15]
        res["decode_step"] = {"wall_ms": step_ms, "kernel_busy_ms": busy, "kernels": nk, "top_kernels": top}
        print(f"decode step {step_ms:.3f} ms wall, {busy:.3f} ms kernels busy, {nk} kernels; idle {step_ms - busy:.3f} ms")
        for name, ms in top:
            print(f"   {ms:7.3f} ms  {name[:110]}")
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        pre = []
        for _ in range(6):
            torch.cuda.synchronize(); a.record(); dec.prefill(prompt); b.record(); torch.cuda.synchronize()
            pre.append(a.elapsed_time(b))
        res["prefill_ms_eager"] = statistics.median(pre[1:])
        print(f"prefill {args.prompt} tokens (eager, as used): {res['prefill_ms_eager']:.2f} ms")
        for label, T in (("decode", 1), ("prefill", args.prompt)):
            comps, by, fl = components(dec, T, args.prompt + (1 if T == 1 else 0))
            rows = []
            for name, fn in comps.items():
                ms = graph_ms(fn, reps=50 if T == 1 else 10)
                r = {"component": name, "ms": ms}
                if name in by and T == 1:
                    r["gbs"] = by[name] / ms / 1e6
                    r["peak_frac"] = r["gbs"] / PEAK_GBS
                if name in fl and T > 1:
                    r["tflops"] = fl[name] / ms / 1e9
                rows.append(r)
                extra = (f"  {r['gbs']:7.0f} GB/s ({100 * r['peak_frac']:.0f}% of peak)" if "gbs" in r else "") + \
                        (f"  {r['tflops']:6.1f} TFLOP/s" if "tflops" in r else "")
                print(f"{label:8} {name:36} {ms:8.3f} ms{extra}", flush=True)
            res[label] = rows
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_name(cfg)}_profile.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
