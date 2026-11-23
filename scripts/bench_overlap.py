"""Stage E microbenchmark: does launching block l+1's QKV early, on a second CUDA stream, shorten the critical path?

Paths timed on one boundary l, with real activations and the exact mask/rotary arguments
the model passes to block l (captured with a hook during a real forward pass):
  block      block l alone
  qkv        N_(l+1) and q/k/v projections of block l+1 alone
  sequential block l, then N_(l+1), then q/k/v                      (exact baseline)
  overlap_identity   stream 2: N(h_l) → q/k/v   ‖  stream 1: block l   (ĥ = h_l, no correction)
  overlap_linear     stream 2: N(ĥ), ĥ = h_l + h_l A + b (BF16 GEMM) → q/k/v ‖ stream 1: block l
Upper bound on saving: sequential − max(block, qkv path). Each path: warm-up, then repeated runs
timed with CUDA events on the main stream (which waits for stream 2), reporting median, p95, CV.
--graphs captures each path (both streams) into a CUDA graph and times replays, removing CPU
launch overhead. No correction kernels in either mode.
"""

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import torch

from ple.data import wikitext_chunks
from ple.model_loader import decoder_layers, load, load_config, run_name
from ple.speculation_policy import attention_input_linears


def timed(fn, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def as_graph(fn):
    """Capture fn (which may fork onto a side stream and join back) into a CUDA graph; return its replay."""
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(warm)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def summary(times: list[float]) -> dict:
    s = sorted(times)
    med = statistics.median(s)
    return {"median_ms": med, "p95_ms": s[min(len(s) - 1, int(0.95 * len(s)))],
            "cv": statistics.pstdev(s) / statistics.mean(s), "n": len(s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--boundary", type=int, default=26)
    ap.add_argument("--tokens", type=int, nargs="+", default=[128, 512, 2048, 8192])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--graphs", action="store_true")
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    dev = torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    l = args.boundary
    block, nxt = layers[l], layers[l + 1]
    norm, projs = nxt.input_layernorm, attention_input_linears(nxt, d)
    a = (torch.randn(d, d, device=dev, dtype=model.dtype) * d**-0.5)  # timing only; values irrelevant
    b = torch.zeros(d, device=dev, dtype=model.dtype)
    side = torch.cuda.Stream()
    text = torch.cat(wikitext_chunks(tok, "test", 512, 64))

    gpu_state = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu",
                                "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    results = []
    with torch.inference_mode():
        for n_tok in args.tokens:
            ids = text[:n_tok][None].to(dev)
            captured = {}

            def grab(_m, a_, kw):
                captured["h"], captured["kw"] = (a_[0] if a_ else kw.pop("hidden_states")), dict(kw)
            hook = block.register_forward_pre_hook(grab, with_kwargs=True)
            model(input_ids=ids, use_cache=False)
            hook.remove()
            h, kw = captured["h"], captured["kw"]
            kw["use_cache"] = False

            def qkv(x):
                n_ = norm(x)
                return [p(n_) for p in projs]

            def sequential():
                return qkv(block(h, **kw))

            def overlap(predict):
                def run():
                    main = torch.cuda.current_stream()
                    side.wait_stream(main)
                    with torch.cuda.stream(side):
                        spec = qkv(predict(h))
                    out = block(h, **kw)
                    main.wait_stream(side)
                    return out, spec
                return run

            paths = {
                "block": lambda: block(h, **kw),
                "qkv": lambda: qkv(h),
                "sequential": sequential,
                "overlap_identity": overlap(lambda x: x),
                "overlap_linear": overlap(lambda x: x + x @ a + b),
            }
            reps = args.repeats if n_tok <= 2048 else max(20, args.repeats // 4)
            row = {"tokens": n_tok}
            for name, fn in paths.items():
                row[name] = summary(timed(as_graph(fn) if args.graphs else fn, args.warmup, reps))
            seq = row["sequential"]["median_ms"]
            row["bound_saving_pct"] = 100 * (seq - max(row["block"]["median_ms"], row["qkv"]["median_ms"])) / seq
            for name in ("overlap_identity", "overlap_linear"):
                row[f"{name}_saving_pct"] = 100 * (seq - row[name]["median_ms"]) / seq
            results.append(row)
            print(f"T={n_tok:>5}  block {row['block']['median_ms']:.3f}  qkv {row['qkv']['median_ms']:.3f}  "
                  f"seq {seq:.3f}  ovl_id {row['overlap_identity']['median_ms']:.3f} "
                  f"({row['overlap_identity_saving_pct']:+.1f}%)  ovl_lin {row['overlap_linear']['median_ms']:.3f} "
                  f"({row['overlap_linear_saving_pct']:+.1f}%)  bound {row['bound_saving_pct']:.1f}%  "
                  f"cv_seq {row['sequential']['cv']:.3f}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_name(cfg)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
            "boundary": l, "projections": [n for n, m in nxt.named_modules() if m in projs], "cuda_graphs": args.graphs,
            "warmup": args.warmup, "repeats": args.repeats, "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "gpu_state_before": gpu_state, "attn_implementation": cfg["model"].get("attn_implementation"),
            "results": results}
    mode = "graphs" if args.graphs else "eager"
    (out_dir / f"{stem}_bench_overlap_l{l}_{mode}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"gpu before: {gpu_state}")


if __name__ == "__main__":
    main()
