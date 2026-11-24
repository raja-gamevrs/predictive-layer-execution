"""Fused speculative QKV: compute block l+1..l+k's attention input projections inside block l's own.

With the identity predictor (ĥ_{l+j} = h_l), every block's normed input is RMSNorm(h_l) up to its
per-channel gain g: N_j(h) = (h / rms(h)) ⊙ g_j. Folding each gain into its block's weights,
W'_j = W_j · diag(g_j), makes all of them one matrix multiply on the shared h / rms(h):

    [q_l k_l v_l | q_{l+1} k_{l+1} v_{l+1} | ...] = (h / rms(h)) · [W'_l | W'_{l+1} | ...]ᵀ

so the speculative projections need no second stream and no separate kernel. Paths timed (CUDA
graphs, real activations entering block l), for k = number of speculated blocks:

  hf_k        HF modules: per block, its RMSNorm module and three q/k/v GEMMs   (1 + k) norms, 3(1 + k) GEMMs
  packed_k    production layout: per block, a single-kernel RMSNorm and one packed QKV GEMM
  fused_k     one single-kernel rms (no gain), one GEMM with all 1 + k folded weight blocks side by side
  packed_0    block l's own norm + packed QKV, the work block l does anyway

Speculation saving per block = (packed_k - fused_k) / k, reported in ms and as a share of block l's
full forward time (HF module, same input). Numerics: fused vs packed outputs, relative error per block.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_overlap import as_graph, summary, timed  # noqa: E402

from ple.data import wikitext_chunks
from ple.model_loader import decoder_layers, load, load_config, run_name
from ple.speculation_policy import attention_input_linears


def eps_of(norm: torch.nn.Module) -> float:
    return getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))


def rms(x: torch.Tensor, norm: torch.nn.Module, gain: bool = True) -> torch.Tensor:
    """Single-kernel RMSNorm (torch.nn.functional.rms_norm), with or without the module's gain."""
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), norm.weight if gain else None, eps_of(norm))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--boundary", type=int, default=26)
    ap.add_argument("--lookahead", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 16, 64, 128, 512, 2048, 8192])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--inner", type=int, default=50, help="calls per timed sample; reduced to 5 above 2048 tokens")
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    dev = torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    l, kmax = args.boundary, max(args.lookahead)
    assert l + kmax < len(layers), "lookahead runs past the last block"
    blocks = [layers[l + j] for j in range(kmax + 1)]
    norms = [b.input_layernorm for b in blocks]
    projs = [attention_input_linears(b, d) for b in blocks]
    assert all(len(p) == 3 and all(m.bias is None for m in p) for p in projs), "expects bias-free q/k/v"

    with torch.no_grad():
        packed = [torch.cat([m.weight for m in p], 0) for p in projs]                        # [qkv, d] per block
        folded = [(w.float() * n.weight.float()[None, :]).to(w.dtype) for w, n in zip(packed, norms)]
        fused = {k: torch.cat(folded[:k + 1], 0).contiguous() for k in [0, *args.lookahead]}
    width = packed[0].shape[0]
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
            hook = blocks[0].register_forward_pre_hook(grab, with_kwargs=True)
            model(input_ids=ids, use_cache=False)
            hook.remove()
            h, kw = captured["h"], captured["kw"]
            kw["use_cache"] = False

            def hf(k):
                return lambda: [[m(norms[j](h)) for m in projs[j]] for j in range(k + 1)]

            def packed_path(k):
                return lambda: [torch.nn.functional.linear(rms(h, norms[j]), packed[j]) for j in range(k + 1)]

            def fused_path(k):
                return lambda: torch.nn.functional.linear(rms(h, norms[0], gain=False), fused[k])

            reps = args.repeats if n_tok <= 2048 else max(50, args.repeats // 4)
            inner = args.inner if n_tok <= 2048 else min(args.inner, 5)

            def t(fn):
                return summary(timed(as_graph(fn), args.warmup, reps, inner))
            row = {"tokens": n_tok, "inner": inner, "block": t(lambda: blocks[0](h, **kw)),
                   "packed_0": t(packed_path(0))}
            for k in args.lookahead:
                for name, path in (("hf", hf), ("packed", packed_path), ("fused", fused_path)):
                    row[f"{name}_{k}"] = t(path(k))
                ref = packed_path(k)()
                got = fused_path(k)().split(width, dim=-1)
                row[f"fused_{k}_rel_err"] = [((g.float() - r.float()).norm() / r.float().norm()).item() for g, r in zip(got, ref)]
                p, f, b = row[f"packed_{k}"]["median_ms"], row[f"fused_{k}"]["median_ms"], row["block"]["median_ms"]
                extra = p - row["packed_0"]["median_ms"]
                row[f"saving_per_block_ms_{k}"] = (p - f) / k
                row[f"saving_pct_of_block_{k}"] = 100 * (p - f) / k / b
                row[f"hidden_share_{k}"] = (p - f) / extra if extra > 0 else None
            results.append(row)
            print(f"T={n_tok:>5}  block {row['block']['median_ms']:.4f}  packed_0 {row['packed_0']['median_ms']:.4f}  " +
                  "  ".join(f"k={k}: packed {row[f'packed_{k}']['median_ms']:.4f} fused {row[f'fused_{k}']['median_ms']:.4f} "
                            f"save/block {row[f'saving_pct_of_block_{k}']:+.1f}% of block "
                            f"(cv {row[f'fused_{k}']['cv']:.3f}, err {max(row[f'fused_{k}_rel_err']):.1e})" for k in args.lookahead),
                  flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
            "boundary": l, "lookahead": args.lookahead, "qkv_width": width, "hidden_size": d, "cuda_graphs": True,
            "warmup": args.warmup, "repeats": args.repeats, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "gpu_state_before": gpu_state, "attn_implementation": cfg["model"].get("attn_implementation"), "results": results}
    (out_dir / f"{run_name(cfg)}_bench_fused_qkv_l{l}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"gpu before: {gpu_state}")


if __name__ == "__main__":
    main()
