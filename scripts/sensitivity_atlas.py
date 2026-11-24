"""Phase II, D1 — Layer Sensitivity Atlas v1.

For every decoder block l and each operation in it (the whole block, its attention or linear
attention, its MLP), replace the operation's output and measure the damage to the model's output:

  skip     the operation contributes nothing (block: output = input; attn / MLP: output = 0)
  linear   a ridge map fitted on train text stands in for it (block: h → Δh; attn / MLP: input → output)

Position 0 (BOS, an attention sink) is always kept exact and excluded from metrics. Metrics against
the exact model on calibration text (WikiText-2 train chunks disjoint from the fitting chunks):
  kl, top1, ppl        mean KL(p_exact ‖ p), top-1 agreement, perplexity on the real next tokens
  logit_rel            mean per-position ‖z − ẑ‖ / ‖z‖ of the raw logits
  kl_pos_*             mean KL in position bands 1–63, 64–255, 256 to the end
  state_rel, state_cos relative L2 error and cosine of block l's output against the exact one
  final_rel            relative L2 error of the last block's output (how far the damage propagates)
Cost per token: MACs of the operation's linear layers plus attention score/state work, and of the
replacement (0 for skip, d² + d for linear); saved_frac is the share of all decoder MACs removed.
--time-layer times one block, its attention, its MLP and the linear replacement with CUDA graphs.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_overlap import as_graph, summary, timed  # noqa: E402

from ple.data import WIKITEXT, WIKITEXT_REV, wikitext_chunks
from ple.model_loader import decoder_layers, load, load_config, run_name
from ple.predictors.linear import RidgeAccumulator

OPS, MODES = ("block", "attn", "mlp"), ("skip", "linear")
BANDS = ((1, 64), (64, 256), (256, None))


def first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def with_first(out, new):
    return (new, *out[1:]) if isinstance(out, tuple) else new


def module_input(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def attn_module(layer):
    return layer.self_attn if getattr(layer, "self_attn", None) is not None else layer.linear_attn


def block_kind(layer) -> str:
    return getattr(layer, "block_type", None) or getattr(layer, "attention_type", None) or "full_attention"


def op_module(layer, op):
    return layer if op == "block" else attn_module(layer) if op == "attn" else layer.mlp


def affine(p, x):
    return (x.to(p.weight.dtype) @ p.weight + p.bias.to(p.weight.dtype)).to(x.dtype)


def linear_macs(mod) -> int:
    return sum(m.weight.numel() for m in mod.modules() if isinstance(m, torch.nn.Linear))


def mixing_macs(layer, cfg, seq_len: int) -> float:
    """Per-token attention-score / recurrent-state MACs (approximate) on top of the linear layers."""
    if block_kind(layer) == "linear_attention":
        h = getattr(cfg, "linear_num_value_heads", 0)
        return 3 * h * getattr(cfg, "linear_key_head_dim", 0) * getattr(cfg, "linear_value_head_dim", 0)
    heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    return 2 * heads * head_dim * (seq_len + 1) / 2  # q·k and p·v over the mean causal context


def fit_group(model, layers, group, chunks, dev, d):
    """Ridge fits for every op in `group`, positions 1.. only. Block: h → Δh; attn / MLP: input → output."""
    acc = {(l, op): RidgeAccumulator(d, dev) for l in group for op in OPS}
    handles = []

    def hook(l, op):
        def f(_m, args, kwargs, out):
            x = module_input(args, kwargs)[0, 1:].float()
            y = first(out)[0, 1:].float()
            acc[(l, op)].add(x, y - x if op == "block" else y)
        return f
    for l in group:
        for op in OPS:
            handles.append(op_module(layers[l], op).register_forward_hook(hook(l, op), with_kwargs=True))
    try:
        with torch.inference_mode():
            for ids in chunks:
                model(input_ids=ids[None].to(dev), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {k: a.solve() for k, a in acc.items()}


def install(layer, op, mode, pred):
    """Replace op's output (all positions but 0). Returns the hook handle."""
    def f(_m, args, kwargs, out):
        o = first(out)
        x = module_input(args, kwargs)
        if op == "block":
            new = x.clone() if mode == "skip" else pred(x)
        else:
            new = torch.zeros_like(o) if mode == "skip" else affine(pred, x)
        new[:, 0] = o[:, 0]
        return with_first(out, new)
    return op_module(layer, op).register_forward_hook(f, with_kwargs=True)


def time_ops(model, layers, l, tok, dev, d, preds, n_tok, inner):
    text = torch.cat(wikitext_chunks(tok, "test", 512, 64))[:n_tok][None].to(dev)
    cap = {}

    def grab(name):
        def f(_m, args, kwargs):
            cap[name] = (module_input(args, kwargs), {k: v for k, v in kwargs.items() if k != "hidden_states"})
        return f
    layer = layers[l]
    hs = [op_module(layer, op).register_forward_pre_hook(grab(op), with_kwargs=True) for op in OPS]
    with torch.inference_mode():
        model(input_ids=text, use_cache=False)
    for h in hs:
        h.remove()
    out = {}
    with torch.inference_mode():
        for op in OPS:
            x, kw = cap[op]
            kw = {**kw, "use_cache": False} if op == "block" else kw
            mod = op_module(layer, op)
            out[op] = summary(timed(as_graph(lambda: mod(x, **kw)), 10, 50, inner))
            out[f"{op}_linear"] = summary(timed(as_graph(lambda: affine(preds[(l, op)], x)), 10, 50, inner))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fit-seqs", type=int, default=256)
    ap.add_argument("--calib-skip", type=int, default=400, help="first train chunk used for calibration")
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--fit-group", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--time-layer", type=int, default=None)
    ap.add_argument("--time-tokens", type=int, nargs="*", default=[1, 512])
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()
    assert args.calib_skip >= args.fit_seqs, "calibration chunks must not overlap the fitting chunks"

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    n_layers, dev = len(layers), torch.device(cfg["device"])
    mcfg = model.config.get_text_config()
    d = mcfg.hidden_size
    targets_l = args.layers if args.layers is not None else list(range(n_layers))

    t0 = time.time()
    fit_chunks = wikitext_chunks(tok, "train", args.seq_len, args.fit_seqs)
    preds = {}
    for i in range(0, len(targets_l), args.fit_group):
        preds |= fit_group(model, layers, targets_l[i:i + args.fit_group], fit_chunks, dev, d)
        torch.cuda.empty_cache()
    t_fit = time.time() - t0

    cost = {}
    for l, layer in enumerate(layers):
        a = linear_macs(attn_module(layer)) + mixing_macs(layer, mcfg, args.seq_len)
        m = linear_macs(layer.mlp)
        cost[(l, "attn")], cost[(l, "mlp")], cost[(l, "block")] = a, m, a + m
    decoder_macs = sum(cost[(l, "block")] for l in range(n_layers))
    repl = {"skip": 0, "linear": d * d + d}

    configs = [(l, op, mode) for l in targets_l for op in OPS for mode in MODES]
    keys = ["kl", "top1", "nll", "logit_rel", "state_rel", "state_cos", "final_rel"] + [f"kl_b{i}" for i in range(len(BANDS))]
    stats = {c: dict.fromkeys(keys, 0.0) for c in configs}
    band_n = [0] * len(BANDS)
    exact_nll, count = 0.0, 0
    calib = wikitext_chunks(tok, "train", args.seq_len, args.calib_seqs, skip=args.calib_skip)
    with torch.inference_mode():
        for ids in calib:
            ids = ids[None].to(dev)
            targets = ids[0, 2:]
            n = targets.numel()
            seen = {}
            hooks = [layers[l].register_forward_hook(lambda _m, _a, out, l=l: seen.__setitem__(l, first(out)[0, 1:-1].float()))
                     for l in sorted({*targets_l, n_layers - 1})]
            z_exact = model(input_ids=ids, use_cache=False).logits[0, 1:-1].float()
            for h in hooks:
                h.remove()
            exact_states = dict(seen)
            lp_exact = torch.log_softmax(z_exact, -1)
            p_exact, top_exact = lp_exact.exp(), lp_exact.argmax(-1)
            exact_nll += -lp_exact.gather(-1, targets[:, None]).sum().item()
            count += n
            pos = torch.arange(1, n + 1, device=dev)
            masks = [(pos >= a) & (pos < (b if b is not None else n + 1)) for a, b in BANDS]
            for i, mk in enumerate(masks):
                band_n[i] += int(mk.sum())
            for (l, op, mode) in configs:
                seen.clear()
                hk = install(layers[l], op, mode, preds.get((l, op)))
                caps = [layers[j].register_forward_hook(lambda _m, _a, out, j=j: seen.__setitem__(j, first(out)[0, 1:-1].float()))
                        for j in sorted({l, n_layers - 1})]
                try:
                    z = model(input_ids=ids, use_cache=False).logits[0, 1:-1].float()
                finally:
                    hk.remove()
                    for c in caps:
                        c.remove()
                lp = torch.log_softmax(z, -1)
                kl_tok = (p_exact * (lp_exact - lp)).sum(-1)
                s = stats[(l, op, mode)]
                s["kl"] += kl_tok.sum().item()
                s["top1"] += (lp.argmax(-1) == top_exact).sum().item()
                s["nll"] += -lp.gather(-1, targets[:, None]).sum().item()
                s["logit_rel"] += ((z - z_exact).norm(dim=-1) / z_exact.norm(dim=-1)).sum().item()
                hx, hy = seen[l], exact_states[l]
                s["state_rel"] += ((hx - hy).norm(dim=-1) / hy.norm(dim=-1)).sum().item()
                s["state_cos"] += torch.nn.functional.cosine_similarity(hx, hy, dim=-1).sum().item()
                fx, fy = seen[n_layers - 1], exact_states[n_layers - 1]
                s["final_rel"] += ((fx - fy).norm(dim=-1) / fy.norm(dim=-1)).sum().item()
                for i, mk in enumerate(masks):
                    s[f"kl_b{i}"] += kl_tok[mk].sum().item()

    exact_ppl = math.exp(exact_nll / count)
    rows = []
    for (l, op, mode), s in stats.items():
        ppl = math.exp(s["nll"] / count)
        saved = cost[(l, op)] - repl[mode]
        rows.append({"layer": l, "kind": block_kind(layers[l]), "op": op, "mode": mode,
                     "kl": s["kl"] / count, "top1_agreement": s["top1"] / count, "ppl": ppl,
                     "ppl_change_pct": 100 * (ppl / exact_ppl - 1), "logit_rel": s["logit_rel"] / count,
                     **{f"kl_pos_{a}_{b or 'end'}": s[f"kl_b{i}"] / max(1, band_n[i]) for i, (a, b) in enumerate(BANDS)},
                     "state_rel": s["state_rel"] / count, "state_cos": s["state_cos"] / count,
                     "final_rel": s["final_rel"] / count, "op_macs": cost[(l, op)], "replacement_macs": repl[mode],
                     "saved_frac_decoder": saved / decoder_macs})

    timing = {}
    if args.time_layer is not None:
        for n_tok in args.time_tokens:
            timing[str(n_tok)] = time_ops(model, layers, args.time_layer, tok, dev, d, preds, n_tok, 50 if n_tok <= 2048 else 5)
            t = timing[str(n_tok)]
            print(f"T={n_tok}: " + "  ".join(f"{k} {v['median_ms']:.4f} ms" for k, v in t.items()), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_name(cfg)
    with open(out_dir / f"{stem}_sensitivity_atlas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
            "dataset": WIKITEXT, "dataset_revision": WIKITEXT_REV, "seq_len": args.seq_len,
            "fit_split": f"train chunks 0-{args.fit_seqs - 1}",
            "calib_split": f"train chunks {args.calib_skip}-{args.calib_skip + args.calib_seqs - 1}",
            "eval_tokens": count, "exact_ppl": exact_ppl, "decoder_macs_per_token": decoder_macs,
            "layer_kinds": [block_kind(x) for x in layers], "bands": BANDS, "timing_layer": args.time_layer,
            "timing": timing, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "fit_seconds": round(t_fit, 1), "total_seconds": round(time.time() - t0, 1)}
    (out_dir / f"{stem}_sensitivity_atlas.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"exact ppl {exact_ppl:.3f}, {len(rows)} rows, {meta['total_seconds']} s")


if __name__ == "__main__":
    main()
