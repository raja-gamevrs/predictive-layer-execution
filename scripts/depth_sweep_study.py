"""Where in the stack, and how far ahead, can attention inputs be speculated?

Part A — start depth. Keep blocks before `s` exact and speculate every boundary from s onward
(optionally keeping the last `tail` blocks exact too), uncorrected, with the identity or the
one-block linear predictor. Cumulative: this is the regime that would matter for a real speedup.

Part B — lookahead. Predict block l+k's input from h_l, for k in --lookaheads, with identity
(ĥ = h_l) or a ridge map fit for that k. Reports (i) relative error at block l+k's first input
projection after its norm, for every l, on test traces, and (ii) output quality when one source
layer at a time speculates block l+k, uncorrected.

Data and metrics as in approx_quality_study.py: fit on WikiText-2 train, score on test, position 0
kept exact and excluded; KL(p_exact ‖ p_approx) per token, top-1 agreement, perplexity.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from ple.activation_capture import capture_residual_stream
from ple.data import WIKITEXT, WIKITEXT_REV, wikitext_chunks
from ple.model_loader import attention_input_projection, decoder_layers, load, load_config, run_name
from ple.predictors.fitting import fit_linear
from ple.speculation_policy import speculate_attention_inputs


def uncorrected(_l):
    return lambda mod, y_hat, e: y_hat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fit-seqs", type=int, default=256)
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--proj-seqs", type=int, default=32)
    ap.add_argument("--lookaheads", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--starts", type=int, nargs="*", default=None)
    ap.add_argument("--sources", type=int, nargs="*", default=None, help="single source layers for Part B quality")
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    n_layers, dev = len(layers), torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    norms = [layer.input_layernorm for layer in layers]
    projs = [attention_input_projection(layer, d)[1] for layer in layers]

    def traces(split, n):
        for ids in wikitext_chunks(tok, split, args.seq_len, n):
            yield [s[0, 1:] for s in capture_residual_stream(model, layers, input_ids=ids[None].to(dev))]

    eval_ids = [ids[None].to(dev) for ids in wikitext_chunks(tok, "test", args.seq_len, args.eval_seqs)]
    with torch.inference_mode():
        exact = [torch.log_softmax(model(input_ids=ids).logits[0, 1:-1].float(), dim=-1) for ids in eval_ids]
    count = sum(ids.shape[1] - 2 for ids in eval_ids)
    exact_nll = sum(-lp.gather(-1, ids[0, 2:, None]).sum().item() for lp, ids in zip(exact, eval_ids))

    def quality(boundaries, predictor, lookahead):
        kl = top1 = nll = 0.0
        with torch.inference_mode(), speculate_attention_inputs(layers, boundaries, predictor, uncorrected, d, lookahead):
            for lp_exact, ids in zip(exact, eval_ids):
                lp = torch.log_softmax(model(input_ids=ids).logits[0, 1:-1].float(), dim=-1)
                kl += (lp_exact.exp() * (lp_exact - lp)).sum().item()
                top1 += (lp.argmax(-1) == lp_exact.argmax(-1)).sum().item()
                nll += -lp.gather(-1, ids[0, 2:, None]).sum().item()
        return {"kl": kl / count, "top1_agreement": top1 / count, "ppl": math.exp(nll / count)}

    t0 = time.time()
    identity = {l: (lambda h: h) for l in range(n_layers)}
    stem = run_name(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part A: start depth, cumulative, k = 1.
    linear1 = fit_linear(traces("train", args.fit_seqs), range(n_layers - 1), d, dev)
    starts = args.starts if args.starts is not None else list(range(0, n_layers - 2, 4)) + [n_layers - 6, n_layers - 4]
    rows_a = []
    for s in sorted(set(starts)):
        for tail in (0, 2):
            bounds = list(range(s, n_layers - 1 - tail))
            if not bounds:
                continue
            for pname, pred in (("identity", identity), ("linear", linear1)):
                r = quality(bounds, pred, 1) | {"start": s, "tail_exact": tail, "speculated_blocks": len(bounds), "predictor": pname}
                rows_a.append(r)
                print(f"A start={s:>2} tail={tail} n={len(bounds):>2} {pname:<8} KL {r['kl']:.4f} top1 {r['top1_agreement']:.3f} "
                      f"ppl {r['ppl']:.3f}", flush=True)
    del linear1
    torch.cuda.empty_cache()

    # Part B: lookahead k.
    rows_proj, rows_b = [], []
    sources = args.sources if args.sources is not None else list(range(2, n_layers - 1, 4))
    for k in args.lookaheads:
        src = list(range(n_layers - k))
        lin = fit_linear(traces("train", args.fit_seqs), src, d, dev, lookahead=k)
        se = {(l, p): 0.0 for l in src for p in ("identity", "linear")}
        ref = {l: 0.0 for l in src}
        with torch.inference_mode():
            for hs in traces("test", args.proj_seqs):
                for l in src:
                    t = l + k
                    w = projs[t]
                    y_true = w.forward(norms[t](hs[t])).float()
                    ref[l] += y_true.pow(2).sum().item()
                    for pname, h_hat in (("identity", hs[l]), ("linear", lin[l](hs[l]))):
                        se[(l, pname)] += (w.forward(norms[t](h_hat)).float() - y_true).pow(2).sum().item()
        for (l, pname), v in se.items():
            rows_proj.append({"lookahead": k, "source": l, "target_block": l + k, "predictor": pname,
                              "proj_rel_err": math.sqrt(v / ref[l])})
        for l in sources:
            if l + k >= n_layers:
                continue
            for pname, pred in (("identity", identity), ("linear", lin)):
                r = quality([l], pred, k) | {"lookahead": k, "source": l, "target_block": l + k, "predictor": pname}
                rows_b.append(r)
                print(f"B k={k} src={l:>2}→{l + k:>2} {pname:<8} KL {r['kl']:.4f} top1 {r['top1_agreement']:.3f} ppl {r['ppl']:.3f}",
                      flush=True)
        meds = {p: sorted(r["proj_rel_err"] for r in rows_proj if r["lookahead"] == k and r["predictor"] == p) for p in ("identity", "linear")}
        print(f"B k={k} median proj err: identity {meds['identity'][len(meds['identity']) // 2]:.3f}, "
              f"linear {meds['linear'][len(meds['linear']) // 2]:.3f}", flush=True)
        del lin
        torch.cuda.empty_cache()

    for name, rows in (("depth_start", rows_a), ("lookahead_quality", rows_b), ("lookahead_proj", rows_proj)):
        with open(out_dir / f"{stem}_{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
            "quantization": cfg["model"].get("quantization"), "dataset": WIKITEXT, "dataset_revision": WIKITEXT_REV,
            "seq_len": args.seq_len, "fit_seqs": args.fit_seqs, "eval_seqs": args.eval_seqs, "proj_seqs": args.proj_seqs,
            "eval_tokens": count, "exact_ppl": math.exp(exact_nll / count), "lookaheads": args.lookaheads,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{stem}_depth_sweep.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
