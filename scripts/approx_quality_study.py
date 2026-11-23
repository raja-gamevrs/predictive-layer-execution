"""Output-quality cost of approximate PLE: speculate attention inputs, correct cheaply, compare logits.

For each boundary set (one layer at a time, or every layer at once), predictor and correction,
runs held-out WikiText-2 test text through the model with block l+1's attention input projections
replaced by W·N(ĥ) + correction, and compares the next-token distribution with the exact model:
  kl        mean KL(p_exact ‖ p_approx) per token, in nats
  top1      fraction of positions where the argmax token is unchanged
  ppl       perplexity on the actual next tokens (exact model's ppl reported alongside)
Position 0 is kept exact and excluded from metrics. `full` correction is the exactness check.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from ple.activation_capture import capture_residual_stream
from ple.correction import basis_correction, channel_topk_correction
from ple.data import WIKITEXT, WIKITEXT_REV, wikitext_chunks
from ple.model_loader import decoder_layers, load, load_config, run_name
from ple.predictors.fitting import fit_error_bases, fit_linear
from ple.speculation_policy import apply_without_bias, speculate_attention_inputs

RANK, TOPK_FRAC = 256, 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fit-seqs", type=int, default=256)
    ap.add_argument("--basis-seqs", type=int, default=128)
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--single-layers", type=int, nargs="*", default=None)
    ap.add_argument("--corrections", nargs="+",
                    default=["none", f"basis_r{RANK}", "topk_25pct", "full", "full_fp32", "full_module"],
                    help="quantized models: use only none and full_module (weight-based ones need dense weights)")
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    n_layers, dev = len(layers), torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    norms = [layer.input_layernorm for layer in layers]
    boundaries_all = list(range(n_layers - 1))

    def traces(split, n, skip=0):
        for ids in wikitext_chunks(tok, split, args.seq_len, n, skip=skip):
            yield [s[0, 1:] for s in capture_residual_stream(model, layers, input_ids=ids[None].to(dev))]

    t0 = time.time()
    linear = fit_linear(traces("train", args.fit_seqs), boundaries_all, d, dev)
    predictors = {"identity": {l: (lambda h: h) for l in boundaries_all}, "linear": linear}
    bases = {}
    if f"basis_r{RANK}" in args.corrections:
        bases = fit_error_bases(traces("train", args.basis_seqs, skip=args.fit_seqs), boundaries_all, norms,
                                predictors, d, RANK, dev)
    torch.cuda.empty_cache()
    t_fit = time.time() - t0

    k = round(TOPK_FRAC * d)

    def corrections(pred_name):
        # e arrives in FP32. BF16 corrections round it to the weight dtype first, as a BF16 kernel would.
        # Weight-based corrections need a dense float weight; `full_module` goes through the module's own
        # kernel, so for activation-quantized projections it quantizes e (and is then no longer exact).
        def basis(l):
            u = bases[(l, pred_name)]

            def fix(mod, y_hat, e):
                w = mod.weight
                return basis_correction((w.float() @ u).to(w.dtype), u.to(w.dtype), y_hat, e.to(w.dtype))
            return fix
        table = {
            "none": lambda l: (lambda mod, y_hat, e: y_hat),
            f"basis_r{RANK}": basis,
            "topk_25pct": lambda l: (lambda mod, y_hat, e: channel_topk_correction(mod.weight, y_hat, e.to(mod.weight.dtype), k)),
            "full": lambda l: (lambda mod, y_hat, e: y_hat + torch.nn.functional.linear(e.to(mod.weight.dtype), mod.weight)),
            "full_fp32": lambda l: (lambda mod, y_hat, e: y_hat.float() + torch.nn.functional.linear(e, mod.weight.float())),
            "full_module": lambda l: (lambda mod, y_hat, e: y_hat + apply_without_bias(mod, e.to(y_hat.dtype))),
        }
        return {name: table[name] for name in args.corrections}

    singles = args.single_layers if args.single_layers is not None else sorted(
        {*range(2, n_layers - 1, 4), n_layers - 2})
    boundary_sets = {f"layer{l}": [l] for l in singles} | {"all": boundaries_all}
    configs = [(b, p, c) for b in boundary_sets for p in predictors for c in corrections(p)]

    stats = {cfg_: {"kl": 0.0, "top1": 0, "nll": 0.0} for cfg_ in configs}
    exact_nll, count = 0.0, 0
    rerun = {"kl": 0.0, "top1": 0}
    with torch.inference_mode():
        for ids in wikitext_chunks(tok, "test", args.seq_len, args.eval_seqs):
            ids = ids[None].to(dev)
            targets = ids[0, 2:]

            def score(logits):
                lp = torch.log_softmax(logits[0, 1:-1].float(), dim=-1)  # positions 1..T-2 predict 2..T-1
                return lp, -lp.gather(-1, targets[:, None]).sum().item()

            lp_exact, nll = score(model(input_ids=ids).logits)
            exact_nll += nll
            count += targets.numel()
            top_exact = lp_exact.argmax(-1)
            lp_again, _ = score(model(input_ids=ids).logits)  # run-to-run determinism of the exact path
            rerun["kl"] += (lp_exact.exp() * (lp_exact - lp_again)).sum().item()
            rerun["top1"] += (lp_again.argmax(-1) == top_exact).sum().item()
            for b, p, c in configs:
                with speculate_attention_inputs(layers, boundary_sets[b], predictors[p], corrections(p)[c], d):
                    lp, nll = score(model(input_ids=ids).logits)
                s = stats[(b, p, c)]
                s["kl"] += (lp_exact.exp() * (lp_exact - lp)).sum().item()
                s["top1"] += (lp.argmax(-1) == top_exact).sum().item()
                s["nll"] += nll

    rows = [{"boundaries": b, "predictor": p, "correction": c, "kl": s["kl"] / count,
             "top1_agreement": s["top1"] / count, "ppl": math.exp(s["nll"] / count)}
            for (b, p, c), s in stats.items()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_name(cfg)
    with open(out_dir / f"{stem}_approx_quality.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
            "dataset": WIKITEXT, "dataset_revision": WIKITEXT_REV, "seq_len": args.seq_len,
            "fit_seqs": args.fit_seqs, "basis_seqs": args.basis_seqs, "eval_seqs": args.eval_seqs,
            "eval_tokens": count, "exact_ppl": math.exp(exact_nll / count),
            "exact_rerun_kl": rerun["kl"] / count, "exact_rerun_top1": rerun["top1"] / count, "basis_rank": RANK,
            "topk_fraction": TOPK_FRAC, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "fit_seconds": round(t_fit, 1)}
    (out_dir / f"{stem}_approx_quality.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"\n{'boundaries':<10} {'predictor':<9} {'correction':<11} {'KL':>9} {'top1':>7} {'ppl':>8}")
    for r in rows:
        print(f"{r['boundaries']:<10} {r['predictor']:<9} {r['correction']:<11} {r['kl']:>9.4f} "
              f"{r['top1_agreement']:>7.3f} {r['ppl']:>8.3f}")


if __name__ == "__main__":
    main()
