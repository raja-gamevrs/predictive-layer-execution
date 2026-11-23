"""Stage C/D, hypothesis H2/H3: is the speculation error structured enough to correct cheaply?

Setting (C2 in docs/techniques.md): speculate the next block's input projection from ĥ, i.e.
ŷ = W N(ĥ); when h arrives, e = N(h) − N(ĥ) is cheap to form, and the exact fix is W e. This
measures how much of W e each cheaper correction recovers, against its cost as a fraction of
one exact projection (d·d_out MACs per token):

  basis_r{r}     fixed basis U_r (top-r principal directions of e, fitted on held-in train
                 errors), W U_r precomputed                     cost r·(d + d_out) / (d·d_out)
  topk_{p}pct    per-token top-k channels by |e_i|·‖W[:, i]‖    cost k / d  (plus gather)
  tokens_{p}pct  exact fix for the p% of tokens with largest ‖e‖  cost p / 100

Data: ridge predictors fit on train chunks [0, fit), error bases on train chunks
[fit, fit + basis), scored on test. Reported value: proj_rel_err = ‖ỹ − W N(h)‖ / ‖W N(h)‖.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from ple.activation_capture import capture_residual_stream
from ple.correction import basis_correction, channel_topk_correction, token_selective_correction
from ple.data import WIKITEXT, WIKITEXT_REV, wikitext_chunks
from ple.model_loader import attention_input_projection, decoder_layers, load, load_config
from ple.predictors.linear import RidgeAccumulator

RANKS = (16, 64, 256, 512)
TOPK_PCT = (1, 5, 10, 25)
TOKEN_PCT = (5, 10, 25, 50)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fit-seqs", type=int, default=256)
    ap.add_argument("--basis-seqs", type=int, default=128)
    ap.add_argument("--eval-seqs", type=int, default=64)
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    n_layers, dev = len(layers), torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    norms = [getattr(layer, "input_layernorm") for layer in layers]
    proj_name, _ = attention_input_projection(layers[0], d)
    projs = [attention_input_projection(layer, d)[1].weight for layer in layers]
    d_out = projs[0].shape[0]
    targets = range(n_layers - 1)  # block l predicts the input of block l+1

    def run(ids):
        states = capture_residual_stream(model, layers, input_ids=ids[None].to(dev))
        return [s[0, 1:] for s in states]

    t0 = time.time()
    acc = [RidgeAccumulator(d, dev) for _ in range(n_layers)]
    for ids in wikitext_chunks(tok, "train", args.seq_len, args.fit_seqs):
        hs = run(ids)
        for l in targets:
            acc[l].add(hs[l].float(), (hs[l + 1] - hs[l]).float())
    full = {l: acc[l].solve() for l in targets}
    del acc
    torch.cuda.empty_cache()
    predictors = {
        "identity": {l: (lambda h: h) for l in targets},
        "linear_r256": {l: full[l].truncated(256) for l in targets},
        "linear": full,
    }

    def normed_error(l, hs, pred):
        n_true = norms[l + 1](hs[l + 1]).float()
        n_hat = norms[l + 1](pred[l](hs[l])).float()
        return n_true, n_hat, n_true - n_hat

    # Error covariance on train chunks the predictors were not fit on.
    cov = {(l, p): torch.zeros(d, d, dtype=torch.float64, device=dev) for l in targets for p in predictors}
    with torch.inference_mode():
        for ids in wikitext_chunks(tok, "train", args.seq_len, args.basis_seqs, skip=args.fit_seqs):
            hs = run(ids)
            for l in targets:
                for p, pred in predictors.items():
                    e = normed_error(l, hs, pred)[2]
                    cov[(l, p)] += (e.T @ e).double()
    bases, energy = {}, {}
    for key, c in cov.items():
        evals, evecs = torch.linalg.eigh(c)
        evals, evecs = evals.flip(0).clamp_min(0), evecs.flip(1)
        energy[key] = {r: (evals[:r].sum() / evals.sum()).item() for r in RANKS}
        w = projs[key[0] + 1].float()
        bases[key] = {r: (evecs[:, :r].float(), w @ evecs[:, :r].float()) for r in RANKS}
    del cov
    t_fit = time.time() - t0

    methods = (["none"] + [f"basis_r{r}" for r in RANKS] + [f"topk_{p}pct" for p in TOPK_PCT]
               + [f"tokens_{p}pct" for p in TOKEN_PCT])
    cost = {"none": 0.0}
    cost.update({f"basis_r{r}": r * (d + d_out) / (d * d_out) for r in RANKS})
    cost.update({f"topk_{p}pct": round(p / 100 * d) / d for p in TOPK_PCT})
    cost.update({f"tokens_{p}pct": p / 100 for p in TOKEN_PCT})
    col_norms = [w.float().norm(dim=0) for w in projs]

    se = {(l, p, m): 0.0 for l in targets for p in predictors for m in methods}
    ref = {l: 0.0 for l in targets}
    with torch.inference_mode():
        for ids in wikitext_chunks(tok, "test", args.seq_len, args.eval_seqs):
            hs = run(ids)
            for l in targets:
                w = projs[l + 1].float()
                for p, pred in predictors.items():
                    n_true, n_hat, e = normed_error(l, hs, pred)
                    y_true = torch.nn.functional.linear(n_true, w)
                    y_hat = torch.nn.functional.linear(n_hat, w)
                    if p == "identity":
                        ref[l] += y_true.pow(2).sum().item()
                    out = {"none": y_hat}
                    for r in RANKS:
                        u, wu = bases[(l, p)][r]
                        out[f"basis_r{r}"] = basis_correction(wu, u, y_hat, e)
                    for pct in TOPK_PCT:
                        out[f"topk_{pct}pct"] = channel_topk_correction(w, y_hat, e, round(pct / 100 * d), col_norms[l + 1])
                    for pct in TOKEN_PCT:
                        out[f"tokens_{pct}pct"] = token_selective_correction(w, y_hat, e, pct / 100)
                    for m, y in out.items():
                        se[(l, p, m)] += (y - y_true).pow(2).sum().item()

    rows = [{"layer": l, "predictor": p, "method": m, "cost_vs_projection": round(cost[m], 4),
             "proj_rel_err": (se[(l, p, m)] / ref[l]) ** 0.5,
             "train_energy_captured": energy[(l, p)][int(m[7:])] if m.startswith("basis_r") else ""}
            for (l, p, m) in se]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg["model"]["id"].split("/")[-1]
    with open(out_dir / f"{stem}_error_structure.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
            "dataset": WIKITEXT, "dataset_revision": WIKITEXT_REV, "seq_len": args.seq_len,
            "fit_seqs": args.fit_seqs, "basis_seqs": args.basis_seqs, "eval_seqs": args.eval_seqs,
            "projection": proj_name, "d": d, "d_out": d_out, "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__, "fit_seconds": round(t_fit, 1)}
    (out_dir / f"{stem}_error_structure.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"\nmedian proj_rel_err across layers {min(targets)}..{max(targets)}")
    print(f"{'method':<15} {'cost':>6} " + " ".join(f"{p:>12}" for p in predictors))
    for m in methods:
        meds = []
        for p in predictors:
            v = sorted(r["proj_rel_err"] for r in rows if r["predictor"] == p and r["method"] == m)
            meds.append(v[len(v) // 2])
        print(f"{m:<15} {cost[m]:>6.3f} " + " ".join(f"{x:>12.3f}" for x in meds))


if __name__ == "__main__":
    main()
