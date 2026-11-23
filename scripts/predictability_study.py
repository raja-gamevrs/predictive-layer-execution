"""Stage B/C, hypothesis H1: how predictable is h_(l+1) from earlier residual states?

Fits per-layer linear predictors on WikiText-2 train, then scores every predictor on WikiText-2 test:
  rel_err         ‖ĥ − h‖ / ‖h‖                    on the next residual state
  delta_rel_err   ‖ĥ − h‖ / ‖Δ‖                    1.0 = identity; < 1 means the predictor explains part of the update
  delta_cos       cos(Δ̂, Δ)                        direction of the predicted update
  norm_in_rel_err ‖N(ĥ) − N(h)‖ / ‖N(h)‖           at the next block's input norm (what QKV would actually see)
  proj_rel_err    ‖W N(ĥ) − W N(h)‖ / ‖W N(h)‖     at the next block's first input projection (C2 speculation target)
Position 0 is excluded everywhere (attention-sink token with outlier norms).
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from ple.activation_capture import capture_residual_stream
from ple.model_loader import attention_input_projection, decoder_layers, load, load_config
from ple.predictors import identity, residual_extrapolation, velocity
from ple.predictors.linear import RidgeAccumulator

DATASET, DATASET_REV = "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"
RANKS = (64, 256)


def chunks(tokenizer, split: str, seq_len: int, n: int) -> list[torch.Tensor]:
    path = hf_hub_download(DATASET, f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=DATASET_REV)
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    body = seq_len - len(bos)
    out = [torch.tensor(bos + ids[i:i + body]) for i in range(0, len(ids) - body, body)][:n]
    if len(out) < n:
        raise ValueError(f"{split}: only {len(out)} chunks of {seq_len} tokens")
    return out


def low_rank(pred, rank: int):
    u, s, vh = torch.linalg.svd(pred.weight)
    w = (u[:, :rank] * s[:rank]) @ vh[:rank]
    return lambda h: h + (h.float() @ w + pred.bias).to(h.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--train-seqs", type=int, default=64)
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    n_layers, dev = len(layers), torch.device(cfg["device"])
    hidden = model.config.get_text_config().hidden_size
    norms = [getattr(layer, "input_layernorm", None) for layer in layers]
    projs = [attention_input_projection(layer, hidden) for layer in layers]

    def run(ids: torch.Tensor) -> list[torch.Tensor]:
        states = capture_residual_stream(model, layers, input_ids=ids[None].to(dev))
        return [s[0, 1:] for s in states]  # drop batch dim and position 0

    t0 = time.time()
    acc = [RidgeAccumulator(hidden, dev) for _ in range(n_layers)]
    for ids in chunks(tok, "train", args.seq_len, args.train_seqs):
        hs = run(ids)
        for l in range(n_layers):
            acc[l].add(hs[l].float(), (hs[l + 1] - hs[l]).float())
    linear = [a.solve() for a in acc]
    train_tokens = acc[0].count
    del acc
    torch.cuda.empty_cache()
    t_fit = time.time() - t0

    predictors = {
        "identity": lambda l, hs: identity(hs[: l + 1]),
        "extrapolation": lambda l, hs: residual_extrapolation(hs[: l + 1]) if l >= 1 else None,
        "velocity_a0.5": lambda l, hs: velocity(hs[: l + 1], alpha=0.5) if l >= 1 else None,
        "linear": lambda l, hs: linear[l](hs[l]),
    }
    for r in RANKS:
        lr = [low_rank(p, r) for p in linear]
        predictors[f"linear_r{r}"] = lambda l, hs, lr=lr: lr[l](hs[l])

    keys = ("se", "hn", "dn", "cos", "nse", "nn", "pse", "pn")
    sums = {(l, p): dict.fromkeys(keys, 0.0) for l in range(n_layers) for p in predictors}
    eval_tokens = 0
    with torch.inference_mode():
        for ids in chunks(tok, "test", args.seq_len, args.eval_seqs):
            hs = run(ids)
            eval_tokens += hs[0].shape[0]
            for l in range(n_layers):
                h, h_next = hs[l], hs[l + 1]
                delta = (h_next - h).float()
                nxt = l + 1 < n_layers and norms[l + 1] is not None
                if nxt:
                    n_true = norms[l + 1](h_next).float()
                    w = projs[l + 1][1]
                    p_true = torch.nn.functional.linear(n_true.to(w.weight.dtype), w.weight).float()
                for name, fn in predictors.items():
                    h_hat = fn(l, hs)
                    if h_hat is None:
                        continue
                    s = sums[(l, name)]
                    err = (h_hat.float() - h_next.float())
                    s["se"] += err.pow(2).sum().item()
                    s["hn"] += h_next.float().pow(2).sum().item()
                    s["dn"] += delta.pow(2).sum().item()
                    s["cos"] += torch.nn.functional.cosine_similarity(h_hat.float() - h.float(), delta, dim=-1).nan_to_num().sum().item()
                    if nxt:
                        n_hat = norms[l + 1](h_hat).float()
                        p_hat = torch.nn.functional.linear(n_hat.to(w.weight.dtype), w.weight).float()
                        s["nse"] += (n_hat - n_true).pow(2).sum().item()
                        s["nn"] += n_true.pow(2).sum().item()
                        s["pse"] += (p_hat - p_true).pow(2).sum().item()
                        s["pn"] += p_true.pow(2).sum().item()

    rows = []
    for (l, name), s in sums.items():
        if s["hn"] == 0:
            continue
        ratio = lambda a, b: math.sqrt(s[a] / s[b]) if s[b] else ""
        rows.append({
            "layer": l, "predictor": name,
            "rel_err": ratio("se", "hn"), "delta_rel_err": ratio("se", "dn"),
            "delta_cos": s["cos"] / eval_tokens,
            "norm_in_rel_err": ratio("nse", "nn"), "proj_rel_err": ratio("pse", "pn"),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = cfg["model"]["id"].split("/")[-1]
    with open(out / f"{stem}_predictability.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
        "dataset": DATASET, "dataset_revision": DATASET_REV, "train_split": "train", "eval_split": "test",
        "seq_len": args.seq_len, "train_tokens": train_tokens, "eval_tokens": eval_tokens,
        "layers": n_layers, "hidden": hidden, "projection": projs[0][0],
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "fit_seconds": round(t_fit, 1),
    }
    (out / f"{stem}_predictability.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"\n{'predictor':<15} {'median Δrel':>11} {'best Δrel (layer)':>18} {'median proj':>11} {'median cos':>10}")
    for name in predictors:
        r = [x for x in rows if x["predictor"] == name]
        med = lambda k: sorted(x[k] for x in r if x[k] != "")[len([x for x in r if x[k] != ""]) // 2]
        best = min(r, key=lambda x: x["delta_rel_err"])
        print(f"{name:<15} {med('delta_rel_err'):>11.3f} {best['delta_rel_err']:>12.3f} ({best['layer']:>2}) "
              f"{med('proj_rel_err'):>11.3f} {med('delta_cos'):>10.3f}")


if __name__ == "__main__":
    main()
