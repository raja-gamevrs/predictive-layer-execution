"""Phase II, D3 — additivity of damage and static greedy allocation under a quality budget.

Reads a sensitivity atlas (scripts/sensitivity_atlas.py), ranks single operations (layer, op, mode)
by measured KL per unit of decoder compute removed, and on calibration text measures:

  pairs    every pair among the top --pair-top candidates: measured damage of both together vs the
           sum of their single damages (additivity test)
  greedy   the cumulative set formed by adding candidates in rank order, measured after each addition
           (the static greedy allocation; a candidate is skipped if its layer's block is already in)

The ranking uses the atlas (calibration chunks --calib-skip …); the greedy sets are re-measured on the
same chunks and, separately, on --holdout-skip chunks, so the selection's stability can be checked.
The WikiText-2 test split is not used. Position 0 stays exact, as in the atlas.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import torch

import sensitivity_atlas as atlas
from ple.data import WIKITEXT, WIKITEXT_REV, wikitext_chunks
from ple.model_loader import decoder_layers, load, load_config, run_name


def evaluate(model, layers, chunks, dev, sets, preds):
    """Mean KL, top-1 and perplexity for each set of (layer, op, mode) replacements, plus exact ppl."""
    stats = {i: {"kl": 0.0, "top1": 0, "nll": 0.0} for i in range(len(sets))}
    exact_nll, count = 0.0, 0
    with torch.inference_mode():
        for ids in chunks:
            ids = ids[None].to(dev)
            targets = ids[0, 2:]
            lp_exact = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, 1:-1].float(), -1)
            p_exact, top_exact = lp_exact.exp(), lp_exact.argmax(-1)
            exact_nll += -lp_exact.gather(-1, targets[:, None]).sum().item()
            count += targets.numel()
            for i, members in enumerate(sets):
                handles = [atlas.install(layers[l], op, mode, preds.get((l, op))) for l, op, mode in members]
                try:
                    lp = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, 1:-1].float(), -1)
                finally:
                    for h in handles:
                        h.remove()
                s = stats[i]
                s["kl"] += (p_exact * (lp_exact - lp)).sum().item()
                s["top1"] += (lp.argmax(-1) == top_exact).sum().item()
                s["nll"] += -lp.gather(-1, targets[:, None]).sum().item()
    exact_ppl = math.exp(exact_nll / count)
    out = []
    for i in range(len(sets)):
        s = stats[i]
        ppl = math.exp(s["nll"] / count)
        out.append({"kl": s["kl"] / count, "top1_agreement": s["top1"] / count, "ppl": ppl,
                    "ppl_change_pct": 100 * (ppl / exact_ppl - 1)})
    return out, exact_ppl, count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--atlas", required=True, help="sensitivity_atlas CSV for this model")
    ap.add_argument("--modes", nargs="+", default=["skip", "linear"])
    ap.add_argument("--ops", nargs="+", default=["block", "attn", "mlp"])
    ap.add_argument("--pair-top", type=int, default=8)
    ap.add_argument("--greedy-max", type=int, default=24)
    ap.add_argument("--budget-pct", type=float, default=1.0, help="perplexity budget, for reporting only")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fit-seqs", type=int, default=256)
    ap.add_argument("--calib-skip", type=int, default=400)
    ap.add_argument("--holdout-skip", type=int, default=600)
    ap.add_argument("--calib-seqs", type=int, default=16)
    ap.add_argument("--fit-group", type=int, default=12)
    ap.add_argument("--out", default="results/processed")
    args = ap.parse_args()

    with open(args.atlas, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["mode"] in args.modes and r["op"] in args.ops]
    for r in rows:
        r["layer"], r["kl"], r["saved"] = int(r["layer"]), float(r["kl"]), float(r["saved_frac_decoder"])
        r["ppl_change_pct"] = float(r["ppl_change_pct"])
    # Rank by damage per compute removed; per (layer, op) keep the better mode.
    best = {}
    for r in rows:
        if r["saved"] <= 0:
            continue
        key = (r["layer"], r["op"])
        if key not in best or r["kl"] / r["saved"] < best[key]["kl"] / best[key]["saved"]:
            best[key] = r
    ranked = sorted(best.values(), key=lambda r: r["kl"] / r["saved"])

    greedy, taken_blocks, taken_ops = [], set(), set()
    for r in ranked:
        l, op = r["layer"], r["op"]
        if l in taken_blocks or (op == "block" and any(t[0] == l for t in taken_ops)):
            continue  # a block replacement overlaps its own attention / MLP replacements
        greedy.append(r)
        (taken_blocks if op == "block" else taken_ops).add(l if op == "block" else (l, op))
        if len(greedy) == args.greedy_max:
            break
    top = ranked[:args.pair_top]
    pair_sets = [(a, b) for a, b in itertools.combinations(top, 2)
                 if not (a["layer"] == b["layer"] and "block" in (a["op"], b["op"]))]

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    dev = torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    need = sorted({r["layer"] for r in greedy + top if r["mode"] == "linear"})
    fit_chunks = wikitext_chunks(tok, "train", args.seq_len, args.fit_seqs)
    preds = {}
    for i in range(0, len(need), args.fit_group):
        preds |= atlas.fit_group(model, layers, need[i:i + args.fit_group], fit_chunks, dev, d)
        torch.cuda.empty_cache()

    member = lambda r: (r["layer"], r["op"], r["mode"])  # noqa: E731
    sets = [[member(a), member(b)] for a, b in pair_sets] + [[member(r) for r in greedy[:n]] for n in range(1, len(greedy) + 1)]
    calib = wikitext_chunks(tok, "train", args.seq_len, args.calib_seqs, skip=args.calib_skip)
    hold = wikitext_chunks(tok, "train", args.seq_len, args.calib_seqs, skip=args.holdout_skip)
    res_c, ppl_c, n_c = evaluate(model, layers, calib, dev, sets, preds)
    res_h, ppl_h, n_h = evaluate(model, layers, hold, dev, sets[len(pair_sets):], preds)

    out_rows = []
    for i, (a, b) in enumerate(pair_sets):
        m = res_c[i]
        out_rows.append({"kind": "pair", "n": 2, "members": f"{a['layer']}:{a['op']}:{a['mode']} {b['layer']}:{b['op']}:{b['mode']}",
                         "saved_frac_decoder": a["saved"] + b["saved"], "kl": m["kl"], "kl_sum_singles": a["kl"] + b["kl"],
                         "ppl_change_pct": m["ppl_change_pct"], "ppl_change_sum_singles": a["ppl_change_pct"] + b["ppl_change_pct"],
                         "top1_agreement": m["top1_agreement"], "holdout_kl": None, "holdout_ppl_change_pct": None})
    for n in range(1, len(greedy) + 1):
        m, h = res_c[len(pair_sets) + n - 1], res_h[n - 1]
        g = greedy[:n]
        out_rows.append({"kind": "greedy", "n": n, "members": " ".join(f"{r['layer']}:{r['op']}:{r['mode']}" for r in g),
                         "saved_frac_decoder": sum(r["saved"] for r in g), "kl": m["kl"], "kl_sum_singles": sum(r["kl"] for r in g),
                         "ppl_change_pct": m["ppl_change_pct"], "ppl_change_sum_singles": sum(r["ppl_change_pct"] for r in g),
                         "top1_agreement": m["top1_agreement"], "holdout_kl": h["kl"], "holdout_ppl_change_pct": h["ppl_change_pct"]})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_name(cfg) + "_allocation_" + "_".join(args.modes)
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "atlas": args.atlas, "modes": args.modes,
            "ops": args.ops, "dataset": WIKITEXT, "dataset_revision": WIKITEXT_REV, "seq_len": args.seq_len,
            "calib_split": f"train chunks {args.calib_skip}-{args.calib_skip + args.calib_seqs - 1}",
            "holdout_split": f"train chunks {args.holdout_skip}-{args.holdout_skip + args.calib_seqs - 1}",
            "exact_ppl_calib": ppl_c, "exact_ppl_holdout": ppl_h, "eval_tokens": n_c, "budget_pct": args.budget_pct,
            "ranking": "atlas kl / saved_frac_decoder, best mode per (layer, op)"}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for r in out_rows:
        print(f"{r['kind']:<6} n={r['n']:>2} saved {100 * r['saved_frac_decoder']:5.1f}%  kl {r['kl']:.4f} (sum {r['kl_sum_singles']:.4f})  "
              f"ppl {r['ppl_change_pct']:+6.2f}% (sum {r['ppl_change_sum_singles']:+6.2f}%)"
              + (f"  holdout ppl {r['holdout_ppl_change_pct']:+6.2f}%" if r["holdout_ppl_change_pct"] is not None else "")
              + (f"  [{r['members']}]" if r["kind"] == "pair" else ""), flush=True)


if __name__ == "__main__":
    main()
