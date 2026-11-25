"""E2: per-token gate on uncorrected speculation (docs/experiments_e1_e5.md).

At each speculated boundary, tokens whose predicted normed input is far from the true one
(r = ||N(h) - N(ĥ)|| / ||N(h)|| > tau) use the exact projection; the rest keep the speculative one.
Sweeps tau; reports quality and the share of (token, block) pairs left speculated, on calibration
and holdout, against a random gate at the same share (does the error signal matter?).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e_common import CAL, HOLD, Quality, exact_logprobs, gated_speculation, ridge  # noqa: E402

from ple.model_loader import decoder_layers, load, load_config, run_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 1e9])
    ap.add_argument("--out", default="results/processed/e1_e5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    n, dev = len(layers), torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    stem = run_name(cfg)
    pred = ridge(model, layers, tok, stem, dev)
    sets = {"all": list(range(n - 1)), "start4_tail2": list(range(4, n - 3))}
    splits = {name: exact_logprobs(model, tok, span, dev) for name, span in (("cal", CAL), ("hold", HOLD))}

    rows = []

    def evaluate(split, bounds, **kw):
        ids_list, lps, exact_ppl = splits[split]
        q, stats = Quality(), {"spec": 0, "total": 0}
        with torch.inference_mode(), gated_speculation(layers, bounds, pred, d, stats=stats, **kw):
            for ids, lp_exact in zip(ids_list, lps):
                q.add(lp_exact, Quality.logprobs(model(input_ids=ids).logits), ids)
        r = torch.cat(stats.pop("r"))
        return q.row(exact_ppl) | {"spec_share": stats["spec"] / stats["total"], "r_median": r.median().item()}

    for set_name, bounds in sets.items():
        for tau in args.taus:
            for split in ("cal", "hold"):
                row = {"set": set_name, "blocks": len(bounds), "gate": "error", "tau": tau, "split": split}
                row |= evaluate(split, bounds, tau=tau)
                rows.append(row)
                print(f"{set_name:<13} tau {tau:<6g} {split:<4} spec {row['spec_share']:.3f}  KL {row['kl']:.4f}  "
                      f"top1 {row['top1']:.3f}  ppl {row['ppl_change_pct']:+.2f}%", flush=True)
                share = row["spec_share"]
                if 0.02 < share < 0.98:
                    gen = torch.Generator(device=dev).manual_seed(0)
                    rand = {"set": set_name, "blocks": len(bounds), "gate": "random", "tau": tau, "split": split}
                    rand |= evaluate(split, bounds, fallback=lambda l, r: torch.rand(r.shape, device=r.device,
                                                                                    generator=gen) > share)
                    rows.append(rand)
                    print(f"{'':<13} random at same share: spec {rand['spec_share']:.3f}  KL {rand['kl']:.4f}  "
                          f"ppl {rand['ppl_change_pct']:+.2f}%", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"],
            "exact_ppl": {k: v[2] for k, v in splits.items()}, "rows": rows}
    (out / f"{stem}_e2_token_gate.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
