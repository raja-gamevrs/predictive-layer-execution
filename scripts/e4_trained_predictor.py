"""E4: small trained next-state predictors vs the ridge map (docs/experiments_e1_e5.md).

Per boundary l, ĥ = h + h·A + b + MLP_r(h), with A, b the ridge solution and MLP_r a SiLU bottleneck
d -> r -> d whose output layer starts at zero (so training starts from ridge). Variants:
  ridge          ridge as fitted (raw residual MSE)
  linear_norm    A, b trained further on the normed loss, no MLP
  mlp_r          A, b and MLP_r trained on the normed loss
Loss: ||N_(l+1)(h_(l+1)) - N_(l+1)(ĥ)||² / ||N_(l+1)(h_(l+1))||², the relative error at the next
block's projection input. Train on FIT traces, report median relative error on HOLD traces, then the
output quality with every boundary speculated (uncorrected) using each variant, on CAL and HOLD,
and the E2 gate rerun with the best variant.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e_common import CACHE, CAL, FIT, HOLD, Quality, chunks, exact_logprobs, gated_speculation, ridge  # noqa: E402

from ple.activation_capture import capture_residual_stream
from ple.model_loader import decoder_layers, load, load_config, run_name


class Predictor(torch.nn.Module):
    def __init__(self, w, b, rank):
        super().__init__()
        self.w, self.b = torch.nn.Parameter(w.clone().float()), torch.nn.Parameter(b.clone().float())
        self.rank = rank
        if rank:
            d = w.shape[0]
            self.down = torch.nn.Linear(d, rank)
            self.up = torch.nn.Linear(rank, d)
            torch.nn.init.zeros_(self.up.weight)
            torch.nn.init.zeros_(self.up.bias)

    def forward(self, h):
        x = h.float()
        out = x + x @ self.w + self.b
        if self.rank:
            out = out + self.up(F.silu(self.down(x)))
        return out


def rel_err(norm, h_next, h_hat):
    t = norm(h_next.float())
    return (t - norm(h_hat)).norm(dim=-1) / t.norm(dim=-1).clamp(min=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--fit-seqs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4, help="for A, b")
    ap.add_argument("--mlp-lr", type=float, default=1e-3)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    ap.add_argument("--no-quality", action="store_true",
                    help="stop after the error comparison (quality runs only matter if a variant beats ridge)")
    ap.add_argument("--out", default="results/processed/e1_e5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = decoder_layers(model)
    n, dev = len(layers), torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    stem = run_name(cfg)
    base = ridge(model, layers, tok, stem, dev)
    variants = ["ridge", "linear_norm"] + [f"mlp_{r}" for r in args.ranks]

    def capture(span, ls, n_seqs=None):
        """Stack (h_l, h_(l+1)) for boundaries ls over a span's chunks, positions 1.. (BF16, on GPU)."""
        skip, cnt = span
        xs = {l: [] for l in ls}
        with torch.inference_mode():
            for ids in chunks(tok, (skip, n_seqs or cnt)):
                hs = capture_residual_stream(model, layers, input_ids=ids[None].to(dev))
                for l in ls:
                    xs[l].append(torch.stack([hs[l][0, 1:], hs[l + 1][0, 1:]]))
        return {l: torch.cat(v, dim=1) for l, v in xs.items()}

    trained = {v: {} for v in variants}
    err_rows = []
    for g in range(0, n - 1, args.group):
        ls = list(range(g, min(g + args.group, n - 1)))
        train, hold = capture(FIT, ls, args.fit_seqs), capture(HOLD, ls)
        for l in ls:
            norm = layers[l + 1].input_layernorm
            x, y = train[l][0], train[l][1]
            hx, hy = hold[l][0].clone(), hold[l][1].clone()
            row = {"boundary": l}
            for v in variants:
                rank = int(v.split("_")[1]) if v.startswith("mlp") else 0
                p = Predictor(base[l].weight, base[l].bias, rank).to(dev)
                if v != "ridge":
                    groups = [{"params": [p.w, p.b], "lr": args.lr}]
                    if rank:
                        groups.append({"params": [*p.down.parameters(), *p.up.parameters()], "lr": args.mlp_lr})
                    opt = torch.optim.Adam(groups)
                    steps = args.epochs * (x.shape[0] // args.batch)
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
                    gen = torch.Generator(device=dev).manual_seed(l)
                    for _ in range(args.epochs):
                        perm = torch.randperm(x.shape[0], device=dev, generator=gen)
                        for i in range(0, x.shape[0] - args.batch + 1, args.batch):
                            idx = perm[i:i + args.batch]
                            with torch.enable_grad():
                                loss = rel_err(norm, y[idx], p(x[idx])).pow(2).mean()
                                opt.zero_grad(set_to_none=True)
                                loss.backward()
                            opt.step()
                            sched.step()
                p.eval()
                for prm in p.parameters():
                    prm.requires_grad_(False)
                with torch.no_grad():
                    e = torch.cat([rel_err(norm, hy[i:i + 4096], p(hx[i:i + 4096])) for i in range(0, hx.shape[0], 4096)])
                row[v] = e.median().item()
                trained[v][l] = p
            err_rows.append(row)
            print(f"l={l:>2} " + "  ".join(f"{v} {row[v]:.3f}" for v in variants), flush=True)
        del train, hold
        torch.cuda.empty_cache()

    med = {v: float(torch.tensor([r[v] for r in err_rows]).median()) for v in variants}
    print("median over boundaries:", json.dumps(med), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "fit_seqs": args.fit_seqs,
            "epochs": args.epochs, "batch": args.batch, "lr": args.lr, "mlp_lr": args.mlp_lr, "ranks": args.ranks,
            "median_rel_err": med, "per_boundary": err_rows}
    if args.no_quality:
        (out / f"{stem}_e4_trained_predictor.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return
    splits = {name: exact_logprobs(model, tok, span, dev) for name, span in (("cal", CAL), ("hold", HOLD))}
    every = list(range(n - 1))

    def callables(v):
        return {l: (lambda h, m=trained[v][l]: m(h).to(h.dtype)) for l in every}

    def evaluate(v, split, tau=None):
        ids_list, lps, exact_ppl = splits[split]
        q, stats = Quality(), {"spec": 0, "total": 0}
        with torch.inference_mode(), gated_speculation(layers, every, callables(v), d, tau=tau, stats=stats):
            for ids, lp_exact in zip(ids_list, lps):
                q.add(lp_exact, Quality.logprobs(model(input_ids=ids).logits), ids)
        stats.pop("r")
        return q.row(exact_ppl) | {"spec_share": stats["spec"] / stats["total"]}

    quality = []
    for v in variants:
        for split in ("cal", "hold"):
            r = {"variant": v, "split": split, "tau": None} | evaluate(v, split)
            quality.append(r)
            print(f"all blocks {v:<12} {split:<4} KL {r['kl']:.4f} top1 {r['top1']:.3f} ppl {r['ppl_change_pct']:+.2f}%",
                  flush=True)
    best = min(variants[1:], key=lambda v: med[v])
    for tau in args.taus:
        for v in ("ridge", best):
            for split in ("cal", "hold"):
                r = {"variant": v, "split": split, "tau": tau} | evaluate(v, split, tau)
                quality.append(r)
                print(f"gate tau {tau} {v:<12} {split:<4} spec {r['spec_share']:.3f} KL {r['kl']:.4f} "
                      f"ppl {r['ppl_change_pct']:+.2f}%", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({v: {l: m.state_dict() for l, m in trained[v].items()} for v in variants[1:]},
               CACHE / f"{stem}_e4_predictors.pt")
    meta |= {"best": best, "exact_ppl": {k: v[2] for k, v in splits.items()}, "quality": quality}
    (out / f"{stem}_e4_trained_predictor.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
