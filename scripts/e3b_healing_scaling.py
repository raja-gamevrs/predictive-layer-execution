"""E3b: healing scaling curve, "PLE as architecture" (docs/e3b_healing_scaling.md).

One continuous training run per setting. The student runs with uncorrected identity speculation at
every boundary 0..34 (block l+1's q/k/v from RMSNorm_(l+1)(h_l); position 0 exact) and is trained by
KL distillation from the original exact model (adapters disabled, no speculation), recomputed each
step. Adapters are weight deltas inside each projection's own forward (as scripts/e3_healing.py), so
the speculative path goes through them. Evaluated at fixed checkpoints on calibration and holdout,
speculation on and off, against the original exact model (reference recomputed per chunk, so no
log-prob cache is held on the GPU).

Modes: `lora` (rank r on every projection of every block) and `qkv_delta` (a full-rank BF16 trainable
delta on q/k/v, LoRA rank r on the other projections; memory-lean full-rank variant).

`--plot-only` rebuilds the SVG plot from the JSON files in --out.
"""

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e_common import CAL, FIT, HOLD, Quality, chunks, gated_speculation  # noqa: E402

from ple.model_loader import decoder_layers, load, load_config, run_name
from ple.speculation_policy import is_projection

ADAPT = {"on": False}
N_TRAIN_CHUNKS = 4916            # all 512-token chunks of WikiText-2 train with this tokenizer
EXCLUDE = [FIT, CAL, HOLD]       # (start, count) spans never trained on
QKV = ("q_proj", "k_proj", "v_proj")


def add_adapters(layers, mode: str, rank: int, scale: float):
    """Patch every projection's forward with a delta that is active while ADAPT['on']."""
    lora, delta = [], []
    for layer in layers:
        for name, mod in layer.named_modules():
            if not (is_projection(mod) and isinstance(mod, torch.nn.Linear)):
                continue
            dev = mod.weight.device
            if mode == "qkv_delta" and name.split(".")[-1] in QKV:
                mod.delta = torch.nn.Parameter(torch.zeros(mod.out_features, mod.in_features, device=dev,
                                                           dtype=mod.weight.dtype))
                delta.append(mod.delta)

                def forward(self, x):
                    y = F.linear(x, self.weight, self.bias)
                    if ADAPT["on"]:
                        y = y + F.linear(x, self.delta.to(x.dtype))
                    return y
            else:
                mod.lora_a = torch.nn.Parameter(torch.randn(rank, mod.in_features, device=dev) / mod.in_features**0.5)
                mod.lora_b = torch.nn.Parameter(torch.zeros(mod.out_features, rank, device=dev))
                mod.lora_scale = scale
                lora += [mod.lora_a, mod.lora_b]

                def forward(self, x):
                    y = F.linear(x, self.weight, self.bias)
                    if ADAPT["on"]:
                        y = y + F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype)) * self.lora_scale
                    return y
            mod.forward = types.MethodType(forward, mod)
    return lora, delta


class FactoredRMS(torch.optim.Optimizer):
    """Adafactor-style update for 2-D params with no first moment and a fixed lr: the second moment is
    kept as row and column EMAs (O(m + n) state instead of Adam's 2·m·n), update RMS clipped to 1.
    Used for the full-rank q/k/v delta, where Adam's two state tensors would not fit the memory cap."""

    def __init__(self, params, lr, beta2=0.999, eps=1e-30, clip=1.0):
        super().__init__(params, {"lr": lr, "beta2": beta2, "eps": eps, "clip": clip})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["r"] = torch.zeros(p.shape[0], device=p.device)
                    st["c"] = torch.zeros(p.shape[1], device=p.device)
                st["t"] += 1
                g2 = p.grad.float().square_() + g["eps"]
                st["r"].mul_(g["beta2"]).add_(g2.mean(1), alpha=1 - g["beta2"])
                st["c"].mul_(g["beta2"]).add_(g2.mean(0), alpha=1 - g["beta2"])
                del g2
                bc = 1 - g["beta2"] ** st["t"]
                v = torch.outer(st["r"] / bc, st["c"] / bc) / (st["r"].mean() / bc)
                u = p.grad.float().div_(v.sqrt_())
                del v
                u.div_((u.square().mean().sqrt() / g["clip"]).clamp_(min=1.0))
                p.add_(u.to(p.dtype), alpha=-g["lr"])


def train_chunks(tok):
    ids = chunks(tok, (0, N_TRAIN_CHUNKS))
    bad = {i for s, n in EXCLUDE for i in range(s, s + n)}
    keep = [i for i in range(len(ids)) if i not in bad]
    return [ids[i] for i in keep], keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--mode", choices=["lora", "qkv_delta"], default="lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=None, help="LoRA alpha/r; default 2*sqrt(16/r)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--delta-lr", type=float, default=2e-5, help="lr of the full-rank q/k/v delta")
    ap.add_argument("--delta-opt", choices=["adamw", "factored"], default="adamw",
                    help="optimiser of the q/k/v delta (adamw keeps two BF16 state tensors per weight)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 16000])
    ap.add_argument("--cooldown", type=int, default=1000, help="extra steps, lr -> 0, after the last checkpoint")
    ap.add_argument("--eval-chunks", type=int, default=16, help="chunks per split (16 = the full split)")
    ap.add_argument("--mem-frac", type=float, default=0.5)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results/processed/e1_e5/e3b")
    ap.add_argument("--save-adapters", default="results/raw/e1_e5/e3b")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if args.plot_only:
        plot(out)
        return

    torch.cuda.set_per_process_memory_fraction(args.mem_frac, 0)
    scale = args.scale if args.scale is not None else 2 * math.sqrt(16 / args.rank)
    tag = args.tag or (f"lora_r{args.rank}" if args.mode == "lora" else f"qkv_delta_r{args.rank}")
    cfg = load_config(args.config)
    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    model, tok = load(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = decoder_layers(model)
    n, dev = len(layers), torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    bounds = list(range(n - 1))
    pred = {l: (lambda h: h) for l in bounds}
    lora, delta = add_adapters(layers, args.mode, args.rank, scale)
    print(f"{tag}: {sum(p.numel() for p in lora) / 1e6:.1f}M LoRA + {sum(p.numel() for p in delta) / 1e6:.1f}M "
          f"delta params, scale {scale:.3f}, boundaries {bounds[0]}..{bounds[-1]}", flush=True)

    train, train_idx = train_chunks(tok)
    evals = {"cal": chunks(tok, (CAL[0], args.eval_chunks)), "hold": chunks(tok, (HOLD[0], args.eval_chunks))}

    def evaluate(split):
        q_on, q_off, nll, cnt = Quality(), Quality(), 0.0, 0
        with torch.inference_mode():
            for ids in evals[split]:
                ids = ids[None].to(dev)
                ADAPT["on"] = False
                lp_ref = Quality.logprobs(model(input_ids=ids, use_cache=False).logits)
                nll += -lp_ref.gather(-1, ids[0, 2:, None]).sum().item()
                cnt += ids.shape[1] - 2
                ADAPT["on"] = True
                with gated_speculation(layers, bounds, pred, d):
                    q_on.add(lp_ref, Quality.logprobs(model(input_ids=ids, use_cache=False).logits), ids)
                q_off.add(lp_ref, Quality.logprobs(model(input_ids=ids, use_cache=False).logits), ids)
                del lp_ref
        exact_ppl = math.exp(nll / cnt)
        return q_on.row(exact_ppl), q_off.row(exact_ppl), exact_ppl

    groups = [{"params": lora, "lr": args.lr}]
    if delta and args.delta_opt == "adamw":
        groups.append({"params": delta, "lr": args.delta_lr})
    opts = [torch.optim.AdamW(groups, weight_decay=0.0)]
    if delta and args.delta_opt == "factored":
        opts.append(FactoredRMS(delta, lr=args.delta_lr))
    total = args.checkpoints[-1]
    end = total + args.cooldown

    def lr_factor(s):
        if s < args.warmup:
            return (s + 1) / args.warmup
        if s < total:
            return 1.0
        return max(0.0, (end - s) / max(1, args.cooldown))
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, lr_factor) for o in opts]
    sched = scheds[0]

    rows = []
    meta = {
        "experiment": "E3b healing scaling", "model": cfg["model"]["id"], "revision": cfg["model"]["revision"],
        "tag": tag, "mode": args.mode, "rank": args.rank, "lora_scale": scale, "lr": args.lr,
        "delta_lr": args.delta_lr if delta else None, "delta_opt": args.delta_opt if delta else None,
        "warmup": args.warmup,
        "schedule": "linear warmup then constant; cooldown = linear decay to 0 after the last checkpoint",
        "cooldown_steps": args.cooldown, "seq_len": 512, "tokens_per_step": 512, "predictor": "identity",
        "boundaries": bounds, "tau": None, "train_split": "wikitext-2 train",
        "train_chunks": len(train), "excluded_chunk_spans": EXCLUDE, "eval_chunks_per_split": args.eval_chunks,
        "n_lora_params": sum(p.numel() for p in lora), "n_delta_params": sum(p.numel() for p in delta),
        "bar": {"kl": 0.01, "ppl_change_pct": 1.0}, "rows": rows}
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_name(cfg)}_e3b_{tag}.json"

    def checkpoint(step, label, train_kl, train_s):
        t = time.time()
        for split in ("cal", "hold"):
            on, off, exact_ppl = evaluate(split)
            row = {"setting": tag, "label": label, "step": step, "tokens": step * 512, "split": split,
                   "epochs": step / len(train), "exact_ppl": exact_ppl,
                   "kl": on["kl"], "top1": on["top1"], "ppl": on["ppl"], "ppl_change_pct": on["ppl_change_pct"],
                   "nospec_kl": off["kl"], "nospec_top1": off["top1"], "nospec_ppl_change_pct": off["ppl_change_pct"],
                   "train_kl_last100": train_kl, "train_seconds": round(train_s, 1),
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            rows.append(row)
            print(f"[ckpt {label:>10}] {split:<4} spec on: KL {on['kl']:.4f} top1 {on['top1']:.4f} "
                  f"ppl {on['ppl_change_pct']:+.2f}% | spec off: KL {off['kl']:.4f} ppl {off['ppl_change_pct']:+.2f}%",
                  flush=True)
        meta["eval_seconds_last"] = round(time.time() - t, 1)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    checkpoint(0, "0", None, 0.0)
    ckpts = set(args.checkpoints)
    trainable = lora + delta
    gen_epoch, order = -1, None
    losses, t_train, t_last = [], 0.0, time.time()
    for step in range(end):
        epoch, pos = divmod(step, len(train))
        if epoch != gen_epoch:
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            gen_epoch = epoch
        ids = train[order[pos]][None].to(dev)
        t0 = time.time()
        ADAPT["on"] = False
        with torch.no_grad():
            teacher = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, 1:].float(), -1)
        ADAPT["on"] = True
        with gated_speculation(layers, bounds, pred, d):
            student = torch.log_softmax(model(input_ids=ids, use_cache=False).logits[0, 1:].float(), -1)
        loss = (teacher.exp() * (teacher - student)).sum(-1).mean()
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        for o, s in zip(opts, scheds):
            o.step()
            s.step()
        del teacher, student
        losses.append(loss.item())
        t_train += time.time() - t0
        done = step + 1
        if done % 100 == 0 or done == 1:
            run = sum(losses[-100:]) / len(losses[-100:])
            print(f"{tag} step {done:>6} train KL(last100) {run:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{t_train / done:.3f} s/step  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB  "
                  f"wall {time.time() - t_last:.0f} s", flush=True)
        if done in ckpts:
            checkpoint(done, f"{done // 1000}k" if done % 1000 == 0 else str(done),
                       sum(losses[-100:]) / len(losses[-100:]), t_train)
    if args.cooldown:
        checkpoint(end, f"{total // 1000}k+{args.cooldown // 1000}k_cooldown" if total % 1000 == 0 else str(end),
                   sum(losses[-100:]) / len(losses[-100:]), t_train)
    meta["seconds_per_train_step"] = t_train / end
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.save_adapters:
        sd = Path(args.save_adapters)
        sd.mkdir(parents=True, exist_ok=True)
        torch.save([p.detach().cpu() for p in trainable], sd / f"{run_name(cfg)}_e3b_{tag}_adapters.pt")
    print(f"wrote {path}", flush=True)


# ---------- plot (plain SVG; matplotlib is not installed in this environment) ----------

def plot(out: Path) -> None:
    runs = []
    for f in sorted(out.glob("*_e3b_*.json")):
        meta = json.loads(f.read_text(encoding="utf-8"))
        rows = [r for r in meta["rows"] if r["split"] == "hold" and "cooldown" not in r["label"]]
        cool = [r for r in meta["rows"] if r["split"] == "hold" and "cooldown" in r["label"]]
        if len(rows) >= 2 and meta["rows"][0]["tokens"] == 0 and max(r["step"] for r in rows) >= 1000:
            runs.append((meta["tag"], rows, cool))
    if not runs:
        print("nothing to plot")
        return
    colors = ["#2a6fdb", "#d9480f", "#2b8a3e", "#862e9c"]
    W, H, ml, mr, mt, mb = 980, 420, 70, 20, 40, 60
    gap = 90
    pw = (W - ml - mr - gap) / 2
    x0_tok, x1_tok = 2 ** 18, 2 ** 23.4   # step 0 is drawn at the left edge (axis break, labelled 0)

    def xs(tok, left):
        t = max(tok, x0_tok)
        return left + pw * (math.log(t) - math.log(x0_tok)) / (math.log(x1_tok) - math.log(x0_tok))

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W / 2}" y="20" text-anchor="middle" font-size="14">E3b, Qwen3-4B-Base, identity speculation at all 35 '
           f'boundaries, holdout (speculation on: solid; off: dashed)</text>']
    panels = [("kl", "nospec_kl", "mean KL vs exact (nats/token, log)", 0.01, True),
              ("ppl_change_pct", "nospec_ppl_change_pct", "perplexity change (%, log)", 1.0, True)]
    for pi, (key, key_off, ylabel, bar, logy) in enumerate(panels):
        left = ml + pi * (pw + gap)
        vals = [r[k] for _, rows, cool in runs for r in rows + cool for k in (key, key_off) if r[k] > 0 and r["step"] > 0]
        vals += [r[key] for _, rows, _ in runs for r in rows] + [bar]
        lo, hi = min(vals) / 1.5, max(vals) * 1.5
        lo, hi = 10 ** math.floor(math.log10(lo)), 10 ** math.ceil(math.log10(hi))

        def ys(v):
            v = max(v, lo)
            return mt + (H - mt - mb) * (1 - (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        svg.append(f'<rect x="{left}" y="{mt}" width="{pw}" height="{H - mt - mb}" fill="none" stroke="#888"/>')
        e = math.floor(math.log10(lo))
        while 10 ** e <= hi * 1.0001:
            for m in (1, 2, 5):
                v = m * 10 ** e
                if lo <= v <= hi:
                    y = ys(v)
                    svg.append(f'<line x1="{left}" x2="{left + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eee"/>')
                    svg.append(f'<text x="{left - 4}" y="{y + 4:.1f}" text-anchor="end">{v:g}</text>')
            e += 1
        for tok, lab in ((2 ** 18, "0"), (2 ** 19, "0.5M"), (2 ** 20, "1M"), (2 ** 21, "2M"), (2 ** 22, "4M"),
                         (2 ** 23, "8M")):
            x = xs(tok, left)
            svg.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{mt}" y2="{H - mb}" stroke="#eee"/>')
            svg.append(f'<text x="{x:.1f}" y="{H - mb + 16}" text-anchor="middle">{lab}</text>')
        svg.append(f'<text x="{left + pw / 2}" y="{H - mb + 34}" text-anchor="middle">tokens trained (log; 512 per step)</text>')
        svg.append(f'<text x="{left - 52}" y="{(mt + H - mb) / 2}" text-anchor="middle" '
                   f'transform="rotate(-90 {left - 52} {(mt + H - mb) / 2})">{ylabel}</text>')
        yb = ys(bar)
        svg.append(f'<line x1="{left}" x2="{left + pw}" y1="{yb:.1f}" y2="{yb:.1f}" stroke="black" stroke-width="1.5" '
                   f'stroke-dasharray="2,3"/><text x="{left + pw - 4}" y="{yb - 5:.1f}" text-anchor="end">bar {bar:g}</text>')
        for ri, (tag, rows, cool) in enumerate(runs):
            c = colors[ri % len(colors)]
            for k, dash in ((key, ""), (key_off, ' stroke-dasharray="6,4"')):
                pr = rows if k == key else [r for r in rows if r["step"] > 0]   # spec off at step 0 is exactly 0
                pts = " ".join(f"{xs(r['tokens'], left):.1f},{ys(r[k]):.1f}" for r in pr)
                svg.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
                for r in pr:
                    svg.append(f'<circle cx="{xs(r["tokens"], left):.1f}" cy="{ys(r[k]):.1f}" r="3" fill="{c}"/>')
                for r in cool:
                    svg.append(f'<rect x="{xs(r["tokens"], left) - 4:.1f}" y="{ys(r[k]) - 4:.1f}" width="8" height="8" '
                               f'fill="none" stroke="{c}" stroke-width="1.5"/>')
            if pi == 0:
                svg.append(f'<text x="{left + 10}" y="{mt + 16 + 15 * ri}" fill="{c}">{tag}</text>')
    svg.append(f'<text x="{ml}" y="{H - 8}" font-size="11">open squares: +1k-step cooldown (lr decayed to 0) '
               f'after 16k constant-lr steps; x = 0 is the untrained model (axis break); bar = project gate</text>')
    svg.append("</svg>")
    dst = out / "e3b_healing_scaling.svg"
    dst.write_text("\n".join(svg), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
