"""E3: healing — a short LoRA fine-tune with speculation on (docs/experiments_e1_e5.md).

LoRA (rank r, all q/k/v/o and MLP projections of every block) is trained to match the exact,
unmodified model's next-token distribution (KL(p_exact || p_student)) while the student runs with
uncorrected speculation at a boundary set, ridge predictors frozen. The LoRA is a weight delta
inside each projection's own forward, so the speculative path goes through it too. Quality is
measured against the original exact model on CAL and HOLD, with speculation on and off.
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e_common import CAL, HEAL, HOLD, Quality, chunks, exact_logprobs, gated_speculation, ridge  # noqa: E402

from ple.model_loader import decoder_layers, load, load_config, run_name
from ple.speculation_policy import is_projection

LORA = {"on": False}


def add_lora(model, layers, rank: int, alpha: float):
    params = []
    for layer in layers:
        for name, mod in layer.named_modules():
            if not (is_projection(mod) and isinstance(mod, torch.nn.Linear)):
                continue
            mod.lora_a = torch.nn.Parameter(torch.randn(rank, mod.in_features, device=mod.weight.device) / mod.in_features**0.5)
            mod.lora_b = torch.nn.Parameter(torch.zeros(mod.out_features, rank, device=mod.weight.device))
            mod.lora_scale = alpha / rank
            params += [mod.lora_a, mod.lora_b]

            def forward(self, x):
                y = F.linear(x, self.weight, self.bias)
                if LORA["on"]:
                    y = y + F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype)) * self.lora_scale
                return y
            mod.forward = types.MethodType(forward, mod)
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sets", nargs="+", default=["tol9", "start16", "start4_tail2", "all"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=1000)
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
    pred = ridge(model, layers, tok, stem, dev)
    pred_f = {l: (lambda h, p=p: p(h)) for l, p in pred.items()}
    all_sets = {"tol9": list(range(24, n - 3)), "start16": list(range(16, n - 3)),
                "start4_tail2": list(range(4, n - 3)), "all": list(range(n - 1))}
    splits = {name: exact_logprobs(model, tok, span, dev) for name, span in (("cal", CAL), ("hold", HOLD))}
    lora = add_lora(model, layers, args.rank, args.alpha)
    init = [p.detach().clone() for p in lora]
    heal_ids = chunks(tok, (HEAL[0], args.steps))

    def evaluate(split, bounds):
        ids_list, lps, exact_ppl = splits[split]
        q = Quality()
        with torch.inference_mode(), gated_speculation(layers, bounds, pred_f, d):
            for ids, lp_exact in zip(ids_list, lps):
                q.add(lp_exact, Quality.logprobs(model(input_ids=ids).logits), ids)
        return q.row(exact_ppl)

    rows = []
    for set_name in args.sets:
        bounds = all_sets[set_name]
        with torch.no_grad():
            for p, p0 in zip(lora, init):
                p.copy_(p0)
        LORA["on"] = False
        before = {s: evaluate(s, bounds) for s in ("cal", "hold")}
        opt = torch.optim.AdamW(lora, lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=max(0.05, 3 / args.steps))
        t0, losses = time.time(), []
        for step, ids in enumerate(heal_ids):
            ids = ids[None].to(dev)
            LORA["on"] = False
            with torch.no_grad():
                teacher = torch.log_softmax(model(input_ids=ids).logits[0, 1:].float(), -1)
            LORA["on"] = True
            with gated_speculation(layers, bounds, pred_f, d):
                student = torch.log_softmax(model(input_ids=ids).logits[0, 1:].float(), -1)
            loss = (teacher.exp() * (teacher - student)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora, 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            if step % 100 == 0 or step == args.steps - 1:
                print(f"{set_name} step {step:>4} KL {sum(losses[-100:]) / len(losses[-100:]):.4f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
        LORA["on"] = True
        after = {s: evaluate(s, bounds) for s in ("cal", "hold")}
        no_spec = {s: evaluate(s, []) for s in ("cal", "hold")}
        for s in ("cal", "hold"):
            row = {"set": set_name, "blocks": len(bounds), "split": s, "before_kl": before[s]["kl"],
                   "before_ppl_change_pct": before[s]["ppl_change_pct"], "healed_kl": after[s]["kl"],
                   "healed_top1": after[s]["top1"], "healed_ppl_change_pct": after[s]["ppl_change_pct"],
                   "healed_nospec_kl": no_spec[s]["kl"], "healed_nospec_ppl_change_pct": no_spec[s]["ppl_change_pct"]}
            rows.append(row)
            print(f"{set_name:<13} {s:<4} before KL {row['before_kl']:.4f} ({row['before_ppl_change_pct']:+.2f}%)  "
                  f"healed KL {row['healed_kl']:.4f} ({row['healed_ppl_change_pct']:+.2f}%)  "
                  f"healed, no spec KL {row['healed_nospec_kl']:.4f} ({row['healed_nospec_ppl_change_pct']:+.2f}%)",
                  flush=True)
        rows[-1]["train_seconds"] = round(time.time() - t0)
        rows[-1]["final_train_kl"] = sum(losses[-100:]) / len(losses[-100:])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "rank": args.rank, "alpha": args.alpha,
            "lr": args.lr, "steps": args.steps, "seq_len": 512, "heal_chunks": [HEAL[0], HEAL[0] + args.steps],
            "exact_ppl": {k: v[2] for k, v in splits.items()}, "rows": rows}
    (out / f"{stem}_e3_healing.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
