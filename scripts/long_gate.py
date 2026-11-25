"""Follow-up I check of the new long-context code paths, before any long-context timing on that model.

  --against hf    (Gemma 4 12B: ring-buffer sliding attention) this repo's decoder against Transformers:
                  prompt (C tokens) + our 128-token greedy continuation scored teacher-forced by both
                  (Transformers with logits_to_keep), KL / top-1 / violations of our output, as the gate
  --against self  (Qwen3.8: chunked DeltaNet prefill) chunked prefill + extension against this repo's
                  gate-checked one-shot prefill over the same sequence (Transformers and the decoder do
                  not fit together)
Gate as in follow-up E after its amendment: KL <= 0.005, top-1 >= 99%, violations <= the reference's own.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decoder_gate import judge  # noqa: E402
from lever_bench import mode  # noqa: E402
from long_bench import gen, long_prompts  # noqa: E402

from ple.gemma4_decode import make_decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402


def kl_top1(lo, lr):
    kl = torch.nn.functional.kl_div(torch.log_softmax(lo, -1), torch.log_softmax(lr, -1), log_target=True, reduction="none").sum(-1)
    return kl.mean().item(), kl.max().item(), (lo.argmax(-1) == lr.argmax(-1)).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gemma4_12b.yaml")
    ap.add_argument("--against", choices=["hf", "self"], required=True)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    C, n = args.context, args.n
    dec = make_decoder(model, C + n + 64)
    rows = []
    with torch.inference_mode():
        for dom, prs in long_prompts(tok, C, args.prompts, dec.dev).items():
            for i, pr in enumerate(prs):
                mode(dec, "sdpa", "ref")
                first = int(dec.prefill_chunked(pr)[-1].argmax())
                rep, rst, buf = dec.make_step(C, first)
                dec.prefill_chunked(pr)
                toks = torch.tensor(gen(rep, rst, buf, first, n), device=dec.dev)
                del rep, rst, buf
                seq = torch.cat([pr, toks[:-1]])
                lo = dec.prefill_chunked(seq, logits_rows=n).float()
                if args.against == "hf":
                    lr = model(input_ids=seq[None], use_cache=False, logits_to_keep=n).logits[0].float()
                else:
                    lr = dec.prefill(seq)[-n:].float()
                kl, klmax, top1 = kl_top1(lo, lr)
                v, t = judge(lr, toks)
                r = {"domain": dom, "prompt": i, "kl": kl, "kl_max": klmax, "top1": top1, "violations": v, "near_ties": t}
                if args.against == "hf":  # the control: Transformers' cached generate judged by its own forward
                    g = model.generate(input_ids=pr[None], attention_mask=torch.ones_like(pr[None]), max_new_tokens=n,
                                       min_new_tokens=n, do_sample=False, pad_token_id=0)[0, C:]
                    lg = model(input_ids=torch.cat([pr, g[:-1]])[None], use_cache=False, logits_to_keep=n).logits[0].float()
                    r["hf_self_violations"], r["hf_self_near_ties"] = judge(lg, g)
                rows.append(r)
                print(f"C={C} {dom}{i}: KL {kl:.5f} (max {klmax:.4f})  top-1 {100 * top1:.1f}%  our greedy output v{v}/t{t}", flush=True)
                torch.cuda.empty_cache()
    res = {"model": cfg["model"]["id"], "args": vars(args), "rows": rows,
           "kl": sum(r["kl"] for r in rows) / len(rows), "top1": sum(r["top1"] for r in rows) / len(rows),
           "violations": sum(r["violations"] for r in rows),
           "control": sum(r.get("hf_self_violations", 0) for r in rows) if args.against == "hf" else None}
    print(f"LONG GATE C={C} against {args.against}: mean KL {res['kl']:.5f}  top-1 {100 * res['top1']:.2f}%  "
          f"violations {res['violations']}  control {res['control']}")
    path = Path(args.out) / f"{run_name(cfg)}_long_gate_{args.against}_{C}.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
