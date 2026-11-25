"""Correctness gate for a new decoder architecture (docs/decode_levers.md, follow-up E): this repo's
original decoder (attn "sdpa", rope "ref") against Transformers on the 12 decode prompts.

Per prompt: Transformers greedy-generates n tokens (generate, dynamic cache); both models then score
prompt + those tokens in one teacher-forced pass. Reported: mean KL(Transformers || ours) and top-1
agreement over the n scored positions; violations of each side's greedy tokens judged by the other
side's logits (a token that is not the judge's argmax where the judge's top-2 gap is above 0.25);
and whether our own CUDA-graph greedy output equals Transformers'. Gate (fixed before any run):
mean KL <= 0.005, top-1 >= 99%, 0 violations of our greedy output under Transformers' logits.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lever_bench import mode, step_runner  # noqa: E402
from self_spec_bench import NEAR_TIE, code_prompts  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.gemma4_decode import make_decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402


def judge(logits, toks):
    """Violations / near-ties of toks (n,) under judge logits (n, vocab)."""
    top = logits.topk(2, dim=-1)
    gap = top.values[:, 0] - top.values[:, 1]
    wrong = top.indices[:, 0] != toks
    return int((wrong & (gap > NEAR_TIE)).sum()), int((wrong & (gap <= NEAR_TIE)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gemma4_12b.yaml")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--code-rev", default="a118bda")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = make_decoder(model, 512 + args.n + 16)
    prompts = {"cal": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=400)],
               "hold": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=600)],
               "code": [c.to(dec.dev) for c in code_prompts(tok, args.prompts, 512, args.code_rev)]}
    rows = []
    with torch.inference_mode():
        for dom, prs in prompts.items():
            for i, pr in enumerate(prs):
                P = pr.shape[0]
                out = model.generate(input_ids=pr[None], attention_mask=torch.ones_like(pr[None]), max_new_tokens=args.n,
                                     min_new_tokens=args.n, do_sample=False, pad_token_id=0)
                hf_toks = out[0, P:]
                seq = torch.cat([pr, hf_toks[:-1]])
                lh = model(input_ids=seq[None], use_cache=False).logits[0, P - 1:].float()
                mode(dec, "sdpa", "ref")
                lo = dec.prefill(seq)[P - 1:].float()
                kl = torch.nn.functional.kl_div(torch.log_softmax(lo, -1), torch.log_softmax(lh, -1), log_target=True,
                                                reduction="none").sum(-1)
                top1 = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
                # our own CUDA-graph greedy output, judged by Transformers' teacher-forced logits
                first, rep, rst, buf = step_runner(dec, pr, False)
                dec.prefill(pr)
                rst()
                ours = [first]
                for _ in range(args.n - 1):
                    rep()
                    ours.append(int(buf["tok"]))
                ours_t = torch.tensor(ours, device=dec.dev)
                lh2 = model(input_ids=torch.cat([pr, ours_t[:-1]])[None], use_cache=False).logits[0, P - 1:].float()
                v_ours, t_ours = judge(lh2, ours_t)
                v_hf, t_hf = judge(lo, hf_toks)
                same = next((j for j, (a, b) in enumerate(zip(ours, hf_toks.tolist())) if a != b), args.n)
                r = {"domain": dom, "prompt": i, "kl": kl.mean().item(), "kl_max": kl.max().item(), "top1": top1,
                     "ours_violations": v_ours, "ours_near_ties": t_ours, "hf_violations": v_hf, "hf_near_ties": t_hf,
                     "same_as_transformers_until": same}
                rows.append(r)
                print(f"{dom} {i}: KL {r['kl']:.5f} (max {r['kl_max']:.4f})  top-1 {100 * top1:.1f}%  ours v{v_ours}/t{t_ours}  "
                      f"transformers v{v_hf}/t{t_hf}  identical for {same} tokens", flush=True)
                del rep, rst, buf, lh, lo, lh2
                torch.cuda.empty_cache()
    kl = sum(r["kl"] for r in rows) / len(rows)
    top1 = sum(r["top1"] for r in rows) / len(rows)
    viol = sum(r["ours_violations"] for r in rows)
    passed = kl <= 0.005 and top1 >= 0.99 and viol == 0
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args), "rows": rows,
           "summary": {"kl": kl, "top1": top1, "ours_violations": viol, "hf_violations": sum(r["hf_violations"] for r in rows),
                       "passed": passed}}
    print(f"GATE mean KL {kl:.5f}  top-1 {100 * top1:.2f}%  our violations {viol}  -> {'PASS' if passed else 'FAIL'}")
    path = Path(args.out) / f"{run_name(cfg)}_decoder_gate.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
