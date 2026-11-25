"""Control for the decoder gate (docs/decode_levers.md, follow-up E): Transformers against itself.
Per decode prompt, Transformers' cached greedy `generate` output (n tokens) is judged by Transformers'
own teacher-forced, uncached forward over prompt + those tokens, with the usual rule (a token that is not
the judge's argmax where the judge's top-2 gap is above 0.25 is a violation). Reported only.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decoder_gate import judge  # noqa: E402
from self_spec_bench import code_prompts  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402


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
    dev = model.device
    prompts = {"cal": [c.to(dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=400)],
               "hold": [c.to(dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=600)],
               "code": [c.to(dev) for c in code_prompts(tok, args.prompts, 512, args.code_rev)]}
    rows = []
    with torch.inference_mode():
        for dom, prs in prompts.items():
            for i, pr in enumerate(prs):
                P = pr.shape[0]
                g = model.generate(input_ids=pr[None], attention_mask=torch.ones_like(pr[None]), max_new_tokens=args.n,
                                   min_new_tokens=args.n, do_sample=False, pad_token_id=0)[0, P:]
                lh = model(input_ids=torch.cat([pr, g[:-1]])[None], use_cache=False).logits[0, P - 1:].float()
                v, t = judge(lh, g)
                rows.append({"domain": dom, "prompt": i, "violations": v, "near_ties": t})
                print(f"{dom} {i}: Transformers generate under its own forward: v{v}/t{t}", flush=True)
                torch.cuda.empty_cache()
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args), "rows": rows,
           "violations": sum(r["violations"] for r in rows), "near_ties": sum(r["near_ties"] for r in rows)}
    print(f"TOTAL violations {res['violations']}  near-ties {res['near_ties']}")
    path = Path(args.out) / f"{run_name(cfg)}_hf_self_consistency.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
