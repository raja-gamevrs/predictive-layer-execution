"""The decoder gate of scripts/decoder_gate.py (and the control of scripts/hf_self_consistency.py) for
models where Transformers' model and this repo's decoder do not fit in GPU memory together
(docs/decode_levers.md, follow-up G). Three passes, one model in memory at a time:

  --phase ref    Transformers: greedy-generate n tokens per prompt; save its teacher-forced logits over
                 the n scored positions; judge its own generate output by them (the control)
  --phase ours   this repo's original decoder: teacher-forced logits over the same sequences (KL and
                 top-1 against the saved ones); its own CUDA-graph greedy output
  --phase judge  Transformers: teacher-forced logits over prompt + our greedy output; violations
Gate (amended 2026-11-24 for follow-up E, reused unchanged): mean KL <= 0.005, top-1 >= 99%, our
violations <= Transformers' own (the control).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decoder_gate import judge  # noqa: E402
from lever_bench import mode, step_runner  # noqa: E402
from self_spec_bench import code_prompts  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen38_27b_int4.yaml")
    ap.add_argument("--phase", choices=["ref", "ours", "judge"], required=True)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--code-rev", default="a118bda")
    ap.add_argument("--work", default="G:/ple_cache/gate", help="where the per-prompt logits are kept between phases")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dev = model.device
    work = Path(args.work) / run_name(cfg)
    work.mkdir(parents=True, exist_ok=True)
    prompts = {"cal": [c.to(dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=400)],
               "hold": [c.to(dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=600)],
               "code": [c.to(dev) for c in code_prompts(tok, args.prompts, 512, args.code_rev)]}
    path = Path(args.out) / f"{run_name(cfg)}_decoder_gate.json"
    res = json.loads(path.read_text(encoding="utf-8")) if args.phase != "ref" and path.exists() else {
        "model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args), "rows": {}}
    if args.phase == "ours":
        from ple.gemma4_decode import make_decoder
        dec = make_decoder(model, 512 + args.n + 16)
    with torch.inference_mode():
        for dom, prs in prompts.items():
            for i, pr in enumerate(prs):
                key, P = f"{dom}{i}", pr.shape[0]
                r = res["rows"].setdefault(key, {"domain": dom, "prompt": i})
                f = work / f"{key}.pt"
                if args.phase == "ref":
                    g = model.generate(input_ids=pr[None], attention_mask=torch.ones_like(pr[None]), max_new_tokens=args.n,
                                       min_new_tokens=args.n, do_sample=False, pad_token_id=0)[0, P:]
                    lh = model(input_ids=torch.cat([pr, g[:-1]])[None], use_cache=False).logits[0, P - 1:]
                    r["hf_self_violations"], r["hf_self_near_ties"] = judge(lh.float(), g)
                    torch.save({"tokens": g.cpu(), "logits": lh.to(torch.bfloat16).cpu()}, f)
                    print(f"{key}: Transformers generate under its own forward v{r['hf_self_violations']}/t{r['hf_self_near_ties']}", flush=True)
                elif args.phase == "ours":
                    saved = torch.load(f)
                    g, lh = saved["tokens"].to(dev), saved["logits"].to(dev).float()
                    mode(dec, "sdpa", "ref")
                    lo = dec.prefill(torch.cat([pr, g[:-1]]))[P - 1:].float()
                    kl = torch.nn.functional.kl_div(torch.log_softmax(lo, -1), torch.log_softmax(lh, -1), log_target=True,
                                                    reduction="none").sum(-1)
                    r["kl"], r["kl_max"] = kl.mean().item(), kl.max().item()
                    r["top1"] = (lo.argmax(-1) == lh.argmax(-1)).float().mean().item()
                    r["hf_violations"], r["hf_near_ties"] = judge(lo, g)
                    first, rep, rst, buf = step_runner(dec, pr, False)
                    dec.prefill(pr)
                    rst()
                    ours = [first]
                    for _ in range(args.n - 1):
                        rep()
                        ours.append(int(buf["tok"]))
                    r["ours_tokens"] = ours
                    r["same_as_transformers_until"] = next((j for j, (a, b) in enumerate(zip(ours, g.tolist())) if a != b), args.n)
                    print(f"{key}: KL {r['kl']:.5f} (max {r['kl_max']:.4f})  top-1 {100 * r['top1']:.1f}%  "
                          f"identical for {r['same_as_transformers_until']} tokens", flush=True)
                    del rep, rst, buf
                else:
                    ours = torch.tensor(r["ours_tokens"], device=dev)
                    lh2 = model(input_ids=torch.cat([pr, ours[:-1]])[None], use_cache=False).logits[0, P - 1:].float()
                    r["ours_violations"], r["ours_near_ties"] = judge(lh2, ours)
                    print(f"{key}: our greedy output under Transformers' forward v{r['ours_violations']}/t{r['ours_near_ties']}", flush=True)
                torch.cuda.empty_cache()
                path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    rows = list(res["rows"].values())
    if all("ours_violations" in r for r in rows):
        kl = sum(r["kl"] for r in rows) / len(rows)
        top1 = sum(r["top1"] for r in rows) / len(rows)
        viol, control = sum(r["ours_violations"] for r in rows), sum(r["hf_self_violations"] for r in rows)
        res["summary"] = {"kl": kl, "top1": top1, "ours_violations": viol, "hf_self_violations": control,
                          "passed": kl <= 0.005 and top1 >= 0.99 and viol <= control}
        print(f"GATE mean KL {kl:.5f}  top-1 {100 * top1:.2f}%  our violations {viol} (control {control})  -> "
              f"{'PASS' if res['summary']['passed'] else 'FAIL'}")
        path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
