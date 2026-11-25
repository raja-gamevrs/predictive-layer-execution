"""Follow-up H of docs/decode_levers.md: self-speculation with Qwen3.8's own MTP layer.

For each k: capture one MTP round (ple.mtp_spec), then per prompt (8 per domain, 256 tokens): the greedy
check of its output against the original decoder, and `pairs` interleaved timings of the exact
one-token decoders (original and fast = L1 + L2) and of MTP self-speculation (checked every round).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lever_bench import ev_ms, mode, step_runner  # noqa: E402
from self_spec_bench import code_prompts, greedy_check  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.gemma4_decode import make_decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402
from ple.mtp_spec import MTPLayer, MTPSelfSpec  # noqa: E402


def spin(ss, n_new):
    b = ss.buf
    while True:
        ss.graph.replay()
        if int(b["n_out"]) >= n_new:
            return int(b["n_out"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen38_27b_int4.yaml")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--fast", default="fast,fused", help="attn,rope of the fast exact decoder and of the spec rounds")
    ap.add_argument("--code-rev", default="a118bda")
    ap.add_argument("--fused-exact", action="store_true", help="the fast exact decoder uses the fused step (F3a), the best exact setting")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = make_decoder(model, 512 + args.n + 8 * 5 + 16)
    mtp = MTPLayer(dec, cfg["model"]["path"])
    zc = {k: float(v.float().mean()) for k, v in mtp.w.items() if k.endswith("norm.weight") or "norm_" in k}
    print("MTP norm weight means (zero-centred if near 0):", {k: round(v, 3) for k, v in zc.items()}, flush=True)
    fa, fr = args.fast.split(",")
    prompts = {"cal": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=400)],
               "hold": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=600)],
               "code": [c.to(dec.dev) for c in code_prompts(tok, args.prompts, 512, args.code_rev)]}
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args), "mtp_norm_means": zc, "rows": []}
    path = Path(args.out) / f"{run_name(cfg)}_mtp.json"
    with torch.inference_mode():
        for k in args.ks:
            mode(dec, fa, fr)
            ss = MTPSelfSpec(dec, mtp, k, max_new=args.n)
            ss.capture(prompts["cal"][0])
            for dom, prs in prompts.items():
                for i, pr in enumerate(prs):
                    mode(dec, fa, fr)
                    out, rounds, acc = ss.generate(pr, args.n)
                    mode(dec, "sdpa", "ref")
                    viol, ties = greedy_check(dec, pr, out.tolist())
                    mode(dec, fa, fr)
                    dec.compile_step(args.fused_exact)
                    _, repf, rstf, _ = step_runner(dec, pr, False)
                    dec.compile_step(False)
                    mode(dec, "sdpa", "ref")
                    _, repo, rsto, _ = step_runner(dec, pr, False)

                    def exact(rep, rst, m):
                        mode(dec, *m)
                        dec.prefill(pr)
                        rst()
                        ms, _ = ev_ms(lambda: [rep() for _ in range(args.n - 1)])
                        return ms / (args.n - 1)
                    tf, to, ts = [], [], []
                    for _ in range(args.pairs + 1):
                        tf.append(exact(repf, rstf, (fa, fr)))
                        to.append(exact(repo, rsto, ("sdpa", "ref")))
                        mode(dec, fa, fr)
                        ss.start(pr)
                        torch.cuda.synchronize()
                        ms, ntok = ev_ms(lambda: spin(ss, args.n))
                        ts.append(ms / ntok)
                    tf, to, ts = tf[1:], to[1:], ts[1:]
                    r = {"k": k, "domain": dom, "prompt": i, "rounds": rounds, "accepted": acc, "accept_rate": acc / (rounds * k),
                         "tokens_per_round": args.n / rounds, "violations": viol, "near_ties": ties,
                         "ms_per_token": statistics.median(ts), "exact_fast_ms_per_token": statistics.median(tf),
                         "exact_orig_ms_per_token": statistics.median(to),
                         "speedup_vs_fast": statistics.median(a / b for a, b in zip(tf, ts)),
                         "speedup_vs_original": statistics.median(a / b for a, b in zip(to, ts))}
                    res["rows"].append(r)
                    print(f"MTP k={k} {dom}{i}: {r['ms_per_token']:.2f} ms/token  {r['speedup_vs_fast']:.3f}x fast  "
                          f"{r['speedup_vs_original']:.3f}x orig  accept {100 * r['accept_rate']:.0f}%  v{viol}", flush=True)
                    del repf, rstf, repo, rsto
                    torch.cuda.empty_cache()
                    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            sub = [r for r in res["rows"] if r["k"] == k]
            print(f"MTP k={k} " + "  ".join(f"{d} {statistics.mean(r['speedup_vs_fast'] for r in sub if r['domain'] == d):.3f}x "
                                          f"(acc {100 * statistics.mean(r['accept_rate'] for r in sub if r['domain'] == d):.0f}%)"
                                          for d in prompts) + f"  violations {sum(r['violations'] for r in sub)}", flush=True)
            del ss
            torch.cuda.empty_cache()
    print("wrote", path)


if __name__ == "__main__":
    main()
