"""Follow-up I of docs/decode_levers.md: the decode levers at long contexts.

Per model and context (prompt length C): the cache is sized C + 256 + 64; per prompt (2 per domain)
the prompt is prefilled once (chunked, reference mode), its mutable decode state snapshotted, and each
variant's CUDA-graph step generates n tokens from that state; the greedy check scores them as one
teacher-forced extension of the prompt's cache; `pairs` interleaved timings against the original
decoder restore the snapshot instead of re-prefilling. Optionally (--selfspec) the INT4 self-draft
(L7b: GPTQ + clipping, groups of 32, k) at the same contexts.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decoder_gate import judge  # noqa: E402
from lever_bench import VARIANTS, ev_ms, mode, spec_loop  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.gemma4_decode import make_decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402

STDLIB_EXCLUDE = ("test", "idlelib", "site-packages")


def stdlib_prompts(tok, n, length):
    """n consecutive prompts of `length` tokens from the sorted CPython standard library (amendment of I)."""
    root = Path(sys.base_prefix) / "Lib"
    files = sorted(p for p in root.rglob("*.py")
                   if not any(part.startswith(STDLIB_EXCLUDE) for part in p.relative_to(root).parts))
    ids, need = [], n * length
    for f in files:
        ids += tok(f.read_text(encoding="utf-8", errors="replace") + "\n\n", add_special_tokens=False)["input_ids"]
        if len(ids) >= need:
            break
    return [torch.tensor(ids[i * length:(i + 1) * length]) for i in range(n)]


def long_prompts(tok, C, n, dev):
    return {"cal": [c.to(dev) for c in wikitext_chunks(tok, "train", C, n, skip=math.ceil(204800 / C))],
            "hold": [c.to(dev) for c in wikitext_chunks(tok, "train", C, n, skip=math.ceil(1228800 / C))],
            "code": [c.to(dev) for c in stdlib_prompts(tok, n, C)]}


def gen(rep, rst, buf, first, n):
    rst()
    toks = [first]
    for _ in range(n - 1):
        rep()
        toks.append(int(buf["tok"]))
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384, 32256])
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--pairs", type=int, default=2)  # amendment 2026-11-24 (user): 2 timed pairs
    ap.add_argument("--variants", nargs="+", default=["L1 fused rope", "L2 fast attention", "L1+L2", "L1+L2d deterministic"])
    ap.add_argument("--selfspec", action="store_true", help="also the L7b INT4 self-draft (Qwen3 / Llama decoders)")
    ap.add_argument("--draft-codes", help="load the self-draft's saved INT4 codes (follow-up J) instead of quantising")
    ap.add_argument("--no-orig", action="store_true", help="follow-up J: time only this repo's settings, not the original decoder")
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ss-n", type=int, default=256)
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = make_decoder(model, args.contexts[0] + args.ss_n + 64)
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args), "rows": [], "selfspec": []}
    path = Path(args.out) / f"{run_name(cfg)}_long{args.tag}.json"
    ss_cls = None
    if args.selfspec:
        from lever_int4 import Hessians
        from ple.self_spec import SelfSpecDecoder as ss_cls
        if args.draft_codes:
            with torch.inference_mode():
                dec.load_int4_codes(args.draft_codes)
        else:
            hess = Hessians(model, [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, 128, skip=0)])
            with torch.inference_mode():
                dec.quantize_int4(group=32, method="gptq", clip=True, hessians=hess)
            del hess
        torch.cuda.empty_cache()
    with torch.inference_mode():
        for C in args.contexts:
            dec.set_max_len(C + args.ss_n + 64)
            prompts = long_prompts(tok, C, args.prompts, dec.dev)
            for dom, prs in prompts.items():
                for i, pr in enumerate(prs):
                    mode(dec, "sdpa", "ref")
                    first_logits = dec.prefill_chunked(pr)[-1:].float()
                    first = int(first_logits.argmax())
                    snap = dec.snapshot()
                    rep0, rst0, buf0 = dec.make_step(C, first)
                    dec.restore(snap)
                    ref_toks = gen(rep0, rst0, buf0, first, args.n)

                    def check(toks):
                        mode(dec, "sdpa", "ref")
                        dec.restore(snap)
                        ext = dec.prefill_chunked(torch.tensor(toks[:-1], device=dec.dev), start=C, logits_rows=len(toks) - 1).float()
                        dec.restore(snap)
                        return judge(torch.cat([first_logits, ext]), torch.tensor(toks, device=dec.dev))
                    v0, t0 = check(ref_toks)
                    res["rows"].append({"context": C, "variant": "original", "domain": dom, "prompt": i, "violations": v0, "near_ties": t0})
                    line = [f"original v{v0}"]
                    for name in args.variants:
                        attn, rope, _ = VARIANTS[name]
                        mode(dec, attn, rope)
                        rep, rst, buf = dec.make_step(C, first)
                        dec.restore(snap)
                        toks = gen(rep, rst, buf, first, args.n)
                        v, t = check(toks)
                        agree = next((j for j, (a, b) in enumerate(zip(toks, ref_toks)) if a != b), len(toks))
                        te, tv = [], []
                        for _ in range(args.pairs + 1):
                            e = float("nan")
                            if not args.no_orig:
                                mode(dec, "sdpa", "ref")
                                dec.restore(snap)
                                rst0()
                                e, _ = ev_ms(lambda: [rep0() for _ in range(args.n - 1)])
                            mode(dec, attn, rope)
                            dec.restore(snap)
                            rst()
                            w, _ = ev_ms(lambda: [rep() for _ in range(args.n - 1)])
                            te.append(e / (args.n - 1))
                            tv.append(w / (args.n - 1))
                        te, tv = te[1:], tv[1:]
                        ratios = [a / b for a, b in zip(te, tv)]
                        r = {"context": C, "variant": name, "domain": dom, "prompt": i, "violations": v, "near_ties": t,
                             "same_as_original_until": agree, "orig_ms_per_token": statistics.median(te),
                             "ms_per_token": statistics.median(tv), "speedup": statistics.median(ratios),
                             "speedup_spread": (max(ratios) - min(ratios)) / statistics.median(ratios)}
                        res["rows"].append(r)
                        line.append(f"{name} {r['speedup']:.3f}x ({r['ms_per_token']:.2f} ms) v{v}")
                        del rep, rst, buf
                        torch.cuda.empty_cache()
                    print(f"C={C} {dom}{i}: orig {statistics.median(te):.2f} ms  " + "  ".join(line), flush=True)
                    if ss_cls is not None:
                        mode(dec, "fast", "fused")
                        repf, rstf, _ = dec.make_step(C, first)
                        ss = ss_cls(dec, args.k, (), (), max_new=args.ss_n, draft_w4=True)
                        ss.capture(pr)
                        out, rounds, acc = ss.generate(pr, args.ss_n)
                        mode(dec, "sdpa", "ref")
                        dec.prefill_chunked(pr)
                        snap2 = dec.snapshot()
                        v, t = check(out.tolist())
                        tf, to, ts = [], [], []
                        for _ in range(args.pairs + 1):
                            for (rp, rs, m, lst) in ((repf, rstf, ("fast", "fused"), tf), (rep0, rst0, ("sdpa", "ref"), to))[:1 if args.no_orig else 2]:
                                mode(dec, *m)
                                dec.restore(snap2)
                                rs()
                                ms, _ = ev_ms(lambda: [rp() for _ in range(args.ss_n - 1)])
                                lst.append(ms / (args.ss_n - 1))
                            mode(dec, "fast", "fused")
                            ss.start(pr)
                            torch.cuda.synchronize()
                            ms, ntok = ev_ms(lambda: spec_loop(ss, 1, args.ss_n))
                            ts.append(ms / ntok)
                        tf, to, ts = tf[1:], to[1:], ts[1:]
                        s = {"context": C, "domain": dom, "prompt": i, "k": args.k, "accept_rate": acc / (rounds * args.k),
                             "violations": v, "near_ties": t, "ms_per_token": statistics.median(ts),
                             "exact_fast_ms_per_token": statistics.median(tf), "exact_orig_ms_per_token": statistics.median(to) if to else None,
                             "speedup_vs_fast": statistics.median(a / b for a, b in zip(tf, ts)),
                             "speedup_vs_original": statistics.median(a / b for a, b in zip(to, ts)) if to else None}
                        res["selfspec"].append(s)
                        print(f"C={C} {dom}{i}: L7b k={args.k} {s['ms_per_token']:.2f} ms  {s['speedup_vs_fast']:.3f}x fast  "
                              f"{(s['speedup_vs_original'] or float('nan')):.3f}x orig  accept {100 * s['accept_rate']:.0f}%  v{v}", flush=True)
                        del ss, repf, rstf
                    del rep0, rst0, buf0
                    torch.cuda.empty_cache()
                    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            for name in ["original"] + args.variants:
                sub = [r for r in res["rows"] if r["context"] == C and r["variant"] == name]
                if name == "original":
                    print(f"LONG C={C} original violations {sum(r['violations'] for r in sub)}", flush=True)
                    continue
                print(f"LONG C={C} {name:22} {statistics.mean(r['speedup'] for r in sub):.3f}x  "
                      f"{statistics.mean(r['ms_per_token'] for r in sub):.2f} ms/token (orig {statistics.mean(r['orig_ms_per_token'] for r in sub):.2f})  "
                      f"spread {max(r['speedup_spread'] for r in sub):.3f}  violations {sum(r['violations'] for r in sub)}", flush=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
