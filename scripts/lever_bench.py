"""Decode levers end to end on Qwen3-4B (docs/decode_levers.md, step 3).

Parts (select with --parts):
  decode   L1 fused RoPE, L2 fast attention, both, and L6 INT4 weights: correctness (teacher-forced,
           original decoder as judge) on 12 prompts, and timing interleaved with the original decoder
           (A B A B, 5 pairs per prompt, 128 decode steps each, prefill excluded).
  prefill  the same variants' 512-token prefill, interleaved with the original (eager, 5 pairs).
  quality  L6 INT4: KL / top-1 / perplexity change against BF16 on WikiText-2 train chunks 400-415 and
           600-615 (teacher-forced, 512 tokens).
  selfspec L4 (check acceptance every 1, 2 or 4 rounds) and L7 (INT4 self-draft, no skipping), on the
           fast decoder, k in {2, 3, 4, 6}, 8 prompts per domain, 256 tokens: correctness, acceptance and
           speedup over exact greedy decoding with the same fast decoder (and with the original).
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from self_spec_bench import code_prompts, greedy_check  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.gemma4_decode import make_decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402
from ple.self_spec import SelfSpecDecoder  # noqa: E402

VARIANTS = {"original": ("sdpa", "ref", False), "L1 fused rope": ("sdpa", "fused", False),
            "L2 fast attention": ("fast", "ref", False), "L1+L2": ("fast", "fused", False),
            "L1+L2+L6 int4": ("fast", "fused", True), "L1+L2d deterministic": ("fastdet", "fused", False)}


def mode(dec, attn, rope):
    """attn "fastdet": the fast path with the deterministic batched-matmul one-row step (L2d)."""
    dec.attn_impl, dec.rope_impl = ("fast" if attn == "fastdet" else attn), rope
    dec.step_attn = "bmm" if attn == "fastdet" else "cudnn"


def ev_ms(fn):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    r = fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b), r


def step_runner(dec, prompt, w4):
    first = int(dec.prefill(prompt, w4=w4)[-1].argmax())
    replay, reset, buf = dec.make_step(prompt.shape[0], first, w4=w4)
    return first, replay, reset, buf


def part_decode(dec, prompts, n_new, pairs, res):
    rows = []
    # INT4 variants only when the INT4 copies fit (dec.w4 is None after an out-of-memory skip)
    variants = {k: v for k, v in VARIANTS.items() if not v[2] or dec.w4 is not None}
    for dom, prs in prompts.items():
        for i, pr in enumerate(prs):
            mode(dec, "sdpa", "ref")
            f0, rep0, res0, _ = step_runner(dec, pr, False)
            ref_toks = None
            for name, (attn, rope, w4) in variants.items():
                mode(dec, attn, rope)
                first, rep, rst, buf = step_runner(dec, pr, w4)
                dec.prefill(pr, w4=w4)
                rst()
                toks = [first]
                for _ in range(n_new - 1):
                    rep()
                    toks.append(int(buf["tok"]))
                mode(dec, "sdpa", "ref")
                viol, ties = greedy_check(dec, pr, toks)
                if name == "original":
                    ref_toks = toks
                agree = next((j for j, (a, b) in enumerate(zip(toks, ref_toks)) if a != b), len(toks))
                te, tv = [], []
                if name != "original":
                    for _ in range(pairs + 1):
                        dec.prefill(pr)
                        res0()
                        e, _ = ev_ms(lambda: [rep0() for _ in range(n_new - 1)])
                        mode(dec, attn, rope)
                        dec.prefill(pr, w4=w4)
                        rst()
                        v, _ = ev_ms(lambda: [rep() for _ in range(n_new - 1)])
                        mode(dec, "sdpa", "ref")
                        te.append(e / (n_new - 1))
                        tv.append(v / (n_new - 1))
                    te, tv = te[1:], tv[1:]
                ratios = [a / b for a, b in zip(te, tv)]
                r = {"variant": name, "domain": dom, "prompt": i, "violations": viol, "near_ties": ties,
                     "same_as_original_until": agree,
                     "orig_ms_per_token": statistics.median(te) if te else None,
                     "ms_per_token": statistics.median(tv) if tv else None,
                     "speedup": statistics.median(ratios) if ratios else 1.0,
                     "speedup_spread": (max(ratios) - min(ratios)) / statistics.median(ratios) if ratios else 0.0}
                rows.append(r)
                del rep, rst, buf
            del rep0, res0
            torch.cuda.empty_cache()
            print(f"decode {dom}{i}: " + "  ".join(f"{r['variant']} {r['speedup']:.3f}x v{r['violations']}"
                                                  for r in rows[-len(variants):]), flush=True)
    res["decode"] = rows
    for name in variants:
        sub = [r for r in rows if r["variant"] == name]
        print(f"DECODE {name:16} speedup {statistics.mean(r['speedup'] for r in sub):.3f}x  "
              f"max spread {max(r['speedup_spread'] for r in sub):.3f}  violations {sum(r['violations'] for r in sub)}  "
              f"ms/token {statistics.mean(r['ms_per_token'] or 0 for r in sub):.3f}", flush=True)


def part_prefill(dec, prompts, pairs, res):
    rows = []
    for dom, prs in prompts.items():
        for i, pr in enumerate(prs):
            for name, (attn, rope, w4) in VARIANTS.items():
                if name == "original":
                    continue
                te, tv = [], []
                for _ in range(pairs + 1):
                    mode(dec, "sdpa", "ref")
                    e, ref = ev_ms(lambda: dec.prefill(pr))
                    mode(dec, attn, rope)
                    v, got = ev_ms(lambda: dec.prefill(pr, w4=w4))
                    te.append(e)
                    tv.append(v)
                lr, lg = torch.log_softmax(ref.float(), -1), torch.log_softmax(got.float(), -1)
                kl = (lr.exp() * (lr - lg)).sum(-1)[1:].mean().item()
                ratios = [a / b for a, b in zip(te[1:], tv[1:])]
                rows.append({"variant": name, "domain": dom, "prompt": i, "orig_ms": statistics.median(te[1:]),
                             "ms": statistics.median(tv[1:]), "speedup": statistics.median(ratios),
                             "speedup_spread": (max(ratios) - min(ratios)) / statistics.median(ratios),
                             "kl_vs_original": kl, "top1_vs_original": (ref.argmax(-1) == got.argmax(-1)).float().mean().item()})
    mode(dec, "sdpa", "ref")
    res["prefill"] = rows
    for name in VARIANTS:
        sub = [r for r in rows if r["variant"] == name]
        if sub:
            print(f"PREFILL {name:16} {statistics.mean(r['orig_ms'] for r in sub):.2f} -> {statistics.mean(r['ms'] for r in sub):.2f} ms  "
                  f"speedup {statistics.mean(r['speedup'] for r in sub):.3f}x  KL vs original {max(r['kl_vs_original'] for r in sub):.2e}", flush=True)


def part_quality(dec, tok, res):
    rows = []
    mode(dec, "fast", "fused")
    for split, skip in (("cal", 400), ("hold", 600)):
        kls, t1, nll_r, nll_g, n = [], [], 0.0, 0.0, 0
        for ch in wikitext_chunks(tok, "train", 512, 16, skip=skip):
            ch = ch.to(dec.dev)
            lr = torch.log_softmax(dec.prefill(ch).float(), -1)[1:-1]
            lg = torch.log_softmax(dec.prefill(ch, w4=True).float(), -1)[1:-1]
            tgt = ch[2:]
            kls.append((lr.exp() * (lr - lg)).sum(-1).mean().item())
            t1.append((lr.argmax(-1) == lg.argmax(-1)).float().mean().item())
            nll_r += -lr.gather(1, tgt[:, None]).sum().item()
            nll_g += -lg.gather(1, tgt[:, None]).sum().item()
            n += tgt.numel()
        ppl_r, ppl_g = math.exp(nll_r / n), math.exp(nll_g / n)
        rows.append({"split": split, "kl": statistics.mean(kls), "top1": statistics.mean(t1), "ppl_bf16": ppl_r,
                     "ppl_int4": ppl_g, "ppl_change_pct": 100 * (ppl_g / ppl_r - 1)})
        print(f"QUALITY int4 {split}: KL {rows[-1]['kl']:.4f}  top-1 {rows[-1]['top1']:.3f}  ppl {ppl_r:.3f} -> {ppl_g:.3f} "
              f"({rows[-1]['ppl_change_pct']:+.2f}%)", flush=True)
    mode(dec, "sdpa", "ref")
    res["quality_int4"] = rows


def part_selfspec(dec, prompts, n_new, pairs, res, ks, configs):
    rows = []
    for cname, cv in configs.items():
        # (w4, every) or (w4, every, skip_attn, skip_mlp): the draft may also skip sub-layers (follow-up F1)
        w4, every = cv[:2]
        skip_attn, skip_mlp = cv[2:4] if len(cv) > 2 else ((), ())
        for k in ks:
            mode(dec, "fast", "fused")
            ss = SelfSpecDecoder(dec, k, skip_attn, skip_mlp, max_new=n_new, draft_w4=w4)
            ss.capture(prompts["cal"][0])
            for dom, prs in prompts.items():
                for i, pr in enumerate(prs):
                    mode(dec, "fast", "fused")
                    out, rounds, acc = ss.generate(pr, n_new)
                    mode(dec, "sdpa", "ref")
                    viol, ties = greedy_check(dec, pr, out.tolist())
                    mode(dec, "fast", "fused")
                    first, rep, rst, _ = step_runner(dec, pr, False)
                    mode(dec, "sdpa", "ref")
                    f0, rep0, rst0, _ = step_runner(dec, pr, False)
                    mode(dec, "fast", "fused")

                    def exact(r, s):
                        def f():
                            dec.prefill(pr)
                            s()
                            ms, _ = ev_ms(lambda: [r() for _ in range(n_new - 1)])
                            return ms / (n_new - 1)
                        return f
                    tf, to, ts = [], [], []
                    for _ in range(pairs + 1):
                        tf.append(exact(rep, rst)())
                        mode(dec, "sdpa", "ref")
                        to.append(exact(rep0, rst0)())
                        mode(dec, "fast", "fused")
                        ss.start(pr)
                        torch.cuda.synchronize()
                        ms, ntok = ev_ms(lambda: spec_loop(ss, every, n_new))
                        ts.append(ms / ntok)
                    tf, to, ts = tf[1:], to[1:], ts[1:]
                    rows.append({"config": cname, "k": k, "domain": dom, "prompt": i, "rounds": rounds, "accepted": acc,
                                 "tokens_per_round": n_new / rounds, "accept_rate": acc / (rounds * k),
                                 "violations": viol, "near_ties": ties, "ms_per_token": statistics.median(ts),
                                 "exact_fast_ms_per_token": statistics.median(tf), "exact_orig_ms_per_token": statistics.median(to),
                                 "speedup_vs_fast": statistics.median(a / b for a, b in zip(tf, ts)),
                                 "speedup_vs_original": statistics.median(a / b for a, b in zip(to, ts))})
                    del rep, rst, rep0, rst0
            sub = [r for r in rows if r["config"] == cname and r["k"] == k]
            print(f"SELFSPEC {cname:18} k={k}  " + "  ".join(
                f"{d} {statistics.mean(r['speedup_vs_fast'] for r in sub if r['domain'] == d):.3f}x "
                f"(vs orig {statistics.mean(r['speedup_vs_original'] for r in sub if r['domain'] == d):.3f}x, "
                f"acc {100 * statistics.mean(r['accept_rate'] for r in sub if r['domain'] == d):.0f}%)" for d in prompts) +
                f"  violations {sum(r['violations'] for r in sub)}", flush=True)
            del ss
            torch.cuda.empty_cache()
    mode(dec, "sdpa", "ref")
    res["selfspec"] = rows


def spec_loop(ss, every, n_new):
    b = ss.buf
    while True:
        for _ in range(every):
            ss.graph.replay()
        if int(b["n_out"]) >= n_new:
            return int(b["n_out"]) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--parts", nargs="+", default=["decode", "prefill", "quality", "selfspec"])
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--ss-prompts", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--ss-pairs", type=int, default=2)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--ss-configs", nargs="+", help="self-spec configs to run (default: all four)")
    ap.add_argument("--tag", default="", help="suffix for the output file (e.g. a replication run)")
    ap.add_argument("--code-rev", help="git commit to read the code prompts from (default: working tree)")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = make_decoder(model, 512 + 256 + 8 * 7 + 16)
    try:
        dec.quantize_int4()
    except (torch.OutOfMemoryError, RuntimeError) as e:  # the driver can also report it as a RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        dec.w4, dec.w4_head = None, None
        torch.cuda.empty_cache()
        print("INT4 copies do not fit in GPU memory: INT4 variants and INT4 drafts are not run", flush=True)

    def doms(n):
        return {"cal": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, n, skip=400)],
                "hold": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, n, skip=600)],
                "code": [c.to(dec.dev) for c in code_prompts(tok, n, 512, args.code_rev)]}
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{run_name(cfg)}_levers_{'_'.join(args.parts)}{args.tag}.json"
    with torch.inference_mode():
        if "decode" in args.parts:
            part_decode(dec, doms(args.prompts), 128, args.pairs, res)
        if "prefill" in args.parts:
            part_prefill(dec, doms(args.prompts), args.pairs, res)
        if "quality" in args.parts:
            part_quality(dec, tok, res)
        if "selfspec" in args.parts:
            configs = {"L4 bf16 draft, every 1": (False, 1), "L4 bf16 draft, every 2": (False, 2),
                       "L7 int4 draft, every 1": (True, 1), "L7 int4 draft, every 2": (True, 2)}
            if args.ss_configs:
                configs = {k: configs[k] for k in args.ss_configs}
            if dec.w4 is None:
                res["selfspec_not_run"] = [k for k, (w4, _) in configs.items() if w4]
                configs = {k: v for k, v in configs.items() if not v[0]}
            part_selfspec(dec, doms(args.ss_prompts), 256, args.ss_pairs, res, args.ks, configs)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
