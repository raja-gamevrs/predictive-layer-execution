"""Follow-up A and B of docs/decode_levers.md: better INT4 for Qwen3-4B, as a model (L6b) and as a
self-speculative draft (L7b).

  --part quality  every INT4 variant: KL / top-1 / perplexity against BF16 on WikiText-2 train chunks
                  400-415 and 600-615 (as L6), greedy violations of INT4 decoding on 4 prompts per
                  domain (128 tokens, judged by the original BF16 decoder), and decode ms/token against
                  the original decoder (1 prompt per domain, 5 interleaved pairs). GPTQ statistics
                  come from train chunks 0-127.
  --part draft    the two variants with the lowest calibration KL as drafts for the L1 + L2 decoder,
                  k in {3, 4, 6}, acceptance checked every round (part_selfspec of lever_bench.py).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lever_bench import ev_ms, mode, part_quality, part_selfspec, step_runner  # noqa: E402
from self_spec_bench import code_prompts, greedy_check  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.decode import Qwen3Decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402

VARIANTS = {
    "rtn_g128": dict(group=128),
    "rtn_g64": dict(group=64),
    "rtn_g32": dict(group=32),
    "clip_g128": dict(group=128, clip=True),
    "clip_g32": dict(group=32, clip=True),
    "gptq_g128": dict(group=128, method="gptq"),
    "gptq_g32": dict(group=32, method="gptq"),
    "gptq_clip_g32": dict(group=32, method="gptq", clip=True),
}


class Hessians:
    """H = Σ xᵀx of each linear layer's input over calibration chunks, collected with forward hooks on
    the Transformers model `per` layers at a time (a block's four matrices take ~0.5 GB in FP32)."""

    def __init__(self, model, chunks, per=9):
        self.model, self.chunks, self.per = model, chunks, per
        self.cache, self.block = {}, None

    def _collect(self, layers, head):
        self.cache = {}
        hooks = []
        L = self.model.model.layers

        def hook(key):
            def f(_m, inp):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                h = self.cache.get(key)
                self.cache[key] = x.t() @ x if h is None else h.addmm_(x.t(), x)
            return f
        for i in layers:
            for key, mod in (("qkv", L[i].self_attn.q_proj), ("o", L[i].self_attn.o_proj),
                             ("gu", L[i].mlp.gate_proj), ("down", L[i].mlp.down_proj)):
                hooks.append(mod.register_forward_pre_hook(hook((i, key))))
        if head:
            hooks.append(self.model.lm_head.register_forward_pre_hook(hook("head")))
        try:
            for c in self.chunks:
                self.model(input_ids=c[None], use_cache=False)
        finally:
            for h in hooks:
                h.remove()

    def __call__(self, i):
        n = len(self.model.model.layers)
        if i == "head":
            if "head" not in self.cache:
                self._collect([], True)
                self.block = None
            return self.cache["head"]
        b = i // self.per
        if b != self.block:
            self.cache = {}
            torch.cuda.empty_cache()
            self._collect(range(b * self.per, min(n, (b + 1) * self.per)), False)
            self.block = b
        return {k: self.cache[(i, k)] for k in ("qkv", "o", "gu", "down")}


def decode_check(dec, prompts, n_new, pairs, time_on):
    """INT4 decoding (fast attention, fused RoPE): greedy violations per prompt against the original
    BF16 decoder as judge; ms/token against the original decoder on the prompts in time_on."""
    rows = []
    for dom, prs in prompts.items():
        for i, pr in enumerate(prs):
            mode(dec, "fast", "fused")
            first, rep, rst, buf = step_runner(dec, pr, True)
            dec.prefill(pr, w4=True)
            rst()
            toks = [first]
            for _ in range(n_new - 1):
                rep()
                toks.append(int(buf["tok"]))
            mode(dec, "sdpa", "ref")
            viol, ties = greedy_check(dec, pr, toks)
            r = {"domain": dom, "prompt": i, "violations": viol, "near_ties": ties}
            if (dom, i) in time_on:
                f0, rep0, rst0, _ = step_runner(dec, pr, False)
                te, tv = [], []
                for _ in range(pairs + 1):
                    mode(dec, "sdpa", "ref")
                    dec.prefill(pr)
                    rst0()
                    e, _ = ev_ms(lambda: [rep0() for _ in range(n_new - 1)])
                    mode(dec, "fast", "fused")
                    dec.prefill(pr, w4=True)
                    rst()
                    v, _ = ev_ms(lambda: [rep() for _ in range(n_new - 1)])
                    te.append(e / (n_new - 1))
                    tv.append(v / (n_new - 1))
                ratios = [a / b for a, b in zip(te[1:], tv[1:])]
                r.update({"orig_ms": statistics.median(te[1:]), "ms": statistics.median(tv[1:]),
                          "speedup": statistics.median(ratios), "spread": (max(ratios) - min(ratios)) / statistics.median(ratios)})
                del rep0, rst0
            rows.append(r)
            del rep, rst, buf
            torch.cuda.empty_cache()
    mode(dec, "sdpa", "ref")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--part", choices=["quality", "draft"], required=True)
    ap.add_argument("--variants", nargs="*", help="draft: variant names (default: two lowest calibration KL)")
    ap.add_argument("--code-rev", help="git commit to read the code prompts from (default: working tree)")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = Qwen3Decoder(model, 512 + 256 + 8 * 7 + 16)
    out = Path(args.out)
    qpath = out / f"{run_name(cfg)}_int4_quality.json"

    def doms(n):
        return {"cal": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, n, skip=400)],
                "hold": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, n, skip=600)],
                "code": [c.to(dec.dev) for c in code_prompts(tok, n, 512, args.code_rev)]}
    hess = Hessians(model, [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, 128, skip=0)])
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args)}
    with torch.inference_mode():
        if args.part == "quality":
            prompts = doms(4)
            variants = dict(VARIANTS)
            res["variants"] = []

            def run(name, kw):
                t0 = time.time()
                dec.quantize_int4(hessians=hess, **kw)
                tq = time.time() - t0
                r = {}
                part_quality(dec, tok, r)
                rows = decode_check(dec, prompts, 128, 5, {("cal", 0), ("hold", 0), ("code", 0)})
                t = [x for x in rows if "speedup" in x]
                v = {"variant": name, **{k: v for k, v in kw.items()}, "quantize_s": tq, "quality": r["quality_int4"],
                     "decode": rows, "violations": sum(x["violations"] for x in rows),
                     "near_ties": sum(x["near_ties"] for x in rows), "ms_per_token": statistics.mean(x["ms"] for x in t),
                     "orig_ms_per_token": statistics.mean(x["orig_ms"] for x in t),
                     "speedup": statistics.mean(x["speedup"] for x in t), "spread": max(x["spread"] for x in t)}
                res["variants"].append(v)
                q = {x["split"]: x for x in v["quality"]}
                print(f"INT4 {name:22} cal KL {q['cal']['kl']:.4f} ppl {q['cal']['ppl_change_pct']:+.2f}%  hold KL {q['hold']['kl']:.4f} "
                      f"ppl {q['hold']['ppl_change_pct']:+.2f}%  top-1 {q['hold']['top1']:.3f}  violations {v['violations']}  "
                      f"{v['ms_per_token']:.3f} ms/token ({v['speedup']:.3f}x orig)  quantize {tq:.0f}s", flush=True)
                qpath.write_text(json.dumps(res, indent=2), encoding="utf-8")
            for name, kw in variants.items():
                run(name, kw)
            best = min(res["variants"], key=lambda v: next(x["kl"] for x in v["quality"] if x["split"] == "cal"))
            kw = {k: best[k] for k in ("group", "method", "clip") if k in best}
            run(best["variant"] + "_headbf16", {**kw, "head": False})
            print("wrote", qpath)
        else:
            # variants from this model's quality sweep if there is one, else from VARIANTS (--variants)
            q = json.loads(qpath.read_text(encoding="utf-8")) if qpath.exists() else {"variants": []}
            ranked = sorted(q["variants"], key=lambda v: next(x["kl"] for x in v["quality"] if x["split"] == "cal"))
            names = args.variants or [v["variant"] for v in ranked[:2]]
            byname = {**{k: dict(kw) for k, kw in VARIANTS.items()}, **{v["variant"]: v for v in q["variants"]}}
            for name in names:
                v = byname[name]
                kw = {k: v[k] for k in ("group", "method", "clip", "head") if k in v}
                dec.quantize_int4(hessians=hess, **kw)
                r = {}
                part_selfspec(dec, doms(8), 256, 2, r, [3, 4, 6], {f"L7b {name}, every 1": (True, 1)})
                res.setdefault("selfspec", []).extend(r["selfspec"])
                path = out / f"{run_name(cfg)}_int4_draft.json"
                path.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print("wrote", path)


if __name__ == "__main__":
    main()
