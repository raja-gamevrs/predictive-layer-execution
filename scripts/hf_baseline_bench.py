"""Reference measurement for docs/decode_levers.md ("Benchmark against Transformers"): decode speed of
Hugging Face Transformers' own `generate` (greedy, batch 1, BF16, SDPA attention) on the same prompts,
next to this repo's original and fast (L1 + L2) decoders, in one process on an idle GPU.

  transformers dynamic  generate() with the default dynamic KV cache
  transformers static   generate(cache_implementation="static"), no torch.compile (Triton is not
                        installed on this Windows build, so the compiled path is unavailable)

Decode ms/token = (wall time of generate for n tokens - wall time for 1 token) / (n - 1), so prefill is
excluded as in the lever runs. Each prompt: one warm-up, then `reps` timed runs, median. The decoders are
timed as in lever_bench.py (CUDA graph replay, CUDA events). Transformers' greedy tokens are checked
with greedy_check against the original decoder (violations: a token that is not the model's argmax where
the top-2 gap is above 0.25). Reported only; no bar.
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
from lever_bench import ev_ms, mode, step_runner  # noqa: E402
from self_spec_bench import code_prompts, greedy_check  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.gemma4_decode import make_decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402


def hf_time(model, pr, n, **kw):
    ids = pr[None]
    mask = torch.ones_like(ids)

    def run(m):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=m, min_new_tokens=m, do_sample=False,
                             pad_token_id=0, **kw)
        torch.cuda.synchronize()
        return time.perf_counter() - t, out[0, ids.shape[1]:].tolist()
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--code-rev", default="a118bda")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    model.eval()
    dec = make_decoder(model, 512 + args.n + 16)
    prompts = {"cal": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=400)],
               "hold": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=600)],
               "code": [c.to(dec.dev) for c in code_prompts(tok, args.prompts, 512, args.code_rev)]}
    hf_cfgs = {"transformers dynamic": {}, "transformers static": {"cache_implementation": "static", "disable_compile": True}}
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args),
           "torch": torch.__version__, "transformers": __import__("transformers").__version__, "rows": []}
    with torch.inference_mode():
        for dom, prs in prompts.items():
            for i, pr in enumerate(prs):
                r = {"domain": dom, "prompt": i}
                for name, kw in hf_cfgs.items():
                    try:
                        run = hf_time(model, pr, args.n, **kw)
                        run(args.n)
                        run(1)  # warm-up
                        t1 = [run(1)[0] for _ in range(args.reps)]
                        tn, toks = [], None
                        for _ in range(args.reps):
                            t, toks = run(args.n)
                            tn.append(t)
                        ms = 1000 * (statistics.median(tn) - statistics.median(t1)) / (args.n - 1)
                        mode(dec, "sdpa", "ref")
                        v, ties = greedy_check(dec, pr, toks)
                        r[name] = {"ms_per_token": ms, "violations": v, "near_ties": ties}
                    except Exception as e:  # noqa: BLE001
                        r[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
                for name, (attn, rope) in {"original decoder": ("sdpa", "ref"), "L1+L2": ("fast", "fused")}.items():
                    mode(dec, attn, rope)
                    _, rep, rst, _ = step_runner(dec, pr, False)
                    ts = []
                    for _ in range(args.reps + 1):
                        dec.prefill(pr)
                        rst()
                        ts.append(ev_ms(lambda: [rep() for _ in range(args.n - 1)])[0] / (args.n - 1))
                    r[name] = {"ms_per_token": statistics.median(ts[1:])}
                    del rep, rst
                    torch.cuda.empty_cache()
                mode(dec, "sdpa", "ref")
                res["rows"].append(r)
                print(dom, i, "  ".join(f"{k} {v['ms_per_token']:.2f}" if "ms_per_token" in v else f"{k} {v['error']}"
                                       for k, v in r.items() if isinstance(v, dict)), flush=True)
    names = [k for k in res["rows"][0] if isinstance(res["rows"][0][k], dict)]
    res["summary"] = {}
    for k in names:
        ms = [r[k]["ms_per_token"] for r in res["rows"] if "ms_per_token" in r[k]]
        if ms:
            m = statistics.mean(ms)
            res["summary"][k] = {"ms_per_token": m, "tokens_per_s": 1000 / m, "n": len(ms),
                                 "violations": sum(r[k].get("violations", 0) for r in res["rows"])}
            print(f"{k:22} {m:.3f} ms/token  {1000 / m:.0f} tokens/s  violations {res['summary'][k]['violations']}")
    path = Path(args.out) / f"{run_name(cfg)}_hf_baseline.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
