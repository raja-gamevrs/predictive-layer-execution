"""Reference measurement for docs/decode_levers.md ("Benchmark against llama.cpp"): llama.cpp's
llama-server on the same prompts as the self-spec runs (8 per domain, code prompts frozen at a118bda,
256 tokens), greedy (temperature 0), batch 1, every layer on the GPU, flash attention on.

  plain           the target GGUF alone (BF16, or a quantized GGUF for the lossy rows)
  draft k         the BF16 target with a quantized GGUF of the same model as llama.cpp's own draft model
                  (--spec-type draft-simple, --spec-draft-n-max k), llama.cpp's equivalent of L7

Prompts are sent as token ids from the Hugging Face tokenizer, so both sides see the same tokens.
Decode speed is llama-server's own timings (predicted_ms / predicted_n; prompt processing excluded).
Each prompt: one warm-up, then `reps` timed requests, median. Afterwards the server is stopped and
llama.cpp's greedy tokens are checked with greedy_check against this repo's original BF16 decoder
(violations: a token that is not the model's argmax where the top-2 gap is above 0.25).
Reported only; no bar.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lever_bench import mode  # noqa: E402
from self_spec_bench import code_prompts, greedy_check  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.model_loader import load_config, run_name  # noqa: E402


def post(port, path, body, timeout=600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(port, proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    raise TimeoutError("llama-server did not become ready")


def run_config(args, name, model, draft, k, prompts, log):
    cmd = [args.server, "-m", model, "-ngl", "99", "-fa", "on", "-c", str(max(2048, (args.context or 512) + args.n + 64)), "-np", "1",
           "--port", str(args.port), "--no-webui", "-ctk", args.kv, "-ctv", args.kv] + args.extra.split()
    if draft == "mtp":  # the model's own MTP / NextN layer from its GGUF (llama.cpp's draft-mtp)
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(k)]
    elif draft:
        cmd += ["-md", draft, "-ngld", "99", "--spec-type", "draft-simple", "--spec-draft-n-max", str(k),
                "-ctkd", args.kv, "-ctvd", args.kv]
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            wait_ready(args.port, proc)
            rows = []
            for dom, prs in prompts.items():
                for i, ids in enumerate(prs):
                    body = {"prompt": ids, "n_predict": args.n, "temperature": 0, "top_k": 1, "cache_prompt": False,
                            "ignore_eos": True, "return_tokens": True, "stream": False}
                    post(args.port, "/completion", body)  # warm-up
                    ms, acc, toks, pn = [], [], None, None
                    for _ in range(args.reps):
                        r = post(args.port, "/completion", body)
                        t = r["timings"]
                        assert t["predicted_n"] == args.n, t
                        pn = t.get("prompt_n")  # llama.cpp must not add a BOS of its own: checked in the summary
                        ms.append(t["predicted_ms"] / t["predicted_n"])
                        if "draft_n" in t and t["draft_n"]:
                            acc.append(t["draft_n_accepted"] / t["draft_n"])
                        toks = r["tokens"]
                    rows.append({"config": name, "k": k, "domain": dom, "prompt": i, "ms_per_token": statistics.median(ms),
                                 "accept_rate": statistics.mean(acc) if acc else None, "prompt_n": pn, "prompt_len": len(ids), "tokens": toks})
                    print(f"{name:14} {dom:4} {i}  {rows[-1]['ms_per_token']:.3f} ms/token"
                          + (f"  accept {rows[-1]['accept_rate']:.2f}" if acc else ""), flush=True)
            return rows
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--bf16", help="BF16 GGUF of the model (omit when it does not fit in GPU memory)")
    ap.add_argument("--quant", nargs="*", default=[], help="quantized GGUFs, run plain (lossy rows) and as drafts")
    ap.add_argument("--draft", help="quantized GGUF used as the draft (default: first of --quant)")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 6, 8])
    ap.add_argument("--no-spec", action="store_true", help="skip the speculative configs (e.g. target + draft do not fit in GPU memory)")
    ap.add_argument("--server", default="G:/llamacpp/bin/llama-server.exe")
    ap.add_argument("--port", type=int, default=8291)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--context", type=int, help="prompt length: given, the follow-up I / J prompt set (same as long_bench.py); "
                                                  "omitted, the 512-token prompts of the earlier comparisons")
    ap.add_argument("--kv", default="f16", help="llama.cpp KV-cache type for K and V (f16, q8_0, ...)")
    ap.add_argument("--domains", nargs="+", default=["cal", "hold", "code"])
    ap.add_argument("--target-label", default="bf16", help="name of the --bf16 target in the results (e.g. q8_0)")
    ap.add_argument("--mtp-ks", type=int, nargs="*", default=[], help="also llama.cpp's MTP speculation (draft-mtp) on the first --quant / --bf16 model")
    ap.add_argument("--no-check", action="store_true", help="skip the greedy check (follow-up J amended: tokens/s only)")
    ap.add_argument("--extra", default="", help="extra llama-server arguments, e.g. YaRN flags")
    ap.add_argument("--max-gb", type=float, default=31.0,
                    help="skip a config whose estimated GPU memory (GGUF sizes + KV caches) exceeds this: Windows pages "
                         "overcommitted GPU memory to system RAM instead of failing, which would time the paging")
    ap.add_argument("--kv-bytes-per-token", type=float, default=0.0, help="the target's KV bytes per token, for --max-gb")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--code-rev", default="a118bda")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from transformers import AutoTokenizer
    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["path"]) if m.get("path") else AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    if args.context is not None:
        from long_bench import long_prompts
        prompts = {d: [c.tolist() for c in v] for d, v in long_prompts(tok, args.context, args.prompts, "cpu").items()}
    else:
        prompts = {"cal": [c.tolist() for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=400)],
                   "hold": [c.tolist() for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=600)],
                   "code": [c.tolist() for c in code_prompts(tok, args.prompts, 512, args.code_rev)]}
    prompts = {d: v for d, v in prompts.items() if d in args.domains}
    out = Path(args.out)
    logdir = Path(args.server).parent.parent / "runs"
    draft = None if args.no_spec else (args.draft or (args.quant[0] if args.quant else None))
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args), "rows": []}
    tag = ("" if args.context is None else f"_ctx{args.context}") + ("" if args.kv == "f16" else f"_kv{args.kv}") +         ("" if args.target_label == "bf16" else f"_{args.target_label}")
    path = out / f"{run_name(cfg)}_llamacpp{tag}.json"
    configs = ([(args.target_label, args.bf16, None, 0)] if args.bf16 else []) + [(Path(q).stem.split("-")[-1], q, None, 0) for q in args.quant]
    configs += [(f"{args.target_label}+draft k={k}", args.bf16, draft, k) for k in args.ks] if draft and args.bf16 else []
    mtp_model = args.bf16 or (args.quant[0] if args.quant else None)
    configs += [(f"mtp k={k}", mtp_model, "mtp", k) for k in args.mtp_ks]
    for name, model, d, k in configs:
        log = logdir / f"server_{run_name(cfg)}_{name.replace(' ', '_').replace('=', '')}.log"
        ctx_tokens = (args.context or 512) + args.n + 64
        est = (Path(model).stat().st_size + (Path(d).stat().st_size if d and d != "mtp" else 0)) / 1e9 +             (1.1 if d == "mtp" else 2 if d else 1) * args.kv_bytes_per_token * ctx_tokens / 1e9   # MTP adds one layer's KV, a draft model a whole cache
        if args.kv_bytes_per_token and est > args.max_gb:
            res.setdefault("not_run", []).append({"config": name, "error": f"estimated {est:.1f} GB > {args.max_gb} GB"})
            print(f"{name}: not run (estimated {est:.1f} GB of GPU memory)", flush=True)
            continue
        try:
            res["rows"] += run_config(args, name, model, d, k, prompts, log)
        except (RuntimeError, TimeoutError) as e:  # e.g. the target and draft do not fit in GPU memory
            res.setdefault("not_run", []).append({"config": name, "error": str(e), "log": str(log)})
            print(f"{name}: not run ({e}); see {log}", flush=True)
        path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    if args.no_check:
        res["summary"] = []
        for name in dict.fromkeys(r["config"] for r in res["rows"]):
            g = [r for r in res["rows"] if r["config"] == name]
            ms = statistics.mean(r["ms_per_token"] for r in g)
            a = [r["accept_rate"] for r in g if r["accept_rate"] is not None]
            res["summary"].append({"config": name, "ms": ms, "tps": 1000 / ms, "accept": statistics.mean(a) if a else None})
            print(f"{name:16} {1000 / ms:.1f} tokens/s" + (f"  accept {statistics.mean(a):.2f}" if a else ""))
        path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("wrote", path)
        return
    # greedy check against this repo's original BF16 decoder
    from ple.gemma4_decode import make_decoder
    from ple.model_loader import load
    model, _ = load(cfg)
    dec = make_decoder(model, (args.context or 512) + args.n + 64)
    mode(dec, "sdpa", "ref")
    with torch.inference_mode():
        for r in res["rows"]:
            pr = torch.tensor(prompts[r["domain"]][r["prompt"]], device=dec.dev)
            if args.context is not None:  # chunked: the prompt's last-row logits, then the tokens as one extension
                from decoder_gate import judge
                toks = torch.tensor(r["tokens"], device=dec.dev)
                first = dec.prefill_chunked(pr)[-1:].float()
                ext = dec.prefill_chunked(toks[:-1], start=pr.shape[0], logits_rows=toks.shape[0] - 1).float()
                r["violations"], r["near_ties"] = judge(torch.cat([first, ext]), toks)
            else:
                r["violations"], r["near_ties"] = greedy_check(dec, pr, r["tokens"])
    res["summary"] = []
    for name in dict.fromkeys(r["config"] for r in res["rows"]):
        g = [r for r in res["rows"] if r["config"] == name]
        s = {"config": name, "violations": sum(r["violations"] for r in g),
             "prompt_len_mismatches": sum(1 for r in g if r.get("prompt_n") not in (None, r["prompt_len"]))}
        for dom in ("cal", "hold", "code"):
            m = statistics.mean(r["ms_per_token"] for r in g if r["domain"] == dom)
            s[f"ms_{dom}"], s[f"tps_{dom}"] = m, 1000 / m
            a = [r["accept_rate"] for r in g if r["domain"] == dom and r["accept_rate"] is not None]
            s[f"acc_{dom}"] = statistics.mean(a) if a else None
        res["summary"].append(s)
        print(f"{name:16} tokens/s cal {s['tps_cal']:.0f}  hold {s['tps_hold']:.0f}  code {s['tps_code']:.0f}  violations {s['violations']}")
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
