"""Follow-up L quality: a compressed KV cache (int8 / int4) against the BF16 cache, same model, long prompts.

Per context: one WikiText holdout prompt (as tps_bench) and its true 256-token continuation. For each mode the
prompt is prefilled into that mode's cache and the continuation is teacher-forced through it; next-token
distributions are compared with the BF16-cache run (mean KL, top-1 agreement, continuation perplexity). Then
256 greedy tokens are generated with the mode's one-token step and checked against the BF16-cache model's
teacher-forced pass (violations, decoder_gate.judge). Bar (pre-registered): mean KL <= 0.01 and top-1 >= 99%.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from decoder_gate import judge  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.gemma4_decode import make_decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402


def logits_for(dec, prompt, cont, chunk):
    first = dec.prefill_chunked(prompt, chunk=chunk)[-1:].float()
    ext = dec.prefill_chunked(cont[:-1], start=prompt.shape[0], logits_rows=cont.shape[0] - 1).float()
    return torch.cat([first, ext])                       # row i predicts cont[i]


def greedy(dec, prompt, n, chunk):
    first = int(dec.prefill_chunked(prompt, chunk=chunk)[-1].argmax())
    snap = dec.snapshot()
    rep, rst, buf = dec.make_step(prompt.shape[0], first)
    dec.restore(snap)
    rst()
    out = [first]
    for _ in range(n - 1):
        rep()
        out.append(int(buf["tok"].view(-1)[0]))
    del rep, rst, buf
    return torch.tensor(out, device=dec.dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen38_27b_int4.yaml")
    ap.add_argument("--contexts", type=int, nargs="+", default=[4096, 32768, 131072])
    ap.add_argument("--modes", nargs="+", default=["int8", "int4"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = make_decoder(model, 1024)
    dec.attn_impl, dec.rope_impl, dec.step_attn = "fast", "fused", "cudnn"
    dec.compile_step(True)
    res = {"model": cfg["model"]["id"], "args": vars(args), "rows": []}
    path = Path(args.out) / f"{run_name(cfg)}_kv_quality.json"
    with torch.inference_mode():
        for C in args.contexts:
            P = C - 320
            text = wikitext_chunks(tok, "train", P + args.n, 1, skip=math.ceil(1228800 / C))[0].to(dec.dev)
            prompt, cont = text[:P], text[P:P + args.n]
            ref = None
            for mode in ["bf16"] + args.modes:
                try:
                    dec.set_kv(mode)
                    dec.set_max_len(P + 2 * args.n + 64)
                    lg = logits_for(dec, prompt, cont, args.chunk)
                    lp = torch.log_softmax(lg, -1)
                    nll = float(-lp.gather(1, cont[:, None]).mean())
                    if mode == "bf16":
                        ref = lp
                        row = {"context": C, "mode": mode, "ppl": math.exp(nll)}
                    else:
                        kl = (ref.exp() * (ref - lp)).sum(-1)
                        top1 = float((ref.argmax(-1) == lp.argmax(-1)).float().mean())
                        g = greedy(dec, prompt, args.n, args.chunk)
                        dec.set_kv("bf16")
                        dec.set_max_len(P + 2 * args.n + 64)
                        tf = logits_for(dec, prompt, g, args.chunk)
                        v, nt = judge(tf, g)
                        row = {"context": C, "mode": mode, "ppl": math.exp(nll), "kl_mean": float(kl.mean()),
                               "kl_max": float(kl.max()), "top1": top1, "violations": v, "near_ties": nt,
                               "passes": float(kl.mean()) <= 0.01 and top1 >= 0.99}
                    res["rows"].append(row)
                    print({k: (round(x, 5) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
                except torch.OutOfMemoryError:
                    res["rows"].append({"context": C, "mode": mode, "error": "does not fit"})
                    print(f"C={C} {mode}: does not fit", flush=True)
                torch.cuda.empty_cache()
                path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
