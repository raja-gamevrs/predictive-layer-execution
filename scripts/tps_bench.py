"""Follow-up J (amended): stable decode tokens/s of this repo's best settings at long contexts.

Per context (prompt = nominal context - 320 tokens of WikiText-2 holdout text, as follow-up I): the cache
is sized for the prompt + 256 + 64, the prompt prefilled once (chunked), then each setting generates 256
tokens from that state: one warm-up and `reps` timed runs, median ms/token, prefill excluded.
  l12   L1 + L2 (fast attention, fused RoPE), the one-token CUDA-graph step
  l7b   INT4 self-draft (loaded codes, --draft-codes), k = 6, acceptance checked every round
Qwen3-4B past its 32,768 positions uses YaRN (factor 4, original 32,768). A setting that runs out of GPU
memory is recorded as not fitting.
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
from lever_bench import ev_ms, mode, spec_loop  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.gemma4_decode import make_decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402


def oom(e):
    return isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--contexts", type=int, nargs="+", default=[2048, 4096, 8192, 16384, 32768, 65536, 131072])
    ap.add_argument("--settings", nargs="+", default=["l12", "l7b"], help="l12, l12c (compiled step, F3a), l7b")
    ap.add_argument("--draft-codes", default="G:/ple_cache/draft_qwen3_4b_gptq_clip_g32.pt")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--yarn-native", type=int, default=0, help="native positions; above it use YaRN factor 4 (Qwen3-4B: 32768)")
    ap.add_argument("--target", choices=["native", "q8"], default="native", help="q8: the Q8_0 target of follow-up K")
    ap.add_argument("--check", type=int, default=0, help="follow-up K: greedy check on this many prompts at the first context")
    ap.add_argument("--save-draft-codes", help="Gemma + q8: where to save the INT4 draft's codes (for llama.cpp's Q4_1)")
    ap.add_argument("--tag")
    ap.add_argument("--mtp-ks", type=int, nargs="*", default=[], help="Qwen3.5/3.8: MTP self-spec with these k (follow-up H)")
    ap.add_argument("--mtp-head", choices=["bf16", "int4"], default="bf16", help="MTP draft's LM head (follow-up H2: int4 copy)")
    ap.add_argument("--prefill-chunk", type=int, default=2048, help="rows per prefill chunk (smaller: less attention memory)")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    slack = 64 + 8 * (args.k + 1)
    dec = make_decoder(model, 1024)
    has_draft = bool({"l7b", "l7bc"} & set(args.settings)) and hasattr(dec, "load_int4_codes")
    if has_draft:
        with torch.inference_mode():
            dec.load_int4_codes(args.draft_codes)
    if args.target == "q8":
        with torch.inference_mode():
            if hasattr(dec, "load_int4_codes"):
                dec.quantize_q8_target()
            else:  # Gemma: Q8_0 target and its INT4 draft built together, layer by layer (memory)
                dec.quantize_q8_target(model, codes_path=args.save_draft_codes, draft=bool({"l7b", "l7bc"} & set(args.settings)))
                has_draft = bool({"l7b", "l7bc"} & set(args.settings))
    from ple.self_spec import SelfSpecDecoder
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args), "rows": [], "check": []}
    ap_tag = "" if args.tag is None else args.tag
    path = Path(args.out) / f"{run_name(cfg)}_tps{'' if args.target == 'native' else '_' + args.target}{ap_tag}.json"
    with torch.inference_mode():
        for Cn in args.contexts:
            C = Cn - 320
            need = C + args.n + slack
            yarn = (4.0, args.yarn_native) if args.yarn_native and need > args.yarn_native else None
            row = {"context": Cn, "prompt_len": C, "yarn": yarn}
            try:
                dec.set_max_len(need, yarn=yarn) if yarn is not None else dec.set_max_len(need)
                pr = wikitext_chunks(tok, "train", C, 1, skip=math.ceil(1228800 / C))[0].to(dec.dev)
                mode(dec, "fast", "fused")
                first = int(dec.prefill_chunked(pr, chunk=args.prefill_chunk)[-1].argmax())
                snap = dec.snapshot()
            except Exception as e:  # noqa: BLE001
                if not oom(e):
                    raise
                row["error"] = "does not fit (prefill / cache)"
                res["rows"].append(row)
                print(f"C={Cn}: does not fit", flush=True)
                torch.cuda.empty_cache()
                continue
            if args.check and Cn == args.contexts[0]:
                from decoder_gate import judge
                for j, pc in enumerate(wikitext_chunks(tok, "train", C, args.check, skip=math.ceil(1228800 / C))):
                    pc = pc.to(dec.dev)
                    mode(dec, "fast", "fused")
                    fl = dec.prefill_chunked(pc, chunk=args.prefill_chunk)[-1:].float()
                    f0 = int(fl.argmax())
                    sp = dec.snapshot()
                    rep, rst, buf = dec.make_step(C, f0)
                    dec.restore(sp)
                    rst()
                    toks = [f0]
                    for _ in range(127):
                        rep()
                        toks.append(int(buf["tok"]))
                    outs = {"l12": toks}
                    if "l12c" in args.settings:
                        dec.compile_step(True)
                        dec.restore(sp)
                        repc, rstc, bufc = dec.make_step(C, f0)
                        dec.restore(sp)
                        rstc()
                        tc = [f0]
                        for _ in range(127):
                            repc()
                            tc.append(int(bufc["tok"]))
                        outs["l12c"] = tc
                        dec.compile_step(False)
                        del repc, rstc, bufc
                    if has_draft:
                        dec.restore(sp)
                        ss = SelfSpecDecoder(dec, args.k, (), (), max_new=128, draft_w4=True)
                        ss.capture(pc, f0)
                        ss.start(pc, f0)
                        spec_loop(ss, 1, 128)
                        outs["l7b"] = ss.buf["out"][:128].tolist()
                        del ss
                    for name, tk in outs.items():
                        dec.restore(sp)
                        t = torch.tensor(tk, device=dec.dev)
                        ext = dec.prefill_chunked(t[:-1], start=C, logits_rows=127).float()
                        v, nt = judge(torch.cat([fl, ext]), t)
                        res["check"].append({"prompt": j, "setting": name, "violations": v, "near_ties": nt})
                        print(f"check prompt {j} {name}: violations {v}, near-ties {nt} (128 tokens)", flush=True)
                    del rep, rst, buf
                    torch.cuda.empty_cache()
                first = int(dec.prefill_chunked(pr, chunk=args.prefill_chunk)[-1].argmax())
                snap = dec.snapshot()
            for st in ("l12", "l12c"):  # l12c: the same step through torch.compile (follow-up F3a)
                if st not in args.settings:
                    continue
                try:
                    dec.compile_step(st == "l12c")
                    dec.restore(snap)
                    rep, rst, buf = dec.make_step(C, first)
                    ts = []
                    for _ in range(args.reps + 1):
                        dec.restore(snap)
                        rst()
                        ms, _ = ev_ms(lambda: [rep() for _ in range(args.n - 1)])
                        ts.append(ms / (args.n - 1))
                    row[f"{st}_ms"] = statistics.median(ts[1:])
                    row[f"{st}_tps"] = 1000 / row[f"{st}_ms"]
                    del rep, rst, buf
                except Exception as e:  # noqa: BLE001
                    if not oom(e):
                        raise
                    row[f"{st}_error"] = "does not fit"
                finally:
                    dec.compile_step(False)
                torch.cuda.empty_cache()
            for st in ("l7b", "l7bc"):  # l7bc: draft and verify through the fused helpers (F3a)
                if not has_draft or st not in args.settings:
                    continue
                try:
                    dec.compile_step(st == "l7bc")
                    dec.restore(snap)
                    ss = SelfSpecDecoder(dec, args.k, (), (), max_new=args.n, draft_w4=True)
                    ss.capture(pr, first)
                    ts, accs = [], []
                    for _ in range(args.reps + 1):
                        dec.restore(snap)
                        ss.start(pr, first)
                        torch.cuda.synchronize()
                        ms, ntok = ev_ms(lambda: spec_loop(ss, 1, args.n))
                        ts.append(ms / ntok)
                        accs.append(int(ss.buf["accepted"]) / (int(ss.buf["rounds"]) * args.k))
                    row[f"{st}_ms"] = statistics.median(ts[1:])
                    row[f"{st}_tps"] = 1000 / row[f"{st}_ms"]
                    row[f"{st}_accept"] = statistics.mean(accs[1:])
                    del ss
                except Exception as e:  # noqa: BLE001
                    if not oom(e):
                        raise
                    row[f"{st}_error"] = "does not fit"
                finally:
                    dec.compile_step(False)
                torch.cuda.empty_cache()
            for mk in args.mtp_ks:  # follow-up H: the model's own MTP layer as the draft, one-token steps fused
                st = f"mtp{mk}"
                try:
                    from ple.mtp_spec import MTPLayer, MTPSelfSpec
                    if not hasattr(dec, "_mtp_layer"):
                        dec._mtp_layer = MTPLayer(dec, cfg["model"]["path"], head=args.mtp_head)
                    dec._mtp_layer.ensure_len()          # the context (max_len) changes between rows
                    mode(dec, "fast", "fused")
                    dec.compile_step("l12c" in args.settings)   # draft rounds through the fused helpers when l12c is the reference
                    ss = MTPSelfSpec(dec, dec._mtp_layer, mk, max_new=args.n)
                    f1, hl = ss.prefill_long(pr, chunk=args.prefill_chunk)
                    sm = dec.snapshot()
                    reset = lambda: (dec.restore(sm), ss.reset(C, f1, hl))  # noqa: E731
                    ss.capture(pr, reset)
                    if args.check and Cn == args.contexts[0]:
                        from decoder_gate import judge
                        reset()
                        spec_loop(ss, 1, args.n)
                        tk = ss.buf["out"][:args.n]
                        mode(dec, "sdpa", "ref")
                        fz = dec.fuse
                        dec.compile_step(False)
                        dec.restore(sm)
                        fl = dec.prefill_chunked(pr, chunk=args.prefill_chunk)[-1:].float()
                        smr = dec.snapshot()
                        ext = dec.prefill_chunked(tk[:-1], start=C, logits_rows=args.n - 1).float()
                        dec.restore(smr)
                        v, nt = judge(torch.cat([fl, ext]), tk)
                        res["check"].append({"prompt": 0, "setting": st, "violations": v, "near_ties": nt})
                        print(f"check {st}: violations {v}, near-ties {nt} ({args.n} tokens)", flush=True)
                        mode(dec, "fast", "fused")
                        dec.compile_step(fz)
                        f1, hl = ss.prefill_long(pr, chunk=args.prefill_chunk)
                        sm = dec.snapshot()
                    ts, accs = [], []
                    for _ in range(args.reps + 1):
                        reset()
                        torch.cuda.synchronize()
                        ms, ntok = ev_ms(lambda: spec_loop(ss, 1, args.n))
                        ts.append(ms / ntok)
                        accs.append(int(ss.buf["accepted"]) / (int(ss.buf["rounds"]) * mk))
                    row[f"{st}_ms"] = statistics.median(ts[1:])
                    row[f"{st}_tps"] = 1000 / row[f"{st}_ms"]
                    row[f"{st}_accept"] = statistics.mean(accs[1:])
                    del ss
                except Exception as e:  # noqa: BLE001
                    if not oom(e):
                        raise
                    row[f"{st}_error"] = "does not fit"
                finally:
                    dec.compile_step(False)
                torch.cuda.empty_cache()
            res["rows"].append(row)
            print(f"C={Cn}: " + "  ".join(f"{k} {v:.1f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()
                                         if k.endswith(("tps", "accept", "error"))), flush=True)
            path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
