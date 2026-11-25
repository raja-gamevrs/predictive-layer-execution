"""Self-speculative decoding with atlas-guided layer skipping (docs/self_speculative_decoding.md).

For each draft skip set and draft length k, on calibration / holdout WikiText prompts and code prompts
(this repo's own .py files): greedy-decode --new tokens, compare with exact greedy decoding token by
token (divergences are reported with the exact model's top-2 logit gap), and time both with CUDA graphs.
"""

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path

import torch

from ple.data import wikitext_chunks
from ple.decode import Qwen3Decoder
from ple.model_loader import load, load_config, run_name
from ple.self_spec import SelfSpecDecoder

ATLAS = "results/processed/Qwen3-4B-Base_sensitivity_atlas.csv"


def atlas_skip_sets(n_layers: int, targets=(0.25, 0.40, 0.55)) -> dict[str, tuple[list[int], list[int]]]:
    """Attention / MLP sub-layers ranked by atlas skip-mode KL per MAC (block 0 and the last never skipped),
    taken greedily until each target share of decoder MACs is reached."""
    rows = [r for r in csv.DictReader(open(ATLAS, encoding="utf-8")) if r["mode"] == "skip"]
    total = sum(float(r["op_macs"]) for r in rows if r["op"] == "block")
    cand = sorted((r for r in rows if r["op"] in ("attn", "mlp") and 0 < int(r["layer"]) < n_layers - 1),
                  key=lambda r: float(r["kl"]) / float(r["op_macs"]))
    out = {}
    for t in targets:
        share, attn, mlp = 0.0, [], []
        for r in cand:
            if share >= t:
                break
            share += float(r["op_macs"]) / total
            (attn if r["op"] == "attn" else mlp).append(int(r["layer"]))
        out[f"atlas{round(100 * share)}"] = (sorted(attn), sorted(mlp))
    mid = list(range(12, 24))
    out["blocks12_23"] = (mid, mid)
    return out


def code_prompts(tok, n: int, length: int, rev: str | None = None) -> list[torch.Tensor]:
    """n prompts of `length` tokens spread over this repo's Python files (src, scripts, tests). rev: read
    the files at that git commit instead of the working tree, so later edits do not change the prompts."""
    if rev is None:
        files = sorted(p for d in ("src", "scripts", "tests") for p in Path(d).rglob("*.py"))
        text = "\n\n".join(p.read_text(encoding="utf-8") for p in files)
    else:
        names = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev, "--", "src", "scripts", "tests"],
                               capture_output=True, text=True, check=True).stdout.split()
        names = sorted((x for x in names if x.endswith(".py")), key=Path)
        text = "\n\n".join(subprocess.run(["git", "show", f"{rev}:{x}"], capture_output=True, check=True)
                           .stdout.decode("utf-8") for x in names)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    step = max(length, (len(ids) - length) // n)
    return [torch.tensor(ids[i * step:i * step + length]) for i in range(n) if i * step + length <= len(ids)]


def exact_greedy(dec, prompt, n_new, with_gaps):
    """Exact greedy tokens from the graph-captured single-token step, plus top-2 logit gaps."""
    first = int(dec.prefill(prompt)[-1].argmax())
    replay, reset, buf = dec.make_step(prompt.shape[0], first)
    dec.prefill(prompt)
    reset()
    toks, gaps = [first], []
    for _ in range(n_new - 1):
        replay()
        toks.append(int(buf["tok"]) if with_gaps else None)
        if with_gaps:
            top = buf["logits"][0].float().topk(2).values
            gaps.append(float(top[0] - top[1]))
    return toks, gaps, replay, reset


NEAR_TIE = 0.25  # two BF16 ulps at |logit| 16-32


def greedy_check(dec, prompt, toks):
    """Teacher-forced check with one full-model pass over prompt + toks: is every token the model's argmax
    given its prefix? Returns (violations with top-2 gap > NEAR_TIE, near-tie mismatches)."""
    seq = torch.cat([prompt, torch.tensor(toks[:-1], device=prompt.device)])
    logits = dec.prefill(seq)[prompt.shape[0] - 1:].float()
    top = logits.topk(2, dim=-1)
    gap = top.values[:, 0] - top.values[:, 1]
    wrong = top.indices[:, 0] != torch.tensor(toks, device=prompt.device)
    return int((wrong & (gap > NEAR_TIE)).sum()), int((wrong & (gap <= NEAR_TIE)).sum())


def time_exact(dec, prompt, n_new, runs):
    first = int(dec.prefill(prompt)[-1].argmax())
    replay, reset, _ = dec.make_step(prompt.shape[0], first)
    times = []
    for _ in range(runs + 1):
        dec.prefill(prompt)
        reset()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_new - 1):
            replay()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times[1:])


def _decode_ms(fn):
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    n = fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e), n


def time_pair(dec, ss, prompt, n_new, runs):
    """Interleaved exact / self-spec timing on one prompt, decode only (prefill excluded from both).

    Returns (exact ms per token, self-spec ms per token): medians of `runs` alternating runs each,
    after one untimed warm-up of each. Tokens counted after the first (which both take from prefill).
    """
    first = int(dec.prefill(prompt)[-1].argmax())
    replay, reset, _ = dec.make_step(prompt.shape[0], first)

    def exact():
        dec.prefill(prompt)
        reset()
        return _decode_ms(lambda: [replay() for _ in range(n_new - 1)] and n_new - 1)

    def spec():
        ss.start(prompt)
        b = ss.buf

        def loop():
            while True:
                ss.graph.replay()
                if int(b["n_out"]) >= n_new:
                    return int(b["n_out"]) - 1
        return _decode_ms(loop)

    exact(), spec()
    te, ts = [], []
    for _ in range(runs):
        ms, n = exact()
        te.append(ms / n)
        ms, n = spec()
        ts.append(ms / n)
    return statistics.median(te), statistics.median(ts)


def time_only(args, dec, domains, sets, cfg, gpu_state) -> None:
    """Re-time every (set, k, prompt) against an exact run of the same prompt adjacent in time.

    A pair whose exact ms/token is > 5% above the fastest exact seen is treated as contended and
    re-timed at the end (up to 3 more times); pairs still contended are flagged, not dropped.
    """
    rows = []
    with torch.inference_mode():
        for name, (attn, mlp) in sets.items():
            for k in args.ks:
                ss = SelfSpecDecoder(dec, k, attn, mlp, max_new=args.new)
                ss.capture(domains["cal"][0].to(dec.dev))
                for dom, prompts in domains.items():
                    for i, pr in enumerate(prompts):
                        pr = pr.to(dec.dev)
                        out, rounds, acc = ss.generate(pr, args.new)
                        e_ms, s_ms = time_pair(dec, ss, pr, args.new, args.runs)
                        rows.append({"set": name, "k": k, "domain": dom, "prompt": i, "rounds": rounds,
                                     "accepted_drafts": acc, "tokens_per_round": args.new / rounds,
                                     "accept_rate": acc / (rounds * k), "exact_ms_per_token": e_ms,
                                     "ms_per_token": s_ms, "speedup": e_ms / s_ms, "retimed": 0})
                best = min(r["exact_ms_per_token"] for r in rows)
                for tries in range(1, 4):
                    bad = [r for r in rows if r["set"] == name and r["k"] == k and r["exact_ms_per_token"] > 1.05 * best]
                    if not bad:
                        break
                    for r in bad:
                        pr = domains[r["domain"]][r["prompt"]].to(dec.dev)
                        e_ms, s_ms = time_pair(dec, ss, pr, args.new, args.runs)
                        r |= {"exact_ms_per_token": e_ms, "ms_per_token": s_ms, "speedup": e_ms / s_ms, "retimed": tries}
                    best = min(best, min(r["exact_ms_per_token"] for r in bad))
                sub = [r for r in rows if r["set"] == name and r["k"] == k]
                sp = {d: statistics.mean(r["speedup"] for r in sub if r["domain"] == d) for d in domains}
                ex = statistics.median(r["exact_ms_per_token"] for r in sub)
                print(f"{name:<12} k={k}  speedup " + "  ".join(f"{d} {v:.3f}x" for d, v in sp.items()) +
                      f"  exact {ex:.2f} ms/token  retimed {sum(r['retimed'] > 0 for r in sub)}", flush=True)
                del ss
                torch.cuda.empty_cache()
    best = min(r["exact_ms_per_token"] for r in rows)
    for r in rows:
        r["contended"] = r["exact_ms_per_token"] > 1.05 * best
    gpu_after = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm",
                                "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "prompt_len": args.prompt_len,
            "new_tokens": args.new, "ks": args.ks, "sets": sets, "timed": True, "runs": args.runs,
            "protocol": "interleaved exact/self-spec pairs per prompt, decode only, contended pairs re-timed",
            "gpu_state_before": gpu_state, "gpu_state_after": gpu_after, "best_exact_ms_per_token": best,
            "rows": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run_name(cfg)}_self_spec_timed.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"contended pairs left: {sum(r['contended'] for r in rows)} / {len(rows)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--time", action="store_true", help="also time (only on an otherwise idle GPU)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--time-only", action="store_true",
                    help="interleaved exact/self-spec timing only (no correctness check); writes *_timed.json")
    ap.add_argument("--out", default="results/processed/self_spec")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tok = load(cfg)
    n_layers = model.config.num_hidden_layers
    kmax = max(args.ks)
    dec = Qwen3Decoder(model, args.prompt_len + args.new + kmax + 16)
    domains = {"cal": [c for c in wikitext_chunks(tok, "train", args.prompt_len, args.prompts, skip=400)],
               "hold": [c for c in wikitext_chunks(tok, "train", args.prompt_len, args.prompts, skip=600)],
               "code": code_prompts(tok, args.prompts, args.prompt_len)}
    sets = {"none": ([], [])} | atlas_skip_sets(n_layers)  # none: draft = full model (acceptance check)
    if args.sets:
        sets = {k: v for k, v in sets.items() if k in args.sets}
    gpu_state = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm",
                                "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()

    if args.time_only:
        time_only(args, dec, domains, sets, cfg, gpu_state)
        return
    rows, exact_rows = [], []
    with torch.inference_mode():
        ref = {}
        for dom, prompts in domains.items():
            for i, pr in enumerate(prompts):
                pr = pr.to(dec.dev)
                toks, gaps, _, _ = exact_greedy(dec, pr, args.new, True)
                ref[(dom, i)] = (toks, gaps)
                v, nt = greedy_check(dec, pr, toks)
                exact_rows.append({"domain": dom, "prompt": i, "greedy_violations": v, "near_tie_mismatches": nt})
                if args.time:
                    ms = time_exact(dec, pr, args.new, args.runs)
                    exact_rows[-1] |= {"ms": ms, "ms_per_token": ms / (args.new - 1)}
        for name, (attn, mlp) in sets.items():
            for k in args.ks:
                ss = SelfSpecDecoder(dec, k, attn, mlp, max_new=args.new)
                ss.capture(domains["cal"][0].to(dec.dev))
                for dom, prompts in domains.items():
                    for i, pr in enumerate(prompts):
                        pr = pr.to(dec.dev)
                        out, rounds, acc = ss.generate(pr, args.new)
                        got, (want, gaps) = out.tolist(), ref[(dom, i)]
                        bad = [j for j, (a, b) in enumerate(zip(got, want)) if a != b]
                        first_bad = bad[0] if bad else None
                        row = {"set": name, "k": k, "domain": dom, "prompt": i, "rounds": rounds,
                               "accepted_drafts": acc, "tokens_per_round": args.new / rounds,
                               "accept_rate": acc / (rounds * k), "identical": not bad, "first_divergence": first_bad,
                               "gap_at_divergence": gaps[first_bad - 1] if first_bad else None}
                        row["greedy_violations"], row["near_tie_mismatches"] = greedy_check(dec, pr, got)
                        if args.time:
                            times = []
                            for _ in range(args.runs + 1):
                                torch.cuda.synchronize()
                                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                s.record()
                                ss.generate(pr, args.new)
                                e.record()
                                torch.cuda.synchronize()
                                times.append(s.elapsed_time(e))
                            row["ms"] = statistics.median(times[1:])
                        rows.append(row)
                sub = [r for r in rows if r["set"] == name and r["k"] == k]
                summary = {d: statistics.mean(r["tokens_per_round"] for r in sub if r["domain"] == d) for d in domains}
                ident = sum(r["identical"] for r in sub)
                viol = sum(r["greedy_violations"] for r in sub)
                ties = sum(r["near_tie_mismatches"] for r in sub)
                print(f"{name:<12} k={k}  tokens/round " + "  ".join(f"{d} {v:.2f}" for d, v in summary.items()) +
                      f"  identical to 1-token greedy {ident}/{len(sub)}  greedy violations {viol}  near-tie {ties}" +
                      "".join(f"  [{r['domain']}{r['prompt']} diverges at {r['first_divergence']}, gap {r['gap_at_divergence']:.3f}]"
                              for r in sub if not r["identical"] and r["gap_at_divergence"] is not None), flush=True)
                del ss
                torch.cuda.empty_cache()

    if args.time:
        base = {(r["domain"], r["prompt"]): r["ms"] for r in exact_rows}
        for r in rows:
            r["speedup"] = base[(r["domain"], r["prompt"])] / r["ms"]
        for name in sets:
            for k in args.ks:
                sp = {d: statistics.mean(r["speedup"] for r in rows if r["set"] == name and r["k"] == k and r["domain"] == d)
                      for d in domains}
                print(f"{name:<12} k={k}  speedup " + "  ".join(f"{d} {v:.3f}x" for d, v in sp.items()), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "prompt_len": args.prompt_len,
            "new_tokens": args.new, "ks": args.ks, "sets": sets, "timed": args.time, "runs": args.runs,
            "gpu_state_before": gpu_state, "exact": exact_rows, "rows": rows}
    (out / f"{run_name(cfg)}_self_spec{'_timed' if args.time else ''}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
