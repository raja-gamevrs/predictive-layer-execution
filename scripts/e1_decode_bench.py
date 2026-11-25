"""E1 / E5: end-to-end single-stream decode latency with PLE speculation (docs/experiments_e1_e5.md).

Checks the minimal decoder (ple.decode) against Transformers first: prefill logits, teacher-forced
decode logits, and uncorrected identity speculation against the hook-based simulation. Then, per
variant, captures one decode step in a CUDA graph and times `--steps` back-to-back replays after a
`--prompt`-token prefill, `--runs` times. Predictor weights for timing are random (cost only).
"""

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch

from ple.data import wikitext_chunks
from ple.decode import Qwen3Decoder, SpecPlan
from ple.model_loader import decoder_layers, load, load_config, run_name
from ple.speculation_policy import speculate_attention_inputs


def kl_rel(ref: torch.Tensor, got: torch.Tensor) -> dict:
    lp_r, lp_g = torch.log_softmax(ref.float(), -1), torch.log_softmax(got.float(), -1)
    return {"kl": (lp_r.exp() * (lp_r - lp_g)).sum(-1).mean().item(),
            "top1": (ref.argmax(-1) == got.argmax(-1)).float().mean().item(),
            "rel_err": ((got.float() - ref.float()).norm() / ref.float().norm()).item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--reverse", action="store_true", help="time variants in reverse order (order-effect check)")
    ap.add_argument("--out", default="results/processed/e1_e5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.get("seed", 0))
    model, tok = load(cfg)
    layers = decoder_layers(model)
    n, d = len(layers), model.config.hidden_size
    dev = torch.device(cfg["device"])
    max_len = args.prompt + max(args.steps, 32) + 8  # 32 teacher-forced check steps
    dec = Qwen3Decoder(model, max_len)
    text = wikitext_chunks(tok, "train", args.prompt + 64, 1, skip=400)[0].to(dev)  # calibration chunk 400
    prompt, follow = text[:args.prompt], text[args.prompt:args.prompt + 32]

    # ---- correctness checks ----
    checks = {}
    with torch.inference_mode():
        ref = model(input_ids=text[None, :args.prompt + 32]).logits[0]
        got = dec.prefill(prompt)
        checks["prefill_vs_hf"] = kl_rel(ref[1:args.prompt], got[1:])
        steps = []
        for j, t in enumerate(follow):
            pos = torch.tensor([args.prompt + j], device=dev)
            mask = torch.full((1, 1, 1, max_len), float("-inf"), device=dev, dtype=model.dtype)
            mask[..., :args.prompt + j + 1] = 0
            steps.append(dec.forward(t[None], pos, mask)[0])
        checks["decode_vs_hf"] = kl_rel(ref[args.prompt:args.prompt + 32], torch.stack(steps))
        sources = list(range(n - 1))
        with speculate_attention_inputs(layers, sources, {l: (lambda h: h) for l in sources},
                                        lambda l: (lambda mod, y_hat, e: y_hat), d):
            ref_spec = model(input_ids=prompt[None]).logits[0]
        dec.set_plan(SpecPlan("overlap", sources))
        got_spec = dec.prefill(prompt)
        checks["spec_identity_vs_hook_sim"] = kl_rel(ref_spec[1:], got_spec[1:])
        dec.set_plan(SpecPlan())
        torch.cuda.synchronize()
    print(json.dumps(checks, indent=2), flush=True)

    # ---- timing ----
    rand = {l: ((torch.randn(d, d, device=dev, dtype=model.dtype) * d**-0.5 * 0.1),
                torch.zeros(d, device=dev, dtype=model.dtype)) for l in range(n - 1)}
    every = list(range(n - 1))
    tol9 = list(range(24, n - 3))            # start 24, last 2 exact (Phase I: 9 blocks, +0.9% ppl)
    start16 = list(range(16, n - 3))         # 17 blocks
    alternate = list(range(0, n - 1, 2))
    variants = {
        "exact": SpecPlan(),
        "exact_fp8_qkv": SpecPlan("none", [], None, fp8=True),
        "overlap_identity_all": SpecPlan("overlap", every),
        "overlap_linear_all": SpecPlan("overlap", every, rand),
        "overlap_linear_start16": SpecPlan("overlap", start16, rand),
        "overlap_linear_tol9": SpecPlan("overlap", tol9, rand),
        "fused_identity_alternate": SpecPlan("fused", alternate),
        "overlap_identity_all_fp8": SpecPlan("overlap", every, None, fp8=True),
        "overlap_linear_all_fp8": SpecPlan("overlap", every, rand, fp8=True),
        "overlap_linear_start16_fp8": SpecPlan("overlap", start16, rand, fp8=True),
        "overlap_linear_tol9_fp8": SpecPlan("overlap", tol9, rand, fp8=True),
    }
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants or k == "exact"}
    if args.reverse:
        variants = dict(reversed(list(variants.items())))

    gpu_state = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used",
                                "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    results = {}
    with torch.inference_mode():
        first = int(dec.prefill(prompt)[-1].argmax())
        for name, plan in variants.items():
            dec.set_plan(plan)
            dec.prefill(prompt)  # cache consistent with this plan
            replay, reset, buf = dec.make_step(args.prompt, first)
            per_run = []
            for r in range(args.runs + 1):  # first run is warm-up
                reset()
                torch.cuda.synchronize()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(args.steps):
                    replay()
                e.record()
                torch.cuda.synchronize()
                if r:
                    per_run.append(s.elapsed_time(e) / args.steps)
            reset()
            toks = []
            for _ in range(16):
                replay()
                toks.append(int(buf["tok"]))
            med = statistics.median(per_run)
            results[name] = {"ms_per_token": med, "runs_ms": per_run,
                             "cv": statistics.pstdev(per_run) / statistics.mean(per_run),
                             "speculated_blocks": len(plan.sources), "mode": plan.mode, "fp8": plan.fp8,
                             "predictor": "identity" if plan.predictor is None else "linear(random, timing only)",
                             "first_16_tokens": tok.decode(toks)}
            print(f"{name:<28} {med:.4f} ms/token  cv {results[name]['cv']:.3f}  blocks {len(plan.sources)}", flush=True)
            del replay
            torch.cuda.empty_cache()
    base = results["exact"]["ms_per_token"]
    for name, r in results.items():
        r["saving_pct"] = 100 * (base - r["ms_per_token"]) / base
        print(f"{name:<28} saving {r['saving_pct']:+.2f}%")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "dtype": cfg["model"]["dtype"],
            "prompt_tokens": args.prompt, "decode_steps": args.steps, "runs": args.runs, "cuda_graphs": True,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "gpu_state_before": gpu_state,
            "order": list(variants), "checks": checks, "results": results}
    (out / f"{run_name(cfg)}_e1_decode{'_reverse' if args.reverse else ''}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
