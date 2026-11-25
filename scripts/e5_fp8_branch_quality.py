"""E5 (quality half): uncorrected speculation with the speculative q/k/v computed in FP8 (E4M3 weights
and activations, one scale per tensor, torch._scaled_mm) vs in BF16. The latency half is in
e1_decode_bench.py (variants *_fp8). Success needs the FP8 branch's added KL ≤ 10% of speculation's own.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e_common import CAL, HOLD, Quality, exact_logprobs, gated_speculation, ridge  # noqa: E402

from ple.decode import _fp8_weight, fp8_linear
from ple.model_loader import decoder_layers, load, load_config, run_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--taus", type=float, nargs="*", default=[0.2, 0.3])
    ap.add_argument("--out", default="results/processed/e1_e5")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model, tok = load(cfg)
    layers = decoder_layers(model)
    n, dev = len(layers), torch.device(cfg["device"])
    d = model.config.get_text_config().hidden_size
    stem = run_name(cfg)
    pred = ridge(model, layers, tok, stem, dev)
    pred_f = {l: (lambda h, p=p: p(h)) for l, p in pred.items()}
    splits = {name: exact_logprobs(model, tok, span, dev) for name, span in (("cal", CAL), ("hold", HOLD))}
    fp8 = {}

    def fp8_branch(mod, x):
        if mod not in fp8:
            fp8[mod] = _fp8_weight(mod.weight)
        shape = x.shape
        return fp8_linear(x.reshape(-1, shape[-1]), *fp8[mod]).reshape(*shape[:-1], -1)

    sets = {"all": list(range(n - 1)), "start4_tail2": list(range(4, n - 3))}
    rows = []
    for set_name, bounds in sets.items():
        for tau in [None, *args.taus]:
            for branch_name, branch in (("bf16", None), ("fp8", fp8_branch)):
                for split in ("cal", "hold"):
                    ids_list, lps, exact_ppl = splits[split]
                    q, stats = Quality(), {"spec": 0, "total": 0}
                    with torch.inference_mode(), gated_speculation(layers, bounds, pred_f, d, tau=tau, stats=stats,
                                                                   branch=branch):
                        for ids, lp_exact in zip(ids_list, lps):
                            q.add(lp_exact, Quality.logprobs(model(input_ids=ids).logits), ids)
                    stats.pop("r")
                    r = {"set": set_name, "tau": tau, "branch": branch_name, "split": split} | q.row(exact_ppl) | {
                        "spec_share": stats["spec"] / stats["total"]}
                    rows.append(r)
                    print(f"{set_name:<13} tau {str(tau):<5} {branch_name:<5} {split:<4} spec {r['spec_share']:.3f} "
                          f"KL {r['kl']:.4f} top1 {r['top1']:.3f} ppl {r['ppl_change_pct']:+.2f}%", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "fp8": "E4M3, per-tensor scales",
            "exact_ppl": {k: v[2] for k, v in splits.items()}, "rows": rows}
    (out / f"{stem}_e5_fp8_quality.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
