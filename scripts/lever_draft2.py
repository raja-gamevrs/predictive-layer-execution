"""Follow-up F1 / F2 of docs/decode_levers.md: cheaper INT4 self-speculative drafts.

The L7b draft (INT4 GPTQ + clipping search, groups of 32) is the reference, re-measured in this run.
  F1  the same draft also skips the atlas-cheapest attention / MLP sub-layers (5%, 10%, 15% of MACs)
  F2  the draft's INT4 LM head keeps only the V most frequent tokens of WikiText-2 train chunks 0-127
      (the GPTQ calibration chunks), V in {8192, 32768}; its argmax is mapped back to full ids
Verify is the full BF16 model (L1 + L2 decoder), so output stays exact. Self-spec measured as L7b
(part_selfspec of lever_bench.py): k in {3, 4, 6, 8}, every round, 8 prompts per domain, 256 tokens,
2 interleaved pairs.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lever_bench import part_selfspec  # noqa: E402
from lever_int4 import Hessians  # noqa: E402
from self_spec_bench import atlas_skip_sets, code_prompts  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.decode import Qwen3Decoder  # noqa: E402
from ple.model_loader import load, load_config, run_name  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4, 6, 8])
    ap.add_argument("--skips", type=float, nargs="*", default=[0.05, 0.10, 0.15])
    ap.add_argument("--vocabs", type=int, nargs="*", default=[32768, 8192])
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--code-rev", default="a118bda")
    ap.add_argument("--out", default="results/processed/decode_levers")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = Qwen3Decoder(model, 512 + 256 + 8 * 9 + 16)
    calib = [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, 128, skip=0)]
    hess = Hessians(model, calib)
    prompts = {"cal": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=400)],
               "hold": [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, args.prompts, skip=600)],
               "code": [c.to(dec.dev) for c in code_prompts(tok, args.prompts, 512, args.code_rev)]}
    res = {"model": cfg["model"]["id"], "revision": cfg["model"]["revision"], "args": vars(args), "selfspec": []}
    path = Path(args.out) / f"{run_name(cfg)}_draft2.json"
    with torch.inference_mode():
        dec.quantize_int4(group=32, method="gptq", clip=True, hessians=hess)
        n = len(dec.layers)
        sets = atlas_skip_sets(n, tuple(args.skips)) if args.skips else {}
        sets.pop("blocks12_23", None)
        res["skip_sets"] = {k: {"attn": a, "mlp": m} for k, (a, m) in sets.items()}
        groups = [(None, {"L7b reference": (True, 1)} | {f"F1 skip {k}": (True, 1, a, m) for k, (a, m) in sets.items()})]
        counts = Counter(t for c in calib for t in c.tolist())
        for V in args.vocabs:
            groups.append((V, {f"F2 vocab {V}": (True, 1)}))
        res["vocab_coverage"] = {}
        for V, configs in groups:
            if V is None:
                dec.quantize_int4_head(None, hess)
            else:
                top = [t for t, _ in counts.most_common(V)]
                seen = len(top)  # calibration text may hold fewer distinct tokens than V: fill up with the lowest unused ids
                if seen < V:
                    used = set(top)
                    top += [t for t in range(dec.lm_head.shape[0]) if t not in used][:V - seen]
                ids = torch.tensor(sorted(top))
                res.setdefault("vocab_distinct_in_calibration", {})[str(V)] = seen
                dec.quantize_int4_head(ids, hess)
                keep = set(ids.tolist())
                cover = {d: sum(t in keep for p in prs for t in p.tolist()) / sum(p.numel() for p in prs)
                         for d, prs in prompts.items()}
                res["vocab_coverage"][str(V)] = cover
                print(f"vocab {V}: share of prompt tokens covered " + "  ".join(f"{d} {100 * c:.1f}%" for d, c in cover.items()), flush=True)
            r = {}
            part_selfspec(dec, prompts, 256, args.pairs, r, args.ks, configs)
            res["selfspec"].extend(r["selfspec"])
            path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        dec.quantize_int4_head(None, hess)
    print("wrote", path)


if __name__ == "__main__":
    main()
