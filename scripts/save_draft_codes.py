"""Follow-up J: quantise the L7b self-spec draft once (INT4 GPTQ + clipping search, groups of 32, LM head
included; GPTQ statistics from WikiText-2 train chunks 0-127) and save its exact codes, so this repo
(Qwen3Decoder.load_int4_codes) and llama.cpp (scripts/make_matched_gguf.py draft-q4_1) use the same draft."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lever_int4 import Hessians  # noqa: E402

from ple.data import wikitext_chunks  # noqa: E402
from ple.decode import Qwen3Decoder  # noqa: E402
from ple.model_loader import load, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/primary_qwen3_4b.yaml")
    ap.add_argument("--out", default="G:/ple_cache/draft_qwen3_4b_gptq_clip_g32.pt")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tok = load(cfg)
    dec = Qwen3Decoder(model, 1024)
    hess = Hessians(model, [c.to(dec.dev) for c in wikitext_chunks(tok, "train", 512, 128, skip=0)])
    with torch.inference_mode():
        dec.quantize_int4(group=32, method="gptq", clip=True, hessians=hess)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dec.save_int4_codes(args.out)
    print("saved", args.out, len(dec.w4_codes), "matrices")


if __name__ == "__main__":
    main()
