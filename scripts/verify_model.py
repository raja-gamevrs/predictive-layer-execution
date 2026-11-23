"""Stage B smoke test: load the configured model, check its architecture, run one prefill, record metadata."""

import argparse
import json
import platform

import torch
import transformers
import yaml
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    m = cfg["model"]
    torch.manual_seed(cfg.get("seed", 0))

    config = AutoConfig.from_pretrained(m["id"], revision=m["revision"])
    for key, want in m.get("expected", {}).items():
        got = getattr(config, key)
        assert got == want, f"{key}: expected {want}, got {got}"

    tokenizer = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        m["id"], revision=m["revision"], torch_dtype=getattr(torch, m["dtype"]),
        attn_implementation=m.get("attn_implementation"),
    ).to(cfg["device"]).eval()

    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(cfg["device"])
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    torch.cuda.synchronize()

    print(json.dumps({
        "model": m["id"],
        "revision": m["revision"],
        "commit": getattr(config, "_commit_hash", None),
        "dtype": m["dtype"],
        "num_hidden_states": len(out.hidden_states),
        "logits_shape": list(out.logits.shape),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
    }, indent=2))


if __name__ == "__main__":
    main()
