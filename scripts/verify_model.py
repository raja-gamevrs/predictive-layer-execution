"""Stage B smoke test: load the configured model, check its architecture, run one prefill, record metadata."""

import argparse
import json
import platform

import torch
import transformers

from ple.activation_capture import capture_residual_stream
from ple.model_loader import attention_input_projection, decoder_layers, load, load_config
from ple.speculation_policy import attention_input_linears


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    m = cfg["model"]
    torch.manual_seed(cfg.get("seed", 0))
    model, tokenizer = load(cfg)
    text_cfg = model.config.get_text_config()
    for key, want in m.get("expected", {}).items():
        got = getattr(text_cfg, key)
        assert got == want, f"{key}: expected {want}, got {got}"

    layers = decoder_layers(model)
    d = text_cfg.hidden_size
    layer_types = getattr(text_cfg, "layer_types", None) or ["full_attention"] * len(layers)
    inputs = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(cfg["device"])
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        states = capture_residual_stream(model, layers, **inputs)
        out = model(**inputs)
    torch.cuda.synchronize()

    attn_inputs = {}
    for kind in sorted(set(layer_types)):
        i = layer_types.index(kind)
        attn_inputs[kind] = {"example_layer": i, "has_input_layernorm": hasattr(layers[i], "input_layernorm"),
                             "attention_input_linears": [n for n, mod in layers[i].named_modules()
                                                         if mod in attention_input_linears(layers[i], d)],
                             "first_input_projection": attention_input_projection(layers[i], d)[0]}

    print(json.dumps({
        "model": m["id"], "revision": m["revision"], "path": m.get("path"), "dtype": m["dtype"],
        "layers": len(layers), "hidden": d, "captured_states": len(states),
        "layer_type_counts": {k: layer_types.count(k) for k in sorted(set(layer_types))},
        "attention_inputs_by_layer_type": attn_inputs,
        "logits_shape": list(out.logits.shape),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__, "python": platform.python_version(), "os": platform.platform(),
    }, indent=2))


if __name__ == "__main__":
    main()
