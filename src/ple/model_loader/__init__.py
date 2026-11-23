"""Load a configured model and locate its decoder blocks, across plain and multimodal wrappers."""

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_name(cfg: dict) -> str:
    """File stem for results: model name, plus the quantization when there is one."""
    m = cfg["model"]
    return m["id"].split("/")[-1] + (f"_{m['quantization']}" if m.get("quantization") else "")  # format is in the id


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def quantization_config(name: str | None):
    """Load-time quantization of a BF16 checkpoint. Pre-quantized checkpoints carry their own config instead.

    int8_weight_only  torchao INT8 weights, per-channel scales, BF16 activations (W8A16): linear in its input.
    nvfp4             Transformers NVFP4: W4A16 GEMV for single-token decode, W4A4 GEMM (dynamic FP4
                      activation quantization) for prefill, so it is not linear in its input during prefill.
                      Needs a Hub kernel build for the platform (none for Windows / torch 2.14 / cu130).
    nvfp4_w4a16_emulated, nvfp4_w4a4_emulated
                      NVFP4 numerics emulated in BF16 (ple.quant), applied after loading; see `load`.
    """
    if name is None or name.endswith("_emulated"):
        return None
    if name == "int8_weight_only":
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import TorchAoConfig
        return TorchAoConfig(quant_type=Int8WeightOnlyConfig())
    if name == "nvfp4":
        from transformers.utils.quantization_config import NVFP4Config
        return NVFP4Config()
    raise ValueError(f"unknown quantization {name!r}")


def load(cfg: dict):
    """Load from the HF cache at `revision`, or from a local snapshot `path` downloaded at that revision."""
    m = cfg["model"]
    if m.get("format") == "modelopt":
        from ple.quant.modelopt import load_modelopt
        model, tokenizer, _ = load_modelopt(m["path"], cfg["device"], getattr(torch, m["dtype"]))
        return model, tokenizer
    source = {"pretrained_model_name_or_path": m["path"]} if m.get("path") else {
        "pretrained_model_name_or_path": m["id"], "revision": m["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(**source)
    quant = quantization_config(m.get("quantization"))
    kwargs = {"dtype": getattr(torch, m["dtype"]), "attn_implementation": m.get("attn_implementation")}
    if quant is not None:
        kwargs |= {"quantization_config": quant, "device_map": cfg["device"]}
    model = AutoModelForCausalLM.from_pretrained(**source, **kwargs)
    if quant is None and getattr(model, "hf_quantizer", None) is None:
        model = model.to(cfg["device"])
    if m.get("quantization", "").startswith("nvfp4_w4a"):
        from ple.quant import emulate_nvfp4
        emulate_nvfp4(decoder_layers(model), quantize_activations=m["quantization"] == "nvfp4_w4a4_emulated")
    return model.eval(), tokenizer


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """The longest ModuleList named `layers`: the text decoder stack, also inside vision-language wrappers."""
    candidates = [mod for name, mod in model.named_modules()
                  if name.split(".")[-1] == "layers" and isinstance(mod, torch.nn.ModuleList)]
    if not candidates:
        raise ValueError("no decoder `layers` ModuleList found")
    return max(candidates, key=len)


def attention_input_projection(layer: torch.nn.Module, hidden_size: int) -> tuple[str, torch.nn.Module]:
    """First attention input projection of the block: q_proj, or the fused input projection of linear attention.

    Output projections are excluded even when their input width happens to equal the hidden size.
    """
    from ple.speculation_policy import attention_input_linears
    found = attention_input_linears(layer, hidden_size)
    if not found:
        raise ValueError("no input projection found")
    return next(n for n, m in layer.named_modules() if m is found[0]), found[0]
