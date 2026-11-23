"""Load a configured model and locate its decoder blocks, across plain and multimodal wrappers."""

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load(cfg: dict):
    """Load from the HF cache at `revision`, or from a local snapshot `path` downloaded at that revision."""
    m = cfg["model"]
    source = {"pretrained_model_name_or_path": m["path"]} if m.get("path") else {
        "pretrained_model_name_or_path": m["id"], "revision": m["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(**source)
    model = AutoModelForCausalLM.from_pretrained(
        **source, dtype=getattr(torch, m["dtype"]), attn_implementation=m.get("attn_implementation"),
    ).to(cfg["device"]).eval()
    return model, tokenizer


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """The longest ModuleList named `layers`: the text decoder stack, also inside vision-language wrappers."""
    candidates = [mod for name, mod in model.named_modules()
                  if name.split(".")[-1] == "layers" and isinstance(mod, torch.nn.ModuleList)]
    if not candidates:
        raise ValueError("no decoder `layers` ModuleList found")
    return max(candidates, key=len)


def attention_input_projection(layer: torch.nn.Module, hidden_size: int) -> tuple[str, torch.nn.Linear]:
    """First Linear in the block that reads the residual width: q_proj, or the fused input projection of linear attention."""
    for name, mod in layer.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.in_features == hidden_size:
            return name, mod
    raise ValueError("no input projection found")
