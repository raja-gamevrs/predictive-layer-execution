"""Load NVIDIA ModelOpt (`hf_quant_config.json`, quant_algo MIXED_PRECISION) checkpoints for quality studies.

Supported per-layer formats, both emulated in BF16 with the checkpoint's own scales:
  FP8           weight E4M3 × per-tensor `weight_scale`; activations quantised statically to E4M3 with
                the per-tensor `input_scale` (x → clamp(x / s, ±448) → E4M3 → × s) — W8A8.
  W4A16_NVFP4   weight packed E2M1 (two per byte, low nibble first) × E4M3 `weight_scale` per 16
                inputs × FP32 `weight_scale_2`; activations unquantised — W4A16.
Unquantised tensors load as stored (BF16). Speed is not representative; numerics follow the format
up to matmul accumulation order.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ple.quant import E2M1, FP8_MAX

_CODES = torch.cat([E2M1, -E2M1])  # 4-bit code: bit 3 = sign, bits 0–2 = magnitude index


class FP8W8A8Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, input_scale: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.register_buffer("input_scale", input_scale.float())
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.input_scale
        xq = ((x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s).to(x.dtype)
        return torch.nn.functional.linear(xq, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, FP8 W8A8 static (emulated)"


class DequantLinear(torch.nn.Module):
    """Weight-only quantised layer, dequantised once at load (activations untouched)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, fmt: str):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias, self.fmt = bias, fmt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, {self.fmt} (dequantised)"


def unpack_nvfp4(packed: torch.Tensor, block_scale: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """uint8 (out, in/2) + E4M3 (out, in/16) + FP32 scalar → float32 (out, in)."""
    codes = torch.stack([packed & 0x0F, packed >> 4], dim=-1).reshape(packed.shape[0], -1).long()
    values = _CODES.to(packed.device)[codes]
    scale = block_scale.float().repeat_interleave(16, dim=1) * global_scale.float()
    return values * scale


def load_modelopt(path: str, device: str, dtype=torch.bfloat16):
    path = Path(path)
    quant = json.loads((path / "hf_quant_config.json").read_text())["quantization"]
    layer_algo = {k: v["quant_algo"] for k, v in quant.get("quantized_layers", {}).items()}

    config = AutoConfig.from_pretrained(path)
    for c in (config, getattr(config, "text_config", None)):
        if c is not None and hasattr(c, "quantization_config"):
            delattr(c, "quantization_config")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    expected = model.state_dict()

    files = sorted(path.glob("*.safetensors"))
    tensors = {}
    for f in files:
        with safe_open(f, framework="pt", device=device) as st:
            for k in st.keys():
                tensors[k] = st.get_tensor(k)

    state, replaced = {}, {}
    for name, algo in layer_algo.items():
        if f"{name}.weight" not in expected:
            continue  # e.g. layers the HF model does not instantiate
        w = tensors.pop(f"{name}.weight")
        bias = tensors.pop(f"{name}.bias", None)
        if algo == "FP8":
            weight = (w.float() * tensors.pop(f"{name}.weight_scale").float()).to(dtype)
            replaced[name] = FP8W8A8Linear(weight, tensors.pop(f"{name}.input_scale"), bias)
        elif algo == "W4A16_NVFP4":
            weight = unpack_nvfp4(w, tensors.pop(f"{name}.weight_scale"), tensors.pop(f"{name}.weight_scale_2")).to(dtype)
            tensors.pop(f"{name}.input_scale", None)
            replaced[name] = DequantLinear(weight, bias, "NVFP4 W4A16")
        else:
            raise ValueError(f"{name}: unsupported quant_algo {algo}")
    for k in expected:
        if k in tensors:
            state[k] = tensors[k].to(dtype) if tensors[k].is_floating_point() else tensors[k]

    for name, mod in replaced.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, mod)
    missing = [k for k in model.state_dict() if k not in state and not any(k.startswith(n + ".") for n in replaced)]
    model.load_state_dict(state, strict=False, assign=True)
    if missing:
        # Non-persistent or derived buffers (e.g. rotary inv_freq) are rebuilt by the model on device.
        still_meta = [k for k, t in model.state_dict().items() if t.is_meta]
        if still_meta:
            raise ValueError(f"tensors missing from checkpoint: {still_meta[:8]} ...")
    model = model.to(device).eval()
    counts = {a: sum(1 for n in replaced if layer_algo[n] == a) for a in set(layer_algo.values())}
    return model, AutoTokenizer.from_pretrained(path), counts
