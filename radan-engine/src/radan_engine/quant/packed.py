"""Load pre-quantised checkpoints that do not fit dequantised, keeping weights packed on the GPU.

Each quantised Linear is replaced by a PackedLinear that stores the checkpoint's packed tensors and
dequantises its weight to BF16 on every forward, then (optionally) quantises its input the way the
format's kernel does. Memory is the packed model plus one layer's temporaries; speed is not
representative. Supported:

  compressed-tensors `pack-quantized` INT4 (symmetric, group-wise)      W4A16
  ModelOpt FP8                                                           W8A8, static per-tensor input scale
  ModelOpt W4A16_NVFP4                                                   W4A16
  ModelOpt NVFP4                                                         W4A4, static global input scale,
                                                                         dynamic E4M3 block scales per 16
"""

import json
from fnmatch import fnmatch
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from radan_engine.quant import FP8_MAX, nvfp4_fake_quantize
from radan_engine.quant.modelopt import unpack_nvfp4


class PackedLinear(torch.nn.Module):
    def __init__(self, fmt: str, out_features: int, in_features: int, tensors: dict[str, torch.Tensor],
                 bias: torch.Tensor | None, group_size: int | None = None):
        super().__init__()
        self.fmt, self.out_features, self.in_features, self.group_size = fmt, out_features, in_features, group_size
        for k, v in tensors.items():
            self.register_buffer(k, v)
        self.bias = bias

    def dequantized_weight(self) -> torch.Tensor:
        if self.fmt == "int4_w4a16":
            from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
            q = unpack_from_int32(self.weight_packed, 4, torch.Size((self.out_features, self.in_features)))
            scale = self.weight_scale.repeat_interleave(self.group_size, dim=1)[:, :self.in_features]
            return (q.to(torch.bfloat16) * scale).to(torch.bfloat16)
        if self.fmt == "fp8_w8a8":
            return (self.weight.float() * self.weight_scale.float()).to(torch.bfloat16)
        if self.fmt in ("nvfp4_w4a16", "nvfp4_w4a4"):
            return unpack_nvfp4(self.weight, self.weight_scale, self.weight_scale_2, torch.bfloat16)
        raise ValueError(self.fmt)

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.fmt == "fp8_w8a8":
            s = self.input_scale.float()
            return ((x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s).to(x.dtype)
        if self.fmt == "nvfp4_w4a4":
            return nvfp4_fake_quantize(x, global_scale=self.input_scale)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(self.quantize_input(x), self.dequantized_weight().to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, {self.fmt} (packed, emulated)"


def _read(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for f in sorted(path.glob("*.safetensors")):
        if "mtp" in f.name:
            continue
        with safe_open(f, framework="pt", device="cpu") as st:
            for k in st.keys():
                if not k.startswith(("mtp.", "model.visual.")):
                    tensors[k] = st.get_tensor(k)
    return tensors


def _layer_formats(path: Path, config) -> tuple[dict[str, str], int | None]:
    """Checkpoint layer name → PackedLinear format, and the INT4 group size if any."""
    mo = path / "hf_quant_config.json"
    if mo.exists():
        q = json.loads(mo.read_text())["quantization"]
        names = {"FP8": "fp8_w8a8", "W4A16_NVFP4": "nvfp4_w4a16", "NVFP4": "nvfp4_w4a4"}
        return {k: names[v["quant_algo"]] for k, v in q.get("quantized_layers", {}).items()}, None
    qc = config.quantization_config if isinstance(config.quantization_config, dict) else config.quantization_config.to_dict()
    if qc.get("quant_method") != "compressed-tensors" or qc.get("format") != "pack-quantized":
        raise ValueError("unsupported quantisation config")
    (group,) = qc["config_groups"].values()
    w = group["weights"]
    if (w["num_bits"], w["type"], w["symmetric"], w["strategy"]) != (4, "int", True, "group"):
        raise ValueError(f"unsupported compressed-tensors scheme {w}")
    return {"*": "int4_w4a16"}, w["group_size"]


def load_packed(path: str, device: str, dtype=torch.bfloat16):
    path = Path(path)
    config = AutoConfig.from_pretrained(path)
    formats, group_size = _layer_formats(path, config)
    ignore = [] if "*" not in formats else (config.quantization_config.get("ignore", [])
                                            if isinstance(config.quantization_config, dict)
                                            else config.quantization_config.ignore)
    for c in (config, getattr(config, "text_config", None)):
        if c is not None and hasattr(c, "quantization_config"):
            delattr(c, "quantization_config")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    expected = model.state_dict()

    def model_name(k: str) -> str:
        return k.replace("model.language_model.", "model.", 1) if k not in expected else k

    tensors = {model_name(k): v for k, v in _read(path).items()}
    if "*" in formats:
        # compressed-tensors: a layer is quantised exactly when the checkpoint stores `weight_packed` for it.
        # (The publisher's ignore list also names parent modules such as `linear_attn` whose child projections
        # are packed, so it cannot be applied by prefix; layers it really excludes are stored as plain weights.)
        formats = {k[:-len(".weight_packed")]: "int4_w4a16" for k in tensors if k.endswith(".weight_packed")}
        plain = [model_name(p) for p in ignore if f"{model_name(p)}.weight_packed" in tensors]
        if plain:
            raise ValueError(f"ignored layers stored packed: {plain[:4]}")
    else:
        formats = {model_name(k): v for k, v in formats.items()}

    replaced = {}
    for name, fmt in formats.items():
        if f"{name}.weight" not in expected:
            continue
        out_f, in_f = expected[f"{name}.weight"].shape
        keys = [k for k in list(tensors) if k.startswith(name + ".") and k.count(".") == name.count(".") + 1]
        own = {k[len(name) + 1:]: tensors.pop(k).to(device) for k in keys}
        own.pop("weight_shape", None)
        bias = own.pop("bias", None)
        replaced[name] = PackedLinear(fmt, out_f, in_f, own, bias.to(dtype) if bias is not None else None, group_size)

    state = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in tensors.items() if k in expected}
    for name, mod in replaced.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, mod)
    model.load_state_dict(state, strict=False, assign=True)
    text_config = config.get_text_config()
    for name, mod in list(model.named_modules()):
        if any(b.is_meta for b in mod.buffers(recurse=False)) and "rotary" in type(mod).__name__.lower():
            parent, _, leaf = name.rpartition(".")
            with torch.device(device):
                fresh = type(mod)(config=text_config)
            setattr(model.get_submodule(parent) if parent else model, leaf, fresh)
    still_meta = [k for k, t in list(model.state_dict().items()) + list(model.named_buffers()) if t.is_meta]
    if still_meta:
        raise ValueError(f"tensors missing from checkpoint: {still_meta[:8]} ...")
    model = model.to(device).eval()
    counts = {}
    for m in replaced.values():
        counts[m.fmt] = counts.get(m.fmt, 0) + 1
    return model, AutoTokenizer.from_pretrained(path), counts
