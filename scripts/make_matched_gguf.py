"""Follow-up J of docs/decode_levers.md: GGUF files whose 4-bit weights are exactly this repo's.

  draft-q4_1   Qwen3-4B draft: copy of the BF16 GGUF whose attention / MLP matrices, plus a new untied
               `output.weight` (the draft LM head), are Q4_1 blocks built from this repo's saved INT4
               draft codes (scripts/lever_draft2-style GPTQ + clipping, groups of 32): value = q * d + m with
               d = the kernel's bf16 scale and m = zero - 8 * scale (rounded to FP16). Token embeddings stay BF16.
  redhat-q4_0  Qwen3.8: copy of the BF16 GGUF whose INT4 matrices are Q4_0 blocks holding the RedHat
               checkpoint's own codes and scales (each group of 128 = four Q4_0 blocks with the same scale),
               with llama.cpp's converter's V-head reordering applied to codes and scales; everything the
               checkpoint keeps in BF16 (LM head, embeddings, a / b projections, MTP layer) stays BF16.
               Each written tensor is checked: bf16(dequantised) == the converter's BF16 tensor.

Run with llama.cpp's gguf-py on the path (G:/llamacpp/src/llama.cpp-b12359/gguf-py).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

import gguf


def blocks_q4_0(codes, d):
    """codes (out, in) uint8 0..15, d (out, in // 32) float -> (out, in // 32 * 18) uint8 (ggml block_q4_0)."""
    out_f, in_f = codes.shape
    c = codes.view(out_f, in_f // 32, 32)
    qs = c[..., :16] | (c[..., 16:] << 4)
    dh = d.to(torch.float16).contiguous().view(torch.uint8).view(out_f, in_f // 32, 2)
    return torch.cat([dh, qs], -1).reshape(out_f, -1).numpy()


def blocks_q4_1(codes, d, m):
    """value = q * d + m; ggml block_q4_1 = (d fp16, m fp16, 16 bytes of nibbles)."""
    out_f, in_f = codes.shape
    c = codes.view(out_f, in_f // 32, 32)
    qs = c[..., :16] | (c[..., 16:] << 4)
    dh = d.to(torch.float16).contiguous().view(torch.uint8).view(out_f, in_f // 32, 2)
    mh = m.to(torch.float16).contiguous().view(torch.uint8).view(out_f, in_f // 32, 2)
    return torch.cat([dh, mh, qs], -1).reshape(out_f, -1).numpy()


def copy_fields(reader, writer):
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def write(reader, out, replace: dict, extra: dict):
    """replace / extra: name -> (uint8 byte array, GGMLQuantizationType)."""
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(str(out), arch)
    copy_fields(reader, writer)
    tensors = [(t.name, replace.get(t.name)) for t in reader.tensors] + [(n, v) for n, v in extra.items()]
    by_name = {t.name: t for t in reader.tensors}
    for name, rep in tensors:
        if rep is None:
            t = by_name[name]
            writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        else:
            arr, qt = rep
            writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, qt)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for name, rep in tensors:
        writer.write_tensor_data(by_name[name].data if rep is None else rep[0], tensor_endianess=reader.endianess)
    writer.close()


def bf16_tensor(t):
    a = np.array(t.data)
    return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16).reshape(list(reversed([int(x) for x in t.shape])))


def draft_q4_1(args):
    reader = gguf.GGUFReader(args.base)
    d = torch.load(args.codes)
    codes, (group, method, clip) = d["codes"], d["args"]
    assert group == 32, "Q4_1 blocks are groups of 32"
    n_layers = 1 + max(i for i, _ in codes if i != "head")
    shapes = {t.name: tuple(reversed([int(x) for x in t.shape])) for t in reader.tensors}
    replace = {}

    def q41(entry):
        c, sc, zero, g = entry
        sc, zero = sc.float(), zero.float()
        return blocks_q4_1(c, sc, zero - 8 * sc)
    for i in range(n_layers):
        nq, nk = shapes[f"blk.{i}.attn_q.weight"][0], shapes[f"blk.{i}.attn_k.weight"][0]
        c, sc, zero, g = codes[(i, "qkv")]
        for name, lo, hi in (("attn_q", 0, nq), ("attn_k", nq, nq + nk), ("attn_v", nq + nk, c.shape[0])):
            if f"blk.{i}.{name}.weight" not in shapes:   # Gemma 4 full-attention layers: K = V, no attn_v
                assert lo == hi, f"layer {i}: rows left for a missing {name}"
                continue
            part = (c[lo:hi], sc[lo:hi], zero[lo:hi], g)
            assert part[0].shape == shapes[f"blk.{i}.{name}.weight"], name
            replace[f"blk.{i}.{name}.weight"] = (q41(part), gguf.GGMLQuantizationType.Q4_1)
        replace[f"blk.{i}.attn_output.weight"] = (q41(codes[(i, "o")]), gguf.GGMLQuantizationType.Q4_1)
        c, sc, zero, g = codes[(i, "gu")]
        ni = c.shape[0] // 2
        replace[f"blk.{i}.ffn_gate.weight"] = (q41((c[:ni], sc[:ni], zero[:ni], g)), gguf.GGMLQuantizationType.Q4_1)
        replace[f"blk.{i}.ffn_up.weight"] = (q41((c[ni:], sc[ni:], zero[ni:], g)), gguf.GGMLQuantizationType.Q4_1)
        replace[f"blk.{i}.ffn_down.weight"] = (q41(codes[(i, "down")]), gguf.GGMLQuantizationType.Q4_1)
    extra = {"output.weight": (q41(codes[("head", None)]), gguf.GGMLQuantizationType.Q4_1)} if ("head", None) in codes else {}
    write(reader, args.out, replace, extra)
    print(f"wrote {args.out}: {len(replace)} Q4_1 matrices" + (" + Q4_1 output.weight" if extra else ""))


def reorder_v(x, dim, nk, nvk, hd):
    """llama.cpp converter (_reorder_v_heads): V heads from grouped-by-K-head to tiled order along dim."""
    shape = list(x.shape)
    new = shape[:dim] + [nk, nvk, hd] + shape[dim + 1:]
    x = x.reshape(*new)
    perm = list(range(len(new)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return x.permute(*perm).contiguous().reshape(*shape)


def redhat_q4_0(args):
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    from safetensors import safe_open
    import json
    reader = gguf.GGUFReader(args.base)
    cfg = json.loads((Path(args.ckpt) / "config.json").read_text())["text_config"]
    nk, nv, dk, dv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    nvk = nv // nk
    by_name = {t.name: t for t in reader.tensors}
    st = safe_open(str(Path(args.ckpt) / "model.safetensors"), "pt")
    replace, checked, n_inexact_groups = {}, 0, 0
    for i, lt in enumerate(cfg["layer_types"]):
        pre = f"model.language_model.layers.{i}"
        if lt == "linear_attention":
            items = [("attn_qkv", "linear_attn.in_proj_qkv", "qkv"), ("attn_gate", "linear_attn.in_proj_z", "rows"),
                     ("ssm_out", "linear_attn.out_proj", "cols")]
        else:
            items = [("attn_q", "self_attn.q_proj", None), ("attn_k", "self_attn.k_proj", None),
                     ("attn_v", "self_attn.v_proj", None), ("attn_output", "self_attn.o_proj", None)]
        items += [("ffn_gate", "mlp.gate_proj", None), ("ffn_up", "mlp.up_proj", None), ("ffn_down", "mlp.down_proj", None)]
        for gname, src, how in items:
            p, s, sh = (st.get_tensor(f"{pre}.{src}.{k}") for k in ("weight_packed", "weight_scale", "weight_shape"))
            out_f, in_f = (int(x) for x in sh)
            q = unpack_from_int32(p, 4, torch.Size((out_f, in_f))).to(torch.int16)          # -8 .. 7
            scale = s.float()                                                                # (out, in // 128)
            g = in_f // scale.shape[1]
            if how == "qkv":
                v0 = 2 * nk * dk
                q = torch.cat([q[:v0], reorder_v(q[v0:], 0, nk, nvk, dv)])
                scale = torch.cat([scale[:v0], reorder_v(scale[v0:], 0, nk, nvk, dv)])
            elif how == "rows":
                q, scale = reorder_v(q, 0, nk, nvk, dv), reorder_v(scale, 0, nk, nvk, dv)
            elif how == "cols":
                assert dv % g == 0, "column reorder must move whole scale groups"
                q = reorder_v(q, 1, nk, nvk, dv)
                scale = reorder_v(scale, 1, nk, nvk, dv // g)
            d32 = scale.repeat_interleave(g // 32, 1)                                        # one scale per 32
            inexact = ~torch.eq(d32.to(torch.float16).float(), d32)                          # FP16-subnormal scales
            n_inexact_groups += int(inexact[:, ::g // 32].sum())
            name = f"blk.{i}.{gname}.weight"
            ref = bf16_tensor(by_name[name])
            deq = (q.float() * d32.repeat_interleave(32, 1)).to(torch.bfloat16)
            ok = torch.eq(deq, ref) | inexact.repeat_interleave(32, 1)    # exact except in groups whose scale FP16 rounds
            if not bool(ok.all()):
                bad = (deq.float() - ref.float())[~ok].abs().max().item()
                raise ValueError(f"{name}: dequantised codes differ from the converter's BF16 tensor (max {bad})")
            checked += 1
            replace[name] = (blocks_q4_0((q + 8).to(torch.uint8), d32), gguf.GGMLQuantizationType.Q4_0)
        print(f"layer {i} ({lt}): {len(items)} matrices checked", flush=True)
    write(reader, args.out, replace, {})
    print(f"wrote {args.out}: {checked} Q4_0 matrices, each checked equal to the checkpoint's q * scale except "
          f"{n_inexact_groups} groups of 128 whose scale is below FP16's normal range and is rounded to FP16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["draft-q4_1", "redhat-q4_0"])
    ap.add_argument("--base", required=True, help="the BF16 GGUF from llama.cpp's converter")
    ap.add_argument("--codes", help="draft-q4_1: saved INT4 draft codes (Qwen3Decoder.save_int4_codes)")
    ap.add_argument("--ckpt", help="redhat-q4_0: the compressed-tensors checkpoint directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    draft_q4_1(args) if args.mode == "draft-q4_1" else redhat_q4_0(args)


if __name__ == "__main__":
    sys.exit(main())
