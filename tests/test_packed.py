import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

from ple.quant.packed import PackedLinear


def test_int4_packed_linear_matches_dequantised_reference():
    torch.manual_seed(0)
    out_f, in_f, group = 8, 256, 128
    q = torch.randint(-8, 8, (out_f, in_f), dtype=torch.int8)
    scale = (torch.rand(out_f, in_f // group) + 0.5).to(torch.bfloat16)
    lin = PackedLinear("int4_w4a16", out_f, in_f, {"weight_packed": pack_to_int32(q, 4), "weight_scale": scale},
                       None, group)
    ref = (q.float() * scale.float().repeat_interleave(group, dim=1)).to(torch.bfloat16)
    torch.testing.assert_close(lin.dequantized_weight(), ref)
    x = torch.randn(3, in_f, dtype=torch.bfloat16)
    torch.testing.assert_close(lin(x), torch.nn.functional.linear(x, ref))


def test_int4_is_linear_in_input_and_nvfp4_w4a4_is_not():
    torch.manual_seed(0)
    q = torch.randint(-8, 8, (8, 128), dtype=torch.int8)
    lin = PackedLinear("int4_w4a16", 8, 128, {"weight_packed": pack_to_int32(q, 4),
                                               "weight_scale": torch.ones(8, 1, dtype=torch.bfloat16)}, None, 128)
    a, b = torch.randn(4, 128, dtype=torch.float64), torch.randn(4, 128, dtype=torch.float64)
    w = lin.dequantized_weight().double()
    lin_d = lambda x: torch.nn.functional.linear(lin.quantize_input(x), w)
    torch.testing.assert_close(lin_d(a + b), lin_d(a) + lin_d(b))
    packed = torch.zeros(8, 64, dtype=torch.uint8)
    w4a4 = PackedLinear("nvfp4_w4a4", 8, 128, {"weight": packed, "weight_scale": torch.ones(8, 8).to(torch.float8_e4m3fn),
                                                "weight_scale_2": torch.tensor(1.0), "input_scale": torch.tensor(0.01)}, None)
    xa, xb = torch.randn(4, 128), torch.randn(4, 128) * 0.1
    assert not torch.allclose(w4a4.quantize_input(xa + xb), w4a4.quantize_input(xa) + w4a4.quantize_input(xb))
