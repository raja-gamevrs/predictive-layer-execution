"""The minimal decoder (ple.decode) matches Transformers on a tiny random Qwen3, and speculation plans run."""

import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def tiny():
    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
                      tie_word_embeddings=True, attn_implementation="sdpa")
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).to("cuda", torch.float32).eval()


def test_prefill_and_decode_match_transformers():
    from ple.decode import Qwen3Decoder
    model = tiny()
    ids = torch.randint(0, 128, (20,), device="cuda")
    dec = Qwen3Decoder(model, max_len=32)
    with torch.inference_mode():
        ref = model(input_ids=ids[None]).logits[0]
        got = dec.prefill(ids[:16])
        assert torch.allclose(got, ref[:16], atol=1e-4, rtol=1e-4)
        for j in range(16, 20):
            mask = torch.full((1, 1, 1, 32), float("-inf"), device="cuda")
            mask[..., :j + 1] = 0
            step = dec.forward(ids[j:j + 1], torch.tensor([j], device="cuda"), mask)
            assert torch.allclose(step[0], ref[j], atol=1e-4, rtol=1e-4)


def test_speculation_plans_run_and_change_output():
    from ple.decode import Qwen3Decoder, SpecPlan
    model = tiny()
    ids = torch.randint(0, 128, (12,), device="cuda")
    dec = Qwen3Decoder(model, max_len=32)
    with torch.inference_mode():
        exact = dec.prefill(ids)
        for plan in (SpecPlan("overlap", [0, 1, 2]), SpecPlan("fused", [0, 2])):
            dec.set_plan(plan)
            out = dec.prefill(ids)
            assert torch.isfinite(out).all()
            assert torch.allclose(out[0], exact[0], atol=1e-4)  # position 0 stays exact
            assert not torch.allclose(out[1:], exact[1:], atol=1e-4)


def test_gate_with_zero_threshold_is_exact():
    from e_common import gated_speculation
    model = tiny()
    layers = model.model.layers
    ids = torch.randint(0, 128, (1, 12), device="cuda")
    pred = {l: (lambda h: h * 0.5) for l in range(3)}
    with torch.inference_mode():
        exact = model(input_ids=ids).logits
        with gated_speculation(layers, [0, 1, 2], pred, 64, tau=0.0):
            gated = model(input_ids=ids).logits
        with gated_speculation(layers, [0, 1, 2], pred, 64):
            spec = model(input_ids=ids).logits
    assert torch.allclose(gated, exact)
    assert not torch.allclose(spec, exact)


def test_fast_attention_and_fused_rope_match_reference():
    """L1 / L2 (docs/decode_levers.md): fast attention and fused RoPE change only rounding order."""
    from ple.decode import Qwen3Decoder
    model = tiny().to(torch.bfloat16)
    ids = torch.randint(0, 128, (20,), device="cuda")
    dec = Qwen3Decoder(model, max_len=32)
    with torch.inference_mode():
        def run(attn, rope):
            dec.attn_impl, dec.rope_impl = attn, rope
            pre = dec.prefill(ids[:16]).float()
            steps = []
            for j in range(16, 20):
                mask = torch.full((1, 1, 1, 32), float("-inf"), device="cuda", dtype=torch.bfloat16)
                mask[..., :j + 1] = 0
                steps.append(dec.forward(ids[j:j + 1], torch.tensor([j], device="cuda"), mask)[0].float())
            mask = torch.full((1, 1, 3, 32), float("-inf"), device="cuda", dtype=torch.bfloat16)
            for r in range(3):
                mask[..., r, :17 + r] = 0
            dec.prefill(ids[:16])
            multi = dec.forward(ids[16:19], torch.arange(16, 19, device="cuda"), mask).float()  # verify-style rows
            return pre, torch.stack(steps), multi
        ref = run("sdpa", "ref")
        for attn, rope in (("fast", "ref"), ("sdpa", "fused"), ("fast", "fused")):
            got = run(attn, rope)
            for a, b in zip(ref, got):
                assert (a - b).norm() / a.norm() < 2e-2, (attn, rope)


def test_int4_pack_matches_dequantised_weights():
    from ple.decode import int4_linear, int4_pack
    torch.manual_seed(0)
    w = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(3, 512, device="cuda", dtype=torch.bfloat16)
    packed = int4_pack(w)
    got = int4_linear(x, packed).float()
    ref = x.float() @ w.float().t()
    err = ((got - ref).norm() / ref.norm()).item()
    assert 0.03 < err < 0.15  # 4-bit noise, not a packing error (swapped nibbles give ~1)


def test_int4_clip_and_gptq_lower_output_error():
    """Clipping search and GPTQ (docs/decode_levers.md, follow-up) beat plain round-to-nearest."""
    from ple.decode import int4_gptq, int4_linear, int4_pack
    torch.manual_seed(0)
    w = torch.randn(256, 512, device="cuda") * 0.02
    w[:, :8] *= 8                                     # a few outlier input channels
    mix = torch.randn(512, 512, device="cuda") / 512 ** 0.5 + torch.eye(512, device="cuda")
    x = (torch.randn(2048, 512, device="cuda") @ mix).to(torch.bfloat16)   # correlated inputs
    ref = x.float() @ w.t()

    def err(packed, rows=x):
        return ((int4_linear(rows, packed).float() - rows.float() @ w.t()).norm() / (rows.float() @ w.t()).norm()).item()
    xt = (torch.randn(256, 512, device="cuda") @ mix).to(torch.bfloat16)   # held-out rows
    rtn = err(int4_pack(w.bfloat16(), 128), xt)
    clip = err(int4_pack(w.bfloat16(), 128, clip=True), xt)
    g32 = err(int4_pack(w.bfloat16(), 32), xt)
    gptq = err(int4_gptq(w.bfloat16(), x.float().t() @ x.float(), 128), xt)
    assert ref.isfinite().all()
    assert clip <= rtn and g32 < rtn and gptq < rtn, (rtn, clip, g32, gptq)


def test_llama_blocks_match_transformers():
    """Replication on Llama (docs/decode_levers.md, follow-up D): no q/k norm; the reference path
    matches Transformers, and fast attention + fused RoPE match it up to BF16 rounding."""
    from transformers import LlamaConfig, LlamaForCausalLM
    from ple.decode import Qwen3Decoder
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, attn_implementation="sdpa")
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to("cuda", torch.float32).eval()
    ids = torch.randint(0, 128, (20,), device="cuda")

    def run(dec, attn, rope, dt):
        dec.attn_impl, dec.rope_impl = attn, rope
        out = [dec.prefill(ids[:16]).float()]
        for j in range(16, 20):
            mask = torch.full((1, 1, 1, 32), float("-inf"), device="cuda", dtype=dt)
            mask[..., :j + 1] = 0
            out.append(dec.forward(ids[j:j + 1], torch.tensor([j], device="cuda"), mask).float())
        return torch.cat(out)
    with torch.inference_mode():
        ref = model(input_ids=ids[None]).logits[0]
        dec = Qwen3Decoder(model, max_len=32)
        assert not dec.qk_norm
        assert torch.allclose(run(dec, "sdpa", "ref", torch.float32), ref, atol=1e-4, rtol=1e-4)
        model = model.to(torch.bfloat16)
        dec = Qwen3Decoder(model, max_len=32)
        a, b = run(dec, "sdpa", "ref", torch.bfloat16), run(dec, "fast", "fused", torch.bfloat16)
        assert (a - b).norm() / a.norm() < 2e-2
