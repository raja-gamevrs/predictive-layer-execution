"""The simulated speculation path: full correction reproduces the exact model; no correction does not."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ple.model_loader import decoder_layers
from ple.speculation_policy import attention_input_linears, speculate_attention_inputs


def _model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2)
    return LlamaForCausalLM(cfg).double().eval()


def _exact(w, y_hat, e):
    return y_hat + torch.nn.functional.linear(e.to(w.dtype), w)


def _uncorrected(_w, y_hat, _e):
    return y_hat


def test_finds_q_k_v_only():
    layer = decoder_layers(_model())[1]
    names = {n for n, m in layer.named_modules() if m in attention_input_linears(layer, 32)}
    assert names == {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}


def test_full_correction_matches_exact_and_identity_without_correction_differs():
    model = _model()
    layers = decoder_layers(model)
    ids = torch.randint(0, 97, (2, 7))
    exact = model(input_ids=ids).logits
    identity = {l: (lambda h: h) for l in range(3)}

    with speculate_attention_inputs(layers, [0, 1, 2], identity, lambda l: _exact, 32):
        corrected = model(input_ids=ids).logits
    torch.testing.assert_close(corrected, exact)

    with speculate_attention_inputs(layers, [0, 1, 2], identity, lambda l: _uncorrected, 32):
        approx = model(input_ids=ids).logits
    assert not torch.allclose(approx, exact)
    torch.testing.assert_close(approx[:, 0], exact[:, 0])  # position 0 is kept exact

    torch.testing.assert_close(model(input_ids=ids).logits, exact)  # hooks removed
