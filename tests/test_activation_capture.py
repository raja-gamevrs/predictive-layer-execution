import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from ple.activation_capture import capture_residual_stream
from ple.model_loader import attention_input_projection, decoder_layers


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2)
    return LlamaForCausalLM(cfg).eval()


def test_capture_matches_hidden_states_and_preserves_order(tiny_model):
    ids = torch.randint(0, 97, (2, 9))
    layers = decoder_layers(tiny_model)
    states = capture_residual_stream(tiny_model, layers, input_ids=ids)
    ref = tiny_model(input_ids=ids, output_hidden_states=True).hidden_states

    assert len(states) == len(layers) + 1
    for s in states:
        assert s.shape == (2, 9, 32) and s.dtype == torch.float32
    # The last reference state has the final norm applied, so compare all but the last.
    for got, want in zip(states[:-1], ref[:-1]):
        torch.testing.assert_close(got, want)
    torch.testing.assert_close(tiny_model.model.norm(states[-1]), ref[-1])


def test_hooks_are_removed(tiny_model):
    layers = decoder_layers(tiny_model)
    count = lambda: [(len(l._forward_hooks), len(l._forward_pre_hooks)) for l in layers]
    before = count()  # Transformers may register hooks of its own
    capture_residual_stream(tiny_model, layers, input_ids=torch.randint(0, 97, (1, 4)))
    assert count() == before


def test_attention_input_projection_is_q_proj(tiny_model):
    name, proj = attention_input_projection(decoder_layers(tiny_model)[0], 32)
    assert name == "self_attn.q_proj" and proj.in_features == 32
