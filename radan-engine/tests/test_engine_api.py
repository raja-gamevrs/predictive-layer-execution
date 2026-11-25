"""CPU-only checks of the package surface (GPU behaviour is checked by the research repo's benchmarks)."""
import pytest

import radan_engine
from radan_engine.engine import _family, _is_packed


def test_version_and_exports():
    assert radan_engine.__version__ == "0.1.0"
    assert hasattr(radan_engine, "Engine") and hasattr(radan_engine, "Generation")


@pytest.mark.parametrize("cfg,family", [
    ({"model_type": "qwen3"}, "qwen3"),
    ({"model_type": "llama"}, "qwen3"),
    ({"model_type": "gemma4_unified", "text_config": {"model_type": "gemma4_unified_text"}}, "gemma"),
    ({"model_type": "qwen3_5"}, "qwen35"),
])
def test_family(cfg, family):
    assert _family(cfg) == family


def test_unsupported_family():
    with pytest.raises(ValueError):
        _family({"model_type": "inkling2_loop"})


def test_packed_detection():
    assert _is_packed({"quantization_config": {"quant_method": "compressed-tensors"}})
    assert not _is_packed({"model_type": "qwen3"})
