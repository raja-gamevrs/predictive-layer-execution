"""Radan engine: exact, fast single-user LLM decoding on one CUDA GPU.

The engine behind Radan's local agents. CUDA-graph decode steps with fused small kernels, llama.cpp-identical
Q8_0 weights (Triton W8A16 kernel), and self-speculation with each model's own draft (a 4-bit copy of itself,
or its MTP layer), every token checked by the full model, so the output is the model's own greedy text.
"""

__version__ = "0.1.0"

from radan_engine.engine import Engine, Generation  # noqa: E402,F401

__all__ = ["Engine", "Generation", "__version__"]
