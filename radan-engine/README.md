# Radan engine

Fast, exact LLM decoding for one user on one CUDA GPU. Radan engine is the inference engine behind
Radan, our open-source project for running agents locally. It comes out of
the research in this repository (Predictive Layer Execution, `docs/STORY.md`). The research's first idea
failed, but profiling the decode step led to a set of exact optimizations that are now this engine.

**Exact** means the engine writes the model's own greedy text, token for token, for the weights it runs.
Every speed-up below was checked by re-running the model over the engine's output and counting
"violations": tokens where the engine picked something other than the model's clear first choice.

## What it does

- **CUDA-graph decode step.** One graph per token, fast attention (cuDNN or batched matmuls) and a fused
  q/k-norm + RoPE.
- **Fused small kernels.** Norms with their residual adds, activations, RoPE, Gemma's softcap and Qwen3.8's
  whole recurrent DeltaNet update are compiled into a few Triton kernels instead of thousands of tiny ones.
- **8-bit weights identical to llama.cpp's Q8_0.** Same codes and scales byte for byte, with a Triton
  W8A16 kernel (8-bit weights, 16-bit activations).
- **Self-speculation with the model's own draft.**
  - Qwen3.5 / 3.8: the checkpoint's multi-token-prediction (MTP) layer drafts k tokens, with a 4-bit copy
    of the output layer for the draft only.
  - Qwen3 / Llama: a calibrated 4-bit copy of the model.
  - The full model checks every drafted token in one pass, and the whole round is one CUDA graph.

## Models

| Family | Default weights | Default draft | Measured (RTX 5090, tokens/s, 2k → 128k context) |
|---|---|---|---|
| Qwen3 / Llama (e.g. Qwen3-4B) | Q8_0 | none (opt-in 4-bit self-draft) | 280 → 67 (llama.cpp Q8_0: 240 → 47) |
| Gemma 4 dense (e.g. Gemma 4 12B) | Q8_0 | none | 96 → 80 (llama.cpp Q8_0: 97 → 68) |
| Qwen3.5 / 3.8 (e.g. Qwen3.8-27B INT4) | the checkpoint's packed INT4 | MTP, k = 4 | 176 at 2k, 1.6–2.4× the plain step (llama.cpp's own MTP: 158 at 2k) |

How these numbers were measured: llama.cpp ran on the same weights (byte-identical GGUFs), with a BF16
KV cache in both engines, one prompt per context length and 2 timed runs. Full tables are in
`docs/decode_levers.md` and `results/processed/story/story.json`. How often a draft is accepted depends
on the text, so draft speed-ups vary from prompt to prompt.

## Install

```
pip install ./radan-engine                 # torch, transformers, safetensors, numpy, triton(-windows)
pip install "./radan-engine[packed]"       # + compressed-tensors, for packed INT4 checkpoints (Qwen3.8)
```

## Use

```python
from radan_engine import Engine

eng = Engine.load("path/to/Qwen3.8-27B-INT4")          # a local Hugging Face model directory
out = eng.generate("The history of the Roman Empire begins with", max_new_tokens=128)
print(out.text)
print(out.tokens_per_s, out.accept_rate, out.setting)

out = eng.generate("Write a haiku about rivers.", chat=True)    # the model's chat template
out = eng.generate("...", draft="none")                        # turn the draft off for one call
```

```
python -m radan_engine --model path/to/model --prompt "Hello" --max-new 64
```

`Engine.load(path, weights="auto"|"q8"|"native", draft="auto"|"none"|"int4"|"mtp", k=None,
max_context=4096, draft_codes=None)`. For Qwen3 / Llama, the 4-bit self-draft pays off only with a
calibrated draft: pass `draft="int4", draft_codes=...`, with codes from the research repo's
`scripts/save_draft_codes.py` (GPTQ with a clipping search, groups of 32). A plain rounded copy is
accepted too rarely to be faster. On 8-bit targets the draft helps only up to about 8k context, and only
on text it predicts well.

## Limits

- One CUDA GPU, batch size 1, greedy decoding. Sampling and many users at once are not supported yet.
- Tested on an RTX 5090 (Windows, PyTorch 2.14, CUDA 13) with Qwen3-4B-Base, Gemma 4 12B and
  Qwen3.8-27B-INT4 (Red Hat). Other models of the same families should work but are untested.
- Not supported yet: mixture-of-experts models (e.g. Inkling 2 Mini) and Gemma 4's MoE variants.
- Gemma 4 in BF16 is numerically fragile (its own reference implementation disagrees with itself), so
  Gemma's 4-bit self-draft is not offered.

## License

Apache-2.0.
