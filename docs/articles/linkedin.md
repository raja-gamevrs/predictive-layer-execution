# LinkedIn post (draft, not posted)

*Attach: `figures/before_after_bars_2k.png` and `figures/journey_qwen3_4b.png` (optionally `figures/before_after_qwen38_27b.png`).*

---

My idea to make LLMs faster failed. What I learned from the failure made them faster anyway. 🧵

This week I started with "Predictive Layer Execution": guess what the next layer of a language
model will output, start the following layers early, then fix the difference. If it worked, layers
would overlap and you'd get the same answer, faster.

It didn't work:
❌ The best cheap guess was still 18–19% wrong
❌ Fixing the error cost almost as much as computing the layer
❌ Real GPU overlap saved only ~2–4% end to end
❌ Five rescue attempts and a "skip layers" draft all missed their bars, set before any run

So I stopped guessing and profiled where each token's time actually goes. Then I fixed it piece by piece, and checked every time that the output stayed exactly the model's own text:

✅ Faster attention + a fused rotary step: 1.36×
✅ The model's own 4-bit copy as a draft that the full model checks, with the whole round in one CUDA graph: 2.1× on Qwen3-4B, same output
✅ Fusing thousands of tiny GPU kernels into a few: +28% on Qwen3.8-27B, +18% on Gemma 4 12B
✅ Qwen3.8's own multi-token-prediction layer as the draft: 2.4× on Qwen3.8-27B

Then a fair fight with llama.cpp: same quantized weights byte for byte, same KV cache, 2k to 128k context, one RTX 5090:
📈 Qwen3-4B (8-bit): 311 vs 240 tokens/s at 2k, 66 vs 46 at 128k (1.2–1.4× at every length)
📈 Gemma 4 12B (8-bit): was 15–19% behind, now level up to 16k and 1.17× at 128k
📈 Qwen3.8-27B (4-bit), with its own MTP layer as the draft: 176 vs 158 tokens/s at 2k (llama.cpp using its own MTP too), ahead at 5 of 7 lengths
🔎 The Qwen3.8 win came from two small things: a bug (I fed the MTP layer the hidden state before the final norm; llama.cpp's code showed me) and giving only the draft a 4-bit copy of the 2.5 GB output layer

Every result above produces exactly the model's own output, checked token by token.

Lessons:
1. Measure before you invent.
2. Write the pass/fail bar down before you run.
3. Check that "faster" still gives the same answer.

Everything is open, failures included, with every number backed by a result file:
github.com/raja-gamevrs/predictive-layer-execution

#LLM #Inference #GPU #MachineLearning #OpenSource
