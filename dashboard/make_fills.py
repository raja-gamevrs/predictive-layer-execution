"""Chapter texts for the dashboard (fills.json), numbers bound to the story dataset where it has them."""
import json
from pathlib import Path

D = Path(__file__).parent


def n(src, field, where, fmt=None):
    f = f' data-fmt="{fmt}"' if fmt else ""
    return f'<span data-draw="num" class="dash-skeleton" data-source="{src}" data-field="{field}" data-where="{where}"{f}>…</span>'


def L(key, ctx, field):
    return n("st_land", field, f"id={key}|{ctx}")


fills = {
    "MTP_TEXT": f"""
    <p class="ple-plain">Qwen3.8 is already 4-bit, so a 4-bit copy of itself is no cheaper. But it ships with a small extra layer trained to guess the next-next token: multi-token prediction (MTP). One MTP step costs one layer plus the output head instead of 64 layers. We use it as the draft: it guesses k tokens, the full model checks them in one pass, and the whole round (guess, check, accept, roll the recurrent DeltaNet state back to the last kept token) is one CUDA graph.</p>
    <p class="ple-plain"><b>A bug found by comparing with llama.cpp.</b> Our first version fed the MTP layer the model's hidden state <i>before</i> its final normalization; llama.cpp's MTP on the same file was accepted more often (86% against 79% at 2k) and ran faster. Its code feeds the state <i>after</i> the norm, as the model was trained. We fixed it, reran every context, and kept the first run on file.</p>
    <p class="ple-plain">At 2k context Qwen3.8 now writes {L('qwen38_27b', '2k', 'draft')} tokens/s with its MTP draft against {L('qwen38_27b', '2k', 'after')} without it, with the model's own output (0 violations for every k tried). llama.cpp's own MTP on the same weights reaches {L('qwen38_27b', '2k', 'llama_spec')}. With only the bug fixed we were at 150, a little behind: the drafts were accepted equally often in both engines, so llama.cpp's round was cheaper. The next step closed that. Hover the Qwen3.8 chart in chapter 12 for every context.</p>
    <p class="ple-plain"><b>The last step: a 4-bit output layer for the draft only.</b> Every MTP guess reads Qwen3.8's whole output layer (2.5 GB in 16-bit). We gave the draft a 4-bit copy (0.6 GB); the full model still checks every token with the real layer, so the output does not change (0 violations), and the draft is right just as often. Written down as follow-up H2 before it ran. It made the MTP round 8–17% cheaper and put us ahead of llama.cpp's own MTP at 2k, 4k, 16k, 32k and 128k (best k), 2–6% behind at 8k and 64k. The numbers above use it (k = 4).</p>
    <p class="ple-test">How often a draft is right depends heavily on the text (87% on the 2k prompt, about 50% on the 8k one), in both engines. One prompt per context length. The 128k run spilled about 0.2 GB of GPU memory into system RAM and may read low.</p>""",
    "INKLING_TEXT": """
    <p class="ple-plain">Thinking Machines' Inkling 2 Mini (released 20 November) is a different design: a mixture of experts (128 small expert networks, 8 used per token), a block of layers that runs three times in a loop, mostly sliding-window attention, and 4-bit NVFP4 expert weights. Our decoders do not run it; porting it is a project of its own. So we measured it in llama.cpp only (b12720, converted with llama.cpp's own converter, quantized to Q4_K_M, the most common way people run GGUF models). The text it writes is coherent, but there is no exactness check without a reference implementation on this GPU.</p>
    <p class="ple-test">At 128k, llama.cpp's default prompt batch needed 12.9 GB of scratch memory; with a smaller batch (<code>-ub 128</code>) it fits. That changes prompt reading only, not the writing speed.</p>""",
    "LANDED_TEXT": f"""
    <ul class="ple-list">
      <li><b>Qwen3-4B, 8-bit:</b> faster than llama.cpp at every length, {L('qwen3_4b', '2k', 'vs_llama')}× at 2k (with its own 4-bit draft) and {L('qwen3_4b', '128k', 'vs_llama')}× at 128k (the plain fused step; the draft stops paying past 8k).</li>
      <li><b>Gemma 4 12B, 8-bit:</b> level with llama.cpp up to 16k, ahead from 32k, {L('gemma4_12b', '128k', 'vs_llama')}× at 128k. Before the kernel fusion it trailed by 15–19%. Gemma's 4-bit draft is shown but not counted: it changes 1 token in 128.</li>
      <li><b>Qwen3.8-27B, 4-bit:</b> the fused step went from 23% behind llama.cpp's plain decoding to level at 128k. With its own MTP layer as the draft (and a 4-bit draft head) it is 1.6–2.4× faster again, and ahead of llama.cpp's own MTP at most lengths.</li>
    </ul>""",
}
(D / "fills.json").write_text(json.dumps(fills, indent=1), encoding="utf-8")
print("ok")
