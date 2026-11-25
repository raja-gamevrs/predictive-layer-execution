"""Dashboard v14: the whole project as one chronological story in plain language.
Keeps v13's styles and every calc / draw (unchanged), adds story.js, and replaces the HTML body.
Usage: python build_v14.py  -> index_v14.html, index_v14_doc.json"""
import html as H
import json
import re
from pathlib import Path

D = Path(__file__).parent
v13 = (D / "index_v13.html").read_text(encoding="utf-8")
style = v13[:v13.index("</style>") + len("</style>")]
js = v13[v13.index("<script>") + 8:v13.rindex("</script>")]
body13 = v13[v13.index("</style>"):v13.index("<script>")]
story_js = (D / "story.js").read_text(encoding="utf-8")


def section(sid):
    i = body13.index(f'id="{sid}"')
    i = body13.rindex("<section", 0, i)
    j = body13.index("</section>", i) + len("</section>")
    return body13[i:j]


def inner(sid, start_pat):
    """Part of an old section from the first match of start_pat to the section's end (keeps its charts / tables)."""
    s = section(sid)
    k = s.index(start_pat)
    return s[k:s.rindex("</section>")]


def n(src, field=None, where=None, fmt=None):
    a = f'<span data-draw="num" class="dash-skeleton" data-source="{src}"'
    if field:
        a += f' data-field="{field}"'
    if where:
        a += f' data-where="{H.escape(where)}"'
    if fmt:
        a += f' data-fmt="{fmt}"'
    return a + ">…</span>"


def L(key, ctx, field):   # a number from the landing table
    return n("st_land", field, f"id={key}|{ctx}")


EXTRA_CSS = """
<style>
.ple-tiles { display: grid; gap: var(--cds-gap-md); grid-template-columns: 1fr; }
@container (min-width: 600px) { .ple-tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
.ple-tile { background: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: var(--cds-pad-lg); display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.ple-tile-k { font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); }
.ple-tile-v { font-family: var(--font-anthropic-serif); font-size: var(--cds-font-size-title); line-height: 1.1; font-variant-numeric: tabular-nums; }
.ple-tile-s { font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); line-height: 1.45; }
.ple-chapter { font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); letter-spacing: .06em; text-transform: uppercase; }
.ple-toc { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: var(--cds-font-size-caption); }
.ple-toc a { color: var(--color-fg-muted); text-decoration: none; border-bottom: 1px solid var(--color-border-line); }
.ple-toc a:hover { color: var(--color-fg); }
.ple-plain { max-width: 72ch; line-height: 1.6; }
.ple-plain + .ple-plain { margin-top: -4px; }
.ple-list { max-width: 72ch; line-height: 1.6; margin: 0; padding-left: 1.2em; }
.ple-list li + li { margin-top: 4px; }
.ple-controls { display: flex; flex-wrap: wrap; gap: var(--cds-gap-sm) var(--cds-gap-md); align-items: center; font-size: var(--cds-font-size-caption); color: var(--color-fg-muted); }
.ple-controls select { font: inherit; color: var(--color-fg); background: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: 6px; padding: 2px 6px; }
.ple-word { font-weight: var(--cds-font-weight-medium); }
</style>
"""

LEGEND = """<div class="ple-legend" aria-hidden="true">
      <span><i class="ple-sw" data-st-swatch="before"></i>Ours, before (dashed line)</span>
      <span><i class="ple-sw" data-st-swatch="after"></i>Ours, after (fused step)</span>
      <span><i class="ple-sw" data-st-swatch="draft"></i>Ours, after + draft</span>
      <span><i class="ple-sw" data-st-swatch="llama"></i>llama.cpp, same weights</span>
      <span><i class="ple-sw" data-st-swatch="llama_spec"></i>llama.cpp + its own draft</span>
    </div>"""

CTX_SELECT = """<label>Context <select class="ple-st-ctx">""" + "".join(
    f'<option value="{c}">{c // 1024}k tokens</option>' for c in (2048, 4096, 8192, 16384, 32768, 65536, 131072)) + "</select></label>"
MODEL_SELECT = """<label>Model <select class="ple-st-model">
      <option value="qwen3_4b">Qwen3-4B, 8-bit (Q8_0)</option>
      <option value="gemma4_12b">Gemma 4 12B, 8-bit (Q8_0)</option>
      <option value="qwen38_27b">Qwen3.8-27B, 4-bit (INT4)</option></select></label>"""

TOC = [("start", "1. The idea"), ("found", "2. What we found"), ("precision", "3. A useful side finding"),
       ("turn", "4. The turn"), ("checks", "5. Checking it"), ("bigger", "6. Bigger models"), ("long", "7. Long context, fair fight"),
       ("quant", "8. 8-bit and 2-bit"), ("fusion", "9. Tiny kernels"), ("mtp", "10. Qwen3.8's own draft"),
       ("inkling", "11. Inkling 2 Mini"), ("landed", "12. Where we landed"), ("lessons", "13. What helped"), ("prior-art", "Prior work")]

body = f"""
<div class="ple">
  <header class="ple-head" id="top">
    <p class="ple-eyebrow">Research log · 23–25 November 2026 · one RTX 5090</p>
    <h1 class="ple-title">Making LLMs write faster: what failed, what worked</h1>
    <p class="ple-lede">We set out to speed up language models by guessing each layer's output ahead of time. That idea failed every test. Measuring where the time really goes led somewhere better: a set of exact changes to the decoding step that make today's popular open models write faster, with the same output as the model itself, and on equal weights faster than or level with llama.cpp.</p>
    <p class="ple-setup">Everything below is in the order it happened. Every number is read from a result file in the repository (raja-gamevrs/predictive-layer-execution, results/processed/). <span class="ple-word">Exact</span> means the output is the model's own greedy text: we re-run the model over what the engine wrote and count "violations", tokens where the engine picked something other than the model's clear first choice. <span class="ple-word">Tokens per second</span> is the steady writing speed, prompt reading not counted. One user at a time (batch 1), greedy decoding, Windows.</p>
    <nav class="ple-toc">{''.join(f'<a href="#{a}">{t}</a>' for a, t in TOC)}</nav>
  </header>

  <section class="ple-section" id="summary">
    <h2>In one minute: before and after, against llama.cpp</h2>
    <p class="ple-sub">Same quantized weights in both engines (byte for byte), same KV cache precision, same prompt. "Before" is our exact decoder when we first matched llama.cpp; "after" is where it is now.</p>
    <div class="ple-tiles">
      <div class="ple-tile"><span class="ple-tile-k">Qwen3-4B, 8-bit, 2k context</span><span class="ple-tile-v">{L('qwen3_4b', '2k', 'ours')} <span class="ple-tile-k">vs</span> {L('qwen3_4b', '2k', 'theirs')}</span><span class="ple-tile-s">tokens/s, ours against llama.cpp ({L('qwen3_4b', '2k', 'vs_llama')}×). Was {L('qwen3_4b', '2k', 'before')} before.</span></div>
      <div class="ple-tile"><span class="ple-tile-k">Gemma 4 12B, 8-bit, 2k context</span><span class="ple-tile-v">{L('gemma4_12b', '2k', 'ours')} <span class="ple-tile-k">vs</span> {L('gemma4_12b', '2k', 'theirs')}</span><span class="ple-tile-s">tokens/s ({L('gemma4_12b', '2k', 'vs_llama')}×). Was {L("gemma4_12b", "2k", "before")} before.</span></div>
      <div class="ple-tile"><span class="ple-tile-k">Qwen3.8-27B, 4-bit, 2k context</span><span class="ple-tile-v">{L('qwen38_27b', '2k', 'ours')} <span class="ple-tile-k">vs</span> {L('qwen38_27b', '2k', 'theirs')}</span><span class="ple-tile-s">tokens/s ({L('qwen38_27b', '2k', 'vs_llama')}×). Was {L("qwen38_27b", "2k", "before")} before.</span></div>
    </div>
    <div class="ple-controls">{CTX_SELECT}</div>
    {LEGEND}
    <div class="ple-chart dash-skeleton" data-draw="st-bars" data-source="st_land"></div>
  </section>

  <section class="ple-section" id="start">
    <p class="ple-chapter">Chapter 1 · 23 November</p>
    <h2>Where we started: guess the next layer</h2>
    <p class="ple-plain">A language model is a tall stack of layers, and each layer waits for the one below it. Our idea, "Predictive Layer Execution" (PLE): <b>guess</b> what a layer will output, <b>start</b> the next layers early on the guess, then <b>fix</b> the small difference when the real output arrives. If the guess is good and the fix is cheap, layers overlap and the same answer comes out sooner.</p>
    <p class="ple-plain">We wrote six hypotheses down before running anything:</p>
    <div class="ple-table-wrap"><table class="ple-table">
      <thead><tr><th>Hypothesis</th><th>In plain words</th><th>Outcome</th></tr></thead>
      <tbody>
        <tr><td>H1</td><td class="ple-wrap">A layer's change is predictable from what came before</td><td><span class="ple-pill weak">Only roughly</span></td></tr>
        <tr><td>H2</td><td class="ple-wrap">The guessing error has a simple shape we can exploit</td><td><span class="ple-pill fail">No</span></td></tr>
        <tr><td>H3</td><td class="ple-wrap">Fixing the error is cheaper than computing the layer</td><td><span class="ple-pill fail">No</span></td></tr>
        <tr><td>H4</td><td class="ple-wrap">Running the guess in parallel shortens each token</td><td><span class="ple-pill weak">A little</span></td></tr>
        <tr><td>H5</td><td class="ple-wrap">It works better on reading the prompt than on writing</td><td><span class="ple-pill fail">No</span></td></tr>
        <tr><td>H6</td><td class="ple-wrap">A smart policy (when and where to guess) beats always guessing</td><td><span class="ple-pill fail">Not enough</span></td></tr>
      </tbody></table></div>
    <p class="ple-test">Models: Qwen3-4B, Qwen3.5-4B, Ornith-1.5-9B, Qwen3.8-27B. Text: WikiText-2, held-out split untouched.</p>
  </section>

  <section class="ple-section" id="found">
    <p class="ple-chapter">Chapter 2 · 23–24 November</p>
    <h2>What we found: guessing layers does not pay</h2>
    <ul class="ple-list">
      <li><b>Guesses are rough.</b> The best cheap guess (a linear map) is still {n('h1', 'linear', 'model=Qwen3-4B', 'pct')} wrong on Qwen3-4B. Guessing "nothing changes" is {n('h1', 'identity', 'model=Qwen3-4B', 'pct')} wrong; smarter extrapolation is worse ({n('h1', 'extrapolation', 'model=Qwen3-4B', 'pct')}).</li>
      <li><b>Fixing is not cheap.</b> Spending a quarter of a layer's cost on the worst channels still leaves {n('econ', 'err', 'model=Qwen3-4B&method=topk_25pct', 'pct')} error. The error is spread everywhere.</li>
      <li><b>Skipping the fix hurts quality.</b> Guess every layer with the linear map and perplexity goes from {n('all_layers', 'exact_ppl', 'variant_id=q3-bf16')} to {n('all_layers', 'linear_ppl', 'variant_id=q3-bf16')} (Qwen3-4B); with the naive guess, to {n('all_layers', 'identity_ppl', 'variant_id=q3-bf16')}.</li>
      <li><b>The GPU barely overlaps the work.</b> For one token at a time the most a guessed layer could save is {n('ov', 'ceiling', 'tokens=1', 'pct')}; we measured {n('ov', 'identity', 'tokens=1', 'spct')}, and nothing for long inputs ({n('ov', 'identity', 'tokens=2048', 'spct')} at 2,048 tokens).</li>
      <li><b>Removing parts outright</b> (Phase II): within a 1% perplexity budget only {n('phase2_best', None, None, 'pct')} of the compute can go. The goal was 10%.</li>
      <li><b>Five rescue attempts (E1–E5)</b>: the best decode speed-up was {n('e1_sum', 'hi', 'variant=overlap_linear_all_fp8', 'spct')}, and only by guessing every layer in FP8, which breaks quality. A per-token gate, LoRA "healing" (stops at {n('e3b_sum', 'bar_multiple', 'rank=16')}× the quality bar) and a trained predictor ({n('e4_r', 'ratio', 'key=Qwen3-4B,256,mlp_1024')}× the error of plain ridge regression) each fell short.</li>
      <li><b>Skipping layers to make a draft</b> keeps the output exact, but a draft with 26% of layers skipped still costs {n('ss_cost', 'draft_pct', 'set=atlas26', 'pct')} of a full step, so the best setting runs at {n('ss_pick', 'sp_hold', None, 'x')}: slower than not trying.</li>
    </ul>
    <p class="ple-plain"><b>Lesson:</b> modern LLM layers are not redundant enough to guess cheaply, and at one user at a time the GPU cannot hide the extra work.</p>
    <div class="ple-legend" aria-hidden="true">
      <span><i class="ple-sw" data-swatch="ceiling"></i>Most possible</span>
      <span><i class="ple-sw" data-swatch="identity"></i>Identity guess</span>
      <span><i class="ple-sw" data-swatch="linear"></i>Linear guess</span>
      <span><i class="ple-sw" data-swatch="fused"></i>Fused QKV</span>
    </div>
    <div class="ple-chart dash-skeleton" id="overlap-chart" data-draw="overlap-chart" data-source="ov"></div>
    <p class="ple-test">Saving per guessed layer (Qwen3-4B, block 26 → 27) by input length, idle-GPU rerun, CUDA graphs.</p>
  </section>

  <section class="ple-section" id="precision">
    <p class="ple-chapter">Chapter 3 · 23 November</p>
    <h2>A useful side finding: what stays exact under quantization</h2>
    <p class="ple-plain">When we fixed a guess <i>exactly</i>, the answer stayed exact only if the <b>weights</b> were quantized and the <b>activations</b> stayed 16-bit (INT8, NVFP4 W4A16, Red Hat's Qwen3.8 INT4). Rounding activations too (FP8 W8A8, NVFP4 W4A4) broke it. That is why our engine later keeps activations in BF16.</p>
    {inner('precision', '<div class="ple-legend"')}
  </section>

  <section class="ple-section" id="turn">
    <p class="ple-chapter">Chapter 4 · 24 November</p>
    <h2>The turn: measure where a token's time goes, then fix it piece by piece</h2>
    <p class="ple-plain">Our original decoder wrote a Qwen3-4B token in {n('lv_sum', 'orig_ms', 'phase=decode&variant=L1')} ms ({n('lv2_raw', 'q_tps', 'key=L0')} tokens/s). We profiled the step and set a speed bar for each fix before running it. Every fix had to keep the output exact.</p>
    <div class="ple-table-wrap"><table class="ple-table">
      <thead><tr><th>Change</th><th class="n">Speed-up</th><th class="n">Tokens/s</th><th>Same output?</th></tr></thead>
      <tbody>
        <tr><td class="ple-wrap">L1: fuse the q/k norm with the rotary position step</td><td class="n">{n('lv_sum', 'speedup', 'phase=decode&variant=L1', 'x')}</td><td class="n">{n('lv2_raw', 'q_tps', 'key=L1')}</td><td><span class="ple-pill ok">Yes</span></td></tr>
        <tr><td class="ple-wrap">L2: faster attention kernels</td><td class="n">{n('lv_sum', 'speedup', 'phase=decode&variant=L2', 'x')}</td><td class="n">{n('lv2_raw', 'q_tps', 'key=L2')}</td><td><span class="ple-pill ok">Yes</span></td></tr>
        <tr><td class="ple-wrap">L1 + L2</td><td class="n">{n('lv_sum', 'speedup', 'phase=decode&variant=L12', 'x')}</td><td class="n">{n('lv2_raw', 'q_tps', 'key=L12')}</td><td><span class="ple-pill ok">Yes</span></td></tr>
        <tr><td class="ple-wrap">L6: just run a 4-bit copy of the model</td><td class="n">{n('lv_sum', 'speedup', 'phase=decode&variant=L6', 'x')}</td><td class="n">{n('lv2_raw', 'q_tps', 'key=L6')}</td><td><span class="ple-pill fail">No, +16–19% perplexity</span></td></tr>
        <tr><td class="ple-wrap"><b>L7b: the 4-bit copy as a draft</b>; the full model checks 6 guessed tokens at once and keeps the ones it agrees with</td><td class="n">{n('lv2_pick', 'orig_hold', 'id=qwen_l7b', 'x')}</td><td class="n">{n('lv2_raw', 'q_tps', 'key=L7b')}</td><td><span class="ple-pill ok">Yes</span></td></tr>
      </tbody></table></div>
    <p class="ple-plain">L7b is speculative decoding, a known technique, with the model's own 4-bit copy as the draft. Two things made it work: a <b>good draft</b> (GPTQ with a clipping search in groups of 32, accepted {n('lv2_pick', 'acc', 'id=qwen_l7b', 'pct')} of the time) and a <b>cheap round</b> (drafting, checking and accepting all inside one CUDA graph, with no trips back to the CPU).</p>
  </section>

  <section class="ple-section" id="checks">
    <p class="ple-chapter">Chapter 5 · 24 November</p>
    <h2>Checking it: a rerun, a second model, and two other engines</h2>
    <ul class="ple-list">
      <li><b>Fresh-process rerun:</b> every speed-up within 0.5%.</li>
      <li><b>TinyLlama-1.1B:</b> L1 + L2 reach {n('lv2_raw', 't_tps', 'key=L12')} tokens/s (from {n('lv2_raw', 't_tps', 'key=L0')}), the draft {n('lv2_raw', 't_tps', 'key=L7b')}.</li>
      <li><b>Hugging Face Transformers</b> <code>generate</code>: {n('lv2_raw', 'q_tps', 'key=HFd')} tokens/s on Qwen3-4B, about 5× slower than L1 + L2 and 7.7× slower than the draft, same text.</li>
      <li><b>llama.cpp</b> (BF16, same prompts): plain decoding is level ({n('lv2_raw', 'q_tps', 'key=LCb')} against our {n('lv2_raw', 'q_tps', 'key=HF12')}). With its own draft llama.cpp reaches {n('lv2_raw', 'q_tps', 'key=LCs')}; ours reaches {n('lv2_raw', 'q_tps', 'key=L7b')}. The difference is the cost of a round: ours is one GPU graph.</li>
    </ul>
  </section>

  <section class="ple-section" id="bigger">
    <p class="ple-chapter">Chapter 6 · 24 November</p>
    <h2>Bigger, newer models: Gemma 4 12B and Qwen3.8-27B</h2>
    <ul class="ple-list">
      <li><b>Gemma 4 12B.</b> We wrote a decoder for it. Gemma in BF16 is fragile: Transformers disagrees with <i>itself</i> {n('gem_gate', 'control', 'run=separate')} times, so its correctness gate became "no worse than Transformers" (decided and written down before any timing). Exact levers: {n('gem_speed', 'tps', 'key=L0')} → {n('gem_speed', 'tps', 'key=L12')} tokens/s; llama.cpp BF16 {n('gem_speed', 'tps', 'key=LCbf16')}.</li>
      <li><b>Qwen3.8-27B (4-bit).</b> A hybrid model: most of its layers are Gated DeltaNet, a recurrent layer, not attention. Our decoder passes its gate (KL 0.0011, top-1 99.4%). It is already 4-bit, so a 4-bit copy is no cheaper as a draft; its own multi-token-prediction (MTP) layer is the natural draft (chapter 10).</li>
    </ul>
  </section>

  <section class="ple-section" id="long">
    <p class="ple-chapter">Chapter 7 · 24–25 November</p>
    <h2>Real-world lengths, and a fair fight with llama.cpp</h2>
    <p class="ple-plain">Real use means quantized models and long conversations, not 16-bit weights and short prompts. So we made the comparison fair:</p>
    <ul class="ple-list">
      <li><b>Same weights, byte for byte.</b> We re-implemented llama.cpp's 8-bit format (Q8_0; every byte checked against <code>llama-quantize</code>) and wrote our 4-bit draft codes and Red Hat's Qwen3.8 INT4 codes into GGUF files, so llama.cpp runs exactly our numbers.</li>
      <li><b>Same KV cache</b> (BF16) in both; YaRN for Qwen3-4B above 32k in both.</li>
      <li><b>7 context lengths</b>, 2k to 128k tokens, steady speed only, 2 timed runs each.</li>
    </ul>
    <p class="ple-plain">In BF16 (first round) our Qwen3-4B was faster at every length and up to {n('tps', 'ratio_best', 'key=qwen,2048', 'x')} with the draft; Gemma and Qwen3.8 trailed llama.cpp.</p>
    <div class="ple-table-wrap">
      <table class="ple-table" id="tps-qwen-table">
        <thead><tr><th>Context</th><th class="n">Ours, L1 + L2</th><th class="n">Ours, INT4 self-draft</th><th class="n">llama.cpp</th><th class="n">llama.cpp + same draft</th><th class="n">Ours / llama.cpp, plain</th><th class="n">Best / best</th></tr></thead>
        <tbody data-draw="tps-qwen-table"></tbody>
      </table>
    </div>
    <p class="ple-test">Qwen3-4B, BF16 weights in both engines.</p>
  </section>

  <section class="ple-section" id="quant">
    <p class="ple-chapter">Chapter 8 · 25 November</p>
    <h2>8-bit models, and why 2-bit drafts are a dead end</h2>
    <p class="ple-plain">Most people run quantized models, so the target became 8-bit (Q8_0) for Qwen3-4B and Gemma 4 12B, with our own 8-bit kernel (Triton). The 8-bit Qwen3-4B runs about 1.6× faster than BF16. But at 8 bits the 4-bit draft helps less: a draft step costs about two thirds of a target step, because ~0.9 ms of every step is thousands of tiny kernels that do not shrink with the weights. That pointed to chapter 9.</p>
    <p class="ple-plain">Could a 2-bit draft be cheap enough? We tried it in llama.cpp first. It needs to be right about 75% of the time; it was not:</p>
    <div class="ple-table-wrap"><table class="ple-table">
      <thead><tr><th>Model (8-bit target)</th><th class="n">Draft tokens per round</th><th class="n">2-bit draft accepted</th><th class="n">Tokens/s</th></tr></thead>
      <tbody data-draw="st-q2-table"></tbody></table></div>
    <p class="ple-test">llama.cpp, pure Q2_K draft of the same model, 2k context, three prompts. No 2-bit kernel was built.</p>
  </section>

  <section class="ple-section" id="fusion">
    <p class="ple-chapter">Chapter 9 · 25 November</p>
    <h2>Fusing thousands of tiny kernels</h2>
    <p class="ple-plain">Each token launched 655 small GPU kernels on Qwen3-4B, 3,039 on Gemma 4 12B and 4,940 on Qwen3.8. Compiling the whole step with <code>torch.compile</code> was <i>slower</i>: it turned every KV-cache write into a copy of the whole cache. Compiling only the small pure pieces (norm + residual, q/k norm + rotary step, activations, Gemma's softcap, Qwen3.8's whole recurrent DeltaNet update) worked. Gemma went from 3,039 to 1,205 kernels per token.</p>
    <div class="ple-table-wrap"><table class="ple-table">
      <thead><tr><th>2k context</th><th class="n">Before</th><th class="n">After</th><th class="n">After + draft</th><th class="n">Gain</th></tr></thead>
      <tbody>
        <tr><td>Qwen3-4B, 8-bit</td><td class="n">{L('qwen3_4b', '2k', 'before')}</td><td class="n">{L('qwen3_4b', '2k', 'after')}</td><td class="n">{L('qwen3_4b', '2k', 'draft')}</td><td class="n">{n('st_land', 'gain_pct', 'id=qwen3_4b|2k', 'spct')}</td></tr>
        <tr><td>Gemma 4 12B, 8-bit</td><td class="n">{L('gemma4_12b', '2k', 'before')}</td><td class="n">{L('gemma4_12b', '2k', 'after')}</td><td class="n">{L('gemma4_12b', '2k', 'draft')}</td><td class="n">{n('st_land', 'gain_pct', 'id=gemma4_12b|2k', 'spct')}</td></tr>
        <tr><td>Qwen3.8-27B, 4-bit</td><td class="n">{L('qwen38_27b', '2k', 'before')}</td><td class="n">{L('qwen38_27b', '2k', 'after')}</td><td class="n">{L('qwen38_27b', '2k', 'draft')}</td><td class="n">{n('st_land', 'gain_pct', 'id=qwen38_27b|2k', 'spct')}</td></tr>
      </tbody></table></div>
    <p class="ple-test">Tokens per second. Gain = our best setting now against before. The output checks are in chapter 12.</p>
  </section>

  <section class="ple-section" id="mtp">
    <p class="ple-chapter">Chapter 10 · 25 November</p>
    <h2>Qwen3.8's own draft: its multi-token-prediction layer</h2>
    <!-- MTP_TEXT -->
  </section>

  <section class="ple-section" id="inkling">
    <p class="ple-chapter">Chapter 11 · 25 November</p>
    <h2>Inkling 2 Mini</h2>
    <!-- INKLING_TEXT -->
    <div class="ple-table-wrap"><table class="ple-table">
      <thead><tr><th>llama.cpp setting</th><th class="n">Context</th><th class="n">Tokens/s</th></tr></thead>
      <tbody data-draw="st-inkling-table"></tbody></table></div>
  </section>

  <section class="ple-section" id="landed">
    <p class="ple-chapter">Chapter 12 · where we landed</p>
    <h2>Before and after, 2k to 128k context, against llama.cpp</h2>
    <p class="ple-plain">Same weights in both engines. Hover a context to see every number; the draft's acceptance rate is in the tooltip.</p>
    <div class="ple-controls">{MODEL_SELECT}</div>
    {LEGEND}
    <div class="ple-chart dash-skeleton" data-draw="st-lines" data-source="st_tps" data-model="qwen3_4b" data-follow=""></div>
    <!-- LANDED_TEXT -->
    <div class="ple-table-wrap"><table class="ple-table">
      <thead><tr><th>Model</th><th class="n">Context</th><th class="n">Ours before</th><th class="n">Ours after</th><th class="n">After + draft</th><th class="n">llama.cpp</th><th class="n">llama.cpp + draft</th><th class="n">Best / best</th></tr></thead>
      <tbody data-draw="st-land-table"></tbody></table></div>
    <p class="ple-test">Tokens per second; "–" = not run or does not fit in 32 GB. Best / best: our faster exact setting against llama.cpp's faster one (green ≥ 1.05×, amber within 5%, red behind).</p>
    <h2>Is the output still the model's own?</h2>
    <div class="ple-table-wrap"><table class="ple-table">
      <thead><tr><th>Model</th><th>Setting</th><th class="n">Checks</th><th class="n">Violations</th><th class="n">Near-ties</th></tr></thead>
      <tbody data-draw="st-check-table"></tbody></table></div>
    <p class="ple-test">Each check is one prompt at 2k context, 128 tokens (256 for the MTP drafts), checked against the same model re-run over the output. Near-ties (the top two tokens almost equal) may flip with any change in arithmetic order and are allowed. Gemma's draft shows 1 violation per 128 tokens; Gemma's own reference implementation disagrees with itself 31 times on the gate text.</p>
  </section>

  <section class="ple-section" id="lessons">
    <p class="ple-chapter">Chapter 13</p>
    <h2>What helped, what did not</h2>
    <div class="ple-grid">
      <div class="ple-card"><h3>Helped (exact, measured)</h3><ul class="ple-list">
        <li>A CUDA-graph decode step with fast attention and a fused rotary step.</li>
        <li>Fusing the many tiny kernels: biggest on models with many small operations (Qwen3.8, Gemma).</li>
        <li>The model's own 4-bit copy as a draft, with the whole round in one GPU graph: big on 16-bit targets, smaller on 8-bit, short context only.</li>
        <li>Matching llama.cpp's 8-bit format exactly: about 1.6× over BF16, and a fair comparison.</li>
      </ul></div>
      <div class="ple-card"><h3>Did not help</h3><ul class="ple-list">
        <li>Guessing layer outputs (the original idea), in every form tried.</li>
        <li>Skipping layers as a draft.</li>
        <li>2-bit drafts.</li>
        <li>Compiling the whole step at once.</li>
        <li>Self-drafts at long context with an 8-bit target.</li>
      </ul></div>
      <div class="ple-card"><h3>Limits</h3><ul class="ple-list">
        <li>One GPU (RTX 5090), Windows, one user at a time, greedy decoding, base models.</li>
        <li>llama.cpp runs everywhere; ours is a research decoder for a few model families.</li>
        <li>One prompt per context length for the long runs, 2 timed runs each.</li>
      </ul></div>
    </div>
  </section>

  {section('prior-art')}
</div>
"""

fills = json.loads((D / "fills.json").read_text(encoding="utf-8")) if (D / "fills.json").exists() else {}
for k in ("MTP_TEXT", "INKLING_TEXT", "LANDED_TEXT"):
    body = body.replace(f"<!-- {k} -->", fills.get(k, '<p class="ple-plain">Runs in progress; this chapter fills in when they finish.</p>'))
page = style + EXTRA_CSS + "\n" + body + "\n<script>\n" + js + "\n" + story_js + "\n</script>\n"
(D / "index_v14.html").write_text(page, encoding="utf-8")
(D / "index_v14_doc.json").write_text(json.dumps({"text": page}), encoding="utf-8")
print(len(page), "chars")
