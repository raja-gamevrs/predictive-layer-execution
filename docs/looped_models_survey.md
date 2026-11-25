# Decoding looped models faster: what exists (survey, 2026-11-25)

A literature search done for follow-up M (`docs/decode_levers.md`). Sources are the papers' arXiv pages,
the Inkling 2 Mini model card and llama.cpp's source. The search was done by a sub-agent; I have not read every
paper in full, so treat the speed-up figures as each paper's own claims. Items the search could not verify are
marked.

## The target: Inkling 2 Mini

- **Layer layout:** 24 stored layers. A 6-layer prelude, a 12-layer core (layers 6–18) run **3 times** per
  token, and a 6-layer coda: 48 layer visits per token. The pass count is fixed (`is_varloop: false`), so
  changing it changes the model's output.
- **Experts and attention:** 128 experts, 8 per token. Sliding-window attention (1,024) except the global
  layers 5, 11, 17 and 23.
- **Its own draft (DSpark/MTP):** 3 layers, block size 8, reading the target at visits 5, 23, 41 and 48.
- **llama.cpp (`inkling2_loop`):**
  - Every pass gets its own KV and conv-state cache slots (48), which point at the same shared weights.
  - Each pass restarts from the prelude output and receives the previous pass through a gated controller.
  - The pass count comes from the GGUF key `inkling2_loop.loop.count`.
- **Engine support:** neither llama.cpp nor vLLM does fewer-pass self-speculation or KV sharing across
  passes for this model.

## Techniques

| Technique | Mechanism | Same output? | Reported speed-up (setup) | Source |
|---|---|---|---|---|
| ACT halting | A learned unit stops the recurrence per position | No | No wall-clock figure | Dehghani et al. 2019 (ICLR) |
| Per-token early exit | Stop looping when states converge (Huginn) or a learned gate fires (Ouro) | No | Compute only: Ouro ~66% MMLU at 2.5 of 4 rounds vs 67.4% at 4 | Geiping et al. 2025; Zhu et al. 2025 |
| Best-available KV for early exit | Early-exiting tokens leave their last-depth KV for deeper readers | No | Up to 30% fewer FLOPs (Ouro) | Mishra et al. 2026 |
| Continuous depth-wise batching | Refill the batch between loop steps | No; batch > 1 | Up to 2–3× (RRT, theoretical), 2.06× (MoR) | Bae et al. 2025 (two papers); Schwethelm et al. 2026 |
| Cross-loop KV sharing | One KV plane for all passes | No | 4× less KV; small accuracy loss at decode, large if shared during prefill | Zhu et al. 2025; Bae et al. 2025; Geiping et al. 2025 |
| Latent warm start | Start the next token's loop from this token's final state | No | 1–2 fewer steps (Huginn); does not apply to Inkling, which restarts from the prelude | Geiping et al. 2025 |
| Fewer-passes self-speculation | Draft with fewer passes, verify at full depth | **Yes** | Proposed, no figure (Huginn) | Geiping et al. 2025 |
| WaveFront Decoding | Tokens staggered across depths: one batched call drafts shallow and verifies deep | **Yes** (without its KV-sharing option) | 2.4× Ouro-2.6B, 3.5× Huginn-3.5B (A6000, batch 1) | Ha & Kim 2026 |
| LoopSpec | Pipelined early-depth drafts with a selective deeper second proposal | **Yes** | 2.8–3.3× Ouro (R = 4), up to 6.8× (R = 32) | Cho et al. 2026 |
| Depth-asynchronous drafts (Mature-V) | The shallow draft reads values already computed at full depth for the prefix | **Yes** (greedy) | 4.0–7.0× (Huginn / Raven, R = 32) | Li et al. 2026 |
| LoopCache | Confidence-guided drafts, a cyclic KV cache that reads one loop per step | Committed tokens exact | 5.1× average (4k–128k context, 8×A100) | Fu et al. 2026 |
| DSpark draft head | A block-parallel draft with a low-rank Markov head and confidence-scheduled verify length | **Yes** | +60–85% per user vs a one-step MTP (DeepSeek-V4, production) | Cheng et al. 2026 |

Could not verify: the venues of Huginn and MoR, LoopCache's per-model numbers, and CDB's hardware.

**Caveat for Inkling:**
- Every looped self-speculation result above is on dense models whose whole stack loops 4 or 32 times.
- Inkling loops 3 times, and 12 of its 48 layer visits are outside the loop.
- Its expert routing makes verifying many tokens at once less nearly free than in dense models.
- Gains should be smaller than the papers report.

## What we test (follow-up M)

1. **M1:** Inkling's own DSpark draft, with a draft-length sweep. It is exact and already in llama.cpp, and
   it is the baseline for everything else.
2. **M2:** a fewer-passes draft, with a Q2_K control at all 3 passes to separate "fewer passes" from "fewer bits".
3. **M3:** how much the third pass changes the output (perplexity with 2 vs 3 passes).
4. **Next, if M2's acceptance is high enough:** a WaveFront / LoopSpec-style schedule.
   - It is exact if each pass keeps its own KV.
   - It needs engine code (this repo or llama.cpp); nothing in llama.cpp does it today.

## References

The BibTeX keys are in `docs/paper/references.bib`: dehghani2019universal, geiping2025recurrent, zhu2025ouro,
bae2025mor, bae2025relaxed, mishra2026dynamic, schwethelm2026cdb, ha2026wavefront, cho2026loopspec, li2026das,
fu2026loopcache, cheng2026dspark, tml2026inkling2mini.
