# Experiments E1–E5: can approximate PLE pay for itself? (pre-registered 2026-11-23)

Written before any of these runs. Bars below do not change after results are seen; any change is
dated here with its reason and every affected run is redone.

## Why these five

Phase I: no cheap predictor or correction reaches exactness (next-block updates are nearly
uncorrelated, cos ≈ 0.08). Exact correction is lossless but costs a full projection on the critical
path after block l, so **lossless PLE saves no time by construction**; the measured overlap saving
(6.9–15.7% of a block at 1–64 tokens, `results/processed/quiet_gpu/`) is for *uncorrected*
speculation. Phase II: static removal / linear replacement saves ≤ 1.8% of compute within 1%
perplexity. What is left is uncorrected (approximate) speculation, made cheaper or more accurate:

| # | Question | Changes |
|---|---|---|
| E1 | What does uncorrected speculation actually buy end to end in single-stream decode? | Latency only (sets the ceiling for E2–E5) |
| E2 | Does a per-token gate (fall back to the exact projection when the predicted input is far off) recover quality cheaply? | Quality vs fraction of tokens speculated |
| E3 | Does a short LoRA "healing" fine-tune with speculation on make the model tolerate it? | Quality at a fixed speculated set |
| E4 | Does a small trained predictor beat the ridge map on the error that matters (after the next norm)? | Predictor error and quality |
| E5 | Does running the speculative branch in low precision (FP8 weights) cut its cost without hurting quality? | Latency and quality |

## Common setup

- Models: Qwen3-4B-Base (all five); Qwen3.5-4B-Base where the code path supports it (E2, E4).
  RTX 5090, BF16, batch 1.
- Data, all WikiText-2 **train** split, 512-token chunks: predictor fitting / training = chunks 0–255;
  calibration = chunks 400–415; holdout = chunks 600–615. E3 healing trains on chunks 1000–1999
  (disjoint from all of the above). **The test split is not used** (reserved for a final evaluation
  under the Phase II gate).
- Quality vs the exact original model: mean KL(p_exact ‖ p_approx) per token, top-1 agreement,
  perplexity change. Position 0 kept exact and excluded.
- Speculation: block l+1's attention input projections (q/k/v) computed from N(ĥ), ĥ = f(h_l),
  uncorrected, unless the experiment says otherwise.

## Pass bar (the Phase II gate, unchanged)

Quality: mean KL ≤ 0.01 nats/token **and** perplexity increase ≤ 1.0%, on calibration and holdout
separately. Benefit: ≥ 5% lower end-to-end decode latency per token (CV < 5% over 3 runs), measured
in E1's harness. An experiment passes only if one configuration meets both on holdout.

## Per-experiment design and success criteria

**E1 — End-to-end decode.** A minimal Qwen3 decode step written against the model's own weights
(static KV cache, whole step captured in one CUDA graph), checked against Transformers' logits.
Variants: exact; speculate every block's QKV from the block before it on a second stream (all 35
boundaries, identity and linear predictor); speculate only the most tolerant set from Phase I
(start 24, last 2 exact: 9 blocks). Prompt 512 tokens, 128 generated tokens timed per step, 3 runs.
Reported: ms per token and saving vs exact. *Ceiling:* if all-boundary speculation saves < 5%, no
quality fix in E2–E5 can pass the benefit bar, and their results are reported as quality-only.

**E2 — Per-token gate.** At each speculated boundary, once h_(l+1) is known, compute
r = ‖N(h) − N(ĥ)‖ / ‖N(h)‖ (cost: two norms) and use the exact projection for tokens with r > τ.
Sweep τ; report quality vs share of (token, block) pairs speculated. Benefit converts through E1's
per-block saving × share speculated. Pass: quality bar met at a share that gives ≥ 5% in E1 terms.

**E3 — Healing.** LoRA (rank 16, q/k/v/o and MLP projections) trained by distilling the exact
model's next-token distribution (KL) while speculation is on, at the 9-block set and at the
start-16 set (17 blocks); ridge predictors frozen. 1,000 chunks × 512 tokens, 1 epoch. Quality is
measured against the *original exact model*, and the healed model is also reported without
speculation. Pass: quality bar met at a set large enough for ≥ 5% in E1 terms.

**E4 — Trained predictor.** Per boundary, ĥ = h + MLP(h) with a bottleneck (d → r → d, r ∈ {256, 1024},
SiLU), trained to minimise ‖N_(l+1)(h_(l+1)) − N_(l+1)(ĥ)‖² (the error that reaches the projections),
initialised from the ridge solution in a parallel linear path. Compared with ridge at equal data.
Success: median relative projection-input error ≤ 0.8 × ridge's (Qwen3-4B ridge: 18%), and then the
E1/E2 quality checks rerun with it.

**E5 — Low-precision speculative branch.** Speculative q/k/v computed with FP8 (E4M3, per-channel
weight scales) copies of block l+1's weights: halves the side stream's weight traffic. Quality
(uncorrected speculation, BF16 vs FP8 branch) and E1 decode latency. Success: latency saving at
least 1.25× the BF16 branch's with quality change from FP8 ≤ 10% of speculation's own KL.

## Amendments

- 2026-11-23, after E1 and before any E3 run: E1 shows the 9- and 17-block sets cannot reach the 5%
  benefit bar (≤ 3.4% even with an FP8 branch), so E3 also heals at start4_tail2 (29 blocks) and at
  every boundary (35 blocks). The 9- and 17-block runs stay as pre-registered. Bars unchanged.
- 2026-11-24, E4: the first run trained on 128 chunks (ridge: 256), against the "equal data" design;
  rerun on 256 chunks for the error comparison (`e1_e5/e4_equal_data/`). Both reported.
- 2026-11-23, E5: row-wise scaled FP8 GEMM is not supported on sm_120 in this PyTorch build, so the
  FP8 branch uses one scale per weight tensor and one per activation tensor (not per channel).

## Results (2026-11-24) — none of the five passes the bar

Raw files: `results/processed/e1_e5/`. Calibration / holdout = WikiText-2 train chunks 400–415 /
600–615; test split not used.

**E1 — decode latency, Qwen3-4B** (`scripts/e1_decode_bench.py`, `src/ple/decode.py`). 512-token
prompt, 128 decode steps per run, 5 runs, whole step in one CUDA graph, run in both variant orders.
Exact: 8.25–8.28 ms/token. The decoder matches Transformers (decode logits KL 0.0008, top-1 100%;
prefill KL 0.0013) and its identity speculation matches the hook simulation (KL 0.0026). Run-to-run
CV ≤ 0.11%. Saving per token vs exact (forward / reverse order):

| Speculated blocks | BF16 branch, identity | BF16 branch, linear | FP8 branch, linear |
|---|---|---|---|
| all 35 | +2.95 / +2.77% | +1.15 / +1.24% | **+6.28 / +6.68%** (identity +6.01 / +6.72%) |
| 17 (from block 16) | | +0.54 / +0.61% | +2.86 / +3.35% |
| 9 (start 24, last 2 exact) | | +0.26 / +0.40% | +1.31 / +1.95% |
| fused identity, 18 alternate blocks | +1.49 / +1.71% | | |

Control: every block's own QKV in FP8 (no speculation) is 0.6–0.8% *slower*, so the FP8 gain is the
side stream interfering less with the main stream, not a faster kernel. **Only near-all-block
speculation with an FP8 branch clears 5%.** Quality there: +18% (holdout) / +24% (cal) perplexity.

**E2 — per-token gate** (`scripts/e2_token_gate.py`). Share of (token, block) pairs left speculated:

| | Qwen3-4B (all / start4_tail2) | Qwen3.5-4B (all / start4_tail2) |
|---|---|---|
| τ = 0.2, cal / hold | 15% / 26% speculated, KL 0.007–0.014 | 5–12%, KL 0.001–0.004 |
| τ = 0.25, cal / hold | 34–45%, KL 0.024–0.033 | 18–28%, KL 0.006–0.013 |
| τ = 0.4 (80–91% speculated) | +7–9.5% ppl, KL 0.11–0.13 | +6–8% ppl, KL 0.08–0.12 |

Largest share within KL ≤ 0.01 (cal / hold): Qwen3-4B 15% / 10% (all) and 15% / 10% (start4_tail2); Qwen3.5-4B 6% / 12% (all) and 18% / 27% (start4_tail2). ~77% speculated is needed for 5% (E1 terms: 5 / 6.5%). The signal is informative (a random gate at the same
share has 3–5× the KL) but far from enough. **Fail.**

**E3 — LoRA healing, Qwen3-4B** (`scripts/e3_healing.py`; rank 16, all projections, 1,000 × 512 tokens,
KL distillation from the exact model, ~4.5 min per set). Holdout (cal in the JSON):

| Set | Before | Healed, speculation on | Healed, speculation off |
|---|---|---|---|
| tol9 (9 blocks) | KL 0.015, +1.19% | KL 0.015, +0.94% | KL 0.007, +0.30% |
| start16 (17) | KL 0.064, +4.37% | KL 0.051, +2.85% | KL 0.017, +0.94% |
| start4_tail2 (29) | KL 0.185, +15.2% | KL 0.117, +7.41% | KL 0.036, +1.69% |
| all (35) | KL 0.218, +18.2% | KL 0.138, +8.78% | KL 0.042, +1.88% |

Healing removes 35–50% of the damage at large sets, and the healed weights drift from the original
(KL 0.04 with speculation off). No set meets KL ≤ 0.01; tol9 meets the ppl bound but gives ≤ 2% speed.
**Fail.**

*Follow-up, E3b* (`docs/e3b_healing_scaling.md`, separately pre-registered): identity predictor at all
35 blocks, LoRA rank 16 trained up to 16k steps (8.2M tokens). Holdout KL 1.65 → 0.21 in the first 1k
steps, then flat (0.196 at 4k, 0.189 at 16k, +10.4% ppl); rank 64 is worse. Its stop rule fired: no
longer run of this recipe is justified.

**E4 — trained predictor** (`scripts/e4_trained_predictor.py`). Median relative error at the next
block's projection input (after its norm), holdout, trained on 256 chunks like ridge
(`e1_e5/e4_equal_data/`; the first run used 128 chunks and is kept):

| | ridge | linear, normed loss | MLP r=256 | MLP r=1024 |
|---|---|---|---|---|
| Qwen3-4B | 0.270 | 0.267 | 0.262 | 0.261 (0.97× ridge) |
| Qwen3.5-4B | 0.298 | 0.298 | 0.298 | 0.298 |

Bar was ≤ 0.8× ridge. With the 128-chunk predictors, all-block quality moved from +18.2% to +17.7%
ppl (holdout). **Fail**: a nonlinear map does not make the next block's input much more predictable.

**E5 — FP8 speculative branch** (`scripts/e5_fp8_branch_quality.py`; E4M3, per-tensor scales).
Added KL from FP8: +0.003–0.004 at all blocks (≈ 1.5% of speculation's own KL 0.22–0.28), ≤ +0.002
with the gate. Latency saving 2.3–5.5× the BF16 branch's (E1). **E5's own criterion is met**; it does
not rescue the overall bar because the quality problem is speculation itself.

**Conclusion.** Uncorrected speculation buys at most ~6.5% decode latency on Qwen3-4B (FP8 branch,
all blocks), at +18–24% perplexity. Gating, healing and a better predictor each recover part of that;
none comes within an order of magnitude of the 1% / KL 0.01 budget at the coverage the speedup needs.
With Phase I (exact correction saves nothing) and Phase II (static removal ≤ 1.8%), single-GPU
approximate PLE does not pay for itself at 4B scale on this hardware.
