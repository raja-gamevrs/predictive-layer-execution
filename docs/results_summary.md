# Results summary (all runs to 2026-11-23)

Condensed from `docs/results_report.md`, which has every table, method note and raw-file path.
WikiText-2 held-out text, RTX 5090, BF16 unless stated. "proj err" = relative error at the next
block's attention input projection after its norm. Perplexity changes are relative to the exact model.

## 1. Predictability (H1) — median over boundaries

| Predictor | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| identity | 26% | 34% |
| extrapolation / velocity | 28–33% (worse than identity) | 40–46% (worse) |
| linear, full rank | 18% | 19% |
| linear rank 256 / 64 | 22% / 24% | 24% / 27% |

Consecutive updates are nearly uncorrelated (cos 0.08 / −0.08). Linear explains ~half the update
energy, only at full rank (~63% of a projection's cost).

## 2. Correction economics (H2/H3) — error left, linear predictor

| Correction | Cost (× one projection) | Qwen3-4B | Qwen3.5-4B |
|---|---|---|---|
| none | 0 | 17.8% | 18.7% |
| rank-512 basis | 0.26–0.33 | 10.1% | 12.3% |
| top 25% channels per token | 0.25 | 7.7% | 9.0% |
| worst 50% tokens | 0.50 | 10.6% | 10.9% |

Not low-rank, not token-concentrated; nothing cheap reaches exactness. **Gate failed.**

## 3. Output quality, uncorrected speculation

| Case | Qwen3-4B (11.40) | Qwen3.5-4B (11.82) | Ornith-1.5-9B (12.11) |
|---|---|---|---|
| one layer, identity | +0–1.2% (+0.25% at 26–30) | +1–4%; last block +21% | +0.9% to +26% |
| one layer, linear | +0–0.5% | +0.2–1.7% | −0.3% to +2.1% |
| every layer, identity | 54 | 462 | ~12,000 |
| every layer, linear | 13.25 | 15.52 | 15.93 |
| every layer, linear + top-25% | 11.50 | 12.14 | — |

Linear-attention blocks tolerate speculation 2–4× better than full-attention blocks at mid depth.

## 4. Start depth and lookahead

Early blocks and last 2 exact, linear: ≤ 1% ppl for 9/36 blocks (Qwen3-4B, +0.9%) and 3/32 (Qwen3.5-4B,
+0.8%); from layer 16: +4.1% / +5.5%; from 0: +16% / +29%. Lookahead k (single block, linear): proj
err 18% → 37% from k = 1 to 8 (Qwen3), median ppl cost +1.5% / +1.6% at k = 8.

## 5. BF16 reformulation floor

Exact model is bit-deterministic. Exact correction gives KL 0.0002–0.0015 and 1–2% top-1 flips,
unchanged by FP32 accumulation: BF16 rounding of the output along two equal paths. Report exactness
relative to this floor.

## 6. Precision — exact correction through each projection's own kernel (headline result)

| Model / variant | Exact ppl | Corrected KL (single boundaries) | Top-1 | Verdict |
|---|---|---|---|---|
| Qwen3.5-4B BF16 | 11.82 | 0.0002–0.0006 | 98.8–99.6% | floor |
| Qwen3.5-4B INT8 W8A16 (torchao) | 11.82 | 0.0002–0.0006 | 98.6–99.5% | exact |
| Qwen3.5-4B NVFP4 W4A16 (emulated) | 12.38 | 0.0002–0.0006 | 98.7–99.6% | exact |
| Qwen3.5-4B NVFP4 W4A4 (emulated) | 13.06 | 0.009–0.061 | 86.6–93.4% | broken, 15–100× |
| Ornith-1.5-9B BF16 | 12.11 | 0.0002–0.0011 | 98.5–99.5% | floor |
| Ornith-1.5-9B INT8 W8A16 | 12.09 | 0.0002–0.0009 | 98.5–99.5% | exact |
| Ornith-1.5-9B ModelOpt (FP8 W8A8 attention, real checkpoint) | 12.76 | 0.0008–0.0091 | 96.0–98.6% | broken, 3–10× |
| Qwen3.8-27B INT4 W4A16 (Red Hat, real checkpoint) | 9.33 | 0.0001–0.0006 (all 63 at once: 0.0006) | 98.7–99.8% | exact |

Qwen3.8-27B per layer is the most tolerant model tested (identity ≤ +1% except the last block, +10%;
linear ≤ +0.3%), but all-layer identity collapses hardest (ppl 234k); all-layer linear +13%.

Weight-only quantisation preserves exact reconciliation; activation quantisation breaks it, worst
at early layers. NVFP4 kernels are W4A4 for prefill, W4A16 for decode.

## 7. Overlap timing (Qwen3-4B, block 26 → 27, CUDA graphs)

Idle-GPU rerun, 2026-11-23, each sample = 50 back-to-back graph replays (CV < 1%). Saving per
speculated block, share of sequential time:

| Tokens | Max possible | Overlap, identity | Overlap, linear | Fused single GEMM, identity |
|---|---|---|---|---|
| 1 | 15% | +6.9% | +5.4% | +2.0% |
| 4–16 | 15–18% | +10.5% to +15.7% | +7.4% to +13.7% | +3.9% to +4.7% |
| 32–64 | 16% | +6.6% to +9.7% | +2.4% to +4.9% | +1.9% to +2.4% |
| 128 | 16% | +4.2% | −1.3% | +1.2% |
| 512 | 14% | +2.8% | −2.4% | +2.8% |
| 2048 | 12% | +0.7% | −4.3% | −0.4% |

(Fused column: share of block l's forward time, k = 1, mean of two runs.) Earlier single-replay runs
on a shared GPU missed the small-T gain. **Per block, overlap works at decode-sized inputs; above
512 tokens it does not.** End to end it is bounded by the quality budget (≈ 9 blocks with the linear
predictor, 1–2 with identity): estimated ≈ 2–4% at 1–16 tokens, not yet measured. Eager-mode
profile (shared GPU): 0–47% of side-stream kernel time concurrent, contention cancels it at ≥ 512.

## 8. Prior art

Mechanism published (Layer Parallelism incl. Qwen3-4B, CQIL, EasySpec, InfiniGen); ridge predictor
into later layers published (RidgeFill, Qwen3.5/3.8); predict-then-correct published (SNLP, Song et
al.). New: the precision result (§6), post-norm measurement, the correction-economics and
single-GPU-overlap negative results. "PLE" collides with Gemma's Per-Layer Embeddings.

## 9. Phase II — sensitivity atlas and static allocation (Qwen3-4B, Qwen3.5-4B)

Plan: `docs/phase2_plan.md`. Calibration on WikiText-2 train chunks disjoint from the fitting chunks;
test split untouched. Each block, attention and MLP removed or replaced by a ridge map:

| | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| Best single block (linear stand-in) | +1.5% ppl, 2.6% compute | +1.5%, 3.0% |
| Single ops ≤ 1% ppl | 25 attention stand-ins (0.6% each) | 13 attention (1.0% each), 2 MLP (1.8%) |
| Greedy set ≤ 1% ppl, calibration and holdout | 2 attention ops, 1.2% compute | 1 MLP op, 1.8% |
| First set removing ≥ 10% compute | 5 ops, +17.0% ppl | 5 ops, +12.6% |
| Rank corr. with KL: local state error / last-block error | 0.82 / 0.98 | 0.69 / 0.98 |

KL is near-additive for pairs (−6% to +15%), super-additive for larger sets (×1.15–1.47 at 5–24 ops,
Qwen3-4B). Middle layers are the most tolerant for blocks and MLPs; block 0 and the last block are
catastrophic. **Static allocation misses the proposed 10%-compute gate by an order of magnitude**;
the plan's stop rule for this route is met at 4B scale (larger models, healing, per-token policies
untested).

## Pending

NVIDIA Qwen3.8-27B-NVFP4 (FP8 W8A8 attention, NVFP4 W4A4 MLP) — download stopped at 1.4 / 20.4 GB
(2-hour job limit); loader (`ple.quant.packed`) is ready. Real NVFP4 kernel confirmation
(Linux). Decode-with-KV-cache test, now also the end-to-end test of small-T overlap.
