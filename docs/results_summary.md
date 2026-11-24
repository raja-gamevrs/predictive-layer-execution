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

Weight-only quantisation preserves exact reconciliation; activation quantisation breaks it, worst
at early layers. NVFP4 kernels are W4A4 for prefill, W4A16 for decode.

## 7. Overlap timing (Qwen3-4B, block 26 → 27, CUDA graphs)

| Tokens | Max possible | Measured | Kernels concurrent (eager profile) |
|---|---|---|---|
| 16–128 | 16–17% | −0.6% to +4.6% | 0–3% |
| 512 | 20% | +0.5% | 35% |
| 2048 | 12% | +0.9% | 47% |
| 8192 | 12% | −0.7% | 24% (QKV kernels +20% slower) |

**Gate failed.** Timing CV exceeds the 5% gate (GPU shared with the desktop).

## 8. Prior art

Mechanism published (Layer Parallelism incl. Qwen3-4B, CQIL, EasySpec, InfiniGen); ridge predictor
into later layers published (RidgeFill, Qwen3.5/3.8); predict-then-correct published (SNLP, Song et
al.). New: the precision result (§6), post-norm measurement, the correction-economics and
single-GPU-overlap negative results. "PLE" collides with Gemma's Per-Layer Embeddings.

## Pending

Qwen3.8-27B INT4 (W4A16) and NVIDIA NVFP4 (W4A4) — downloading. Real NVFP4 kernel confirmation
(Linux). Decode-with-KV-cache test.
