# Results report

Running log of evidence. Exploratory until the pre-registration is finalised.

## 2026-11-23 — H1 first pass: residual-stream predictability, Qwen3-4B-Base (BF16, RTX 5090)

Script: `scripts/predictability_study.py`. Data: WikiText-2 raw (CC BY-SA), fit on `train`
(130,816 tokens, 256 × 512), scored on held-out `test` (32,704 tokens, 64 × 512); position 0
excluded. Raw: [`results/processed/Qwen3-4B-Base_predictability.csv`](../results/processed/Qwen3-4B-Base_predictability.csv).

Metrics: `Δrel` = ‖ĥ − h‖ / ‖Δ‖ (identity = 1.0; lower explains more of the update).
`proj` = relative error at the next block's `q_proj` after its input RMSNorm — the tensor a
speculative QKV launch would actually consume.

| Predictor | Median Δrel | Median proj err | Median cos(Δ̂, Δ) | Cost per token (MACs) |
|---|---|---|---|---|
| Identity (ĥ = h_l) | 1.000 | 0.260 | — | 0 |
| Residual extrapolation | 1.295 | 0.330 | 0.08 | ~0 |
| Velocity, α = 0.5 | 1.050 | 0.284 | 0.08 | ~0 |
| Linear, full rank (ridge) | **0.702** | **0.178** | 0.72 | 6.6 M (≈ 63% of `q_proj`) |
| Linear, rank 256 | 0.859 | 0.221 | 0.53 | 1.3 M (≈ 12% of `q_proj`) |
| Linear, rank 64 | 0.924 | 0.241 | 0.40 | 0.3 M (≈ 3% of `q_proj`) |

For scale: `q_proj` is 2560 → 4096 (10.5 M MACs/token); a whole Qwen3-4B block is roughly 100 M.

### Findings

1. **Zero-parameter predictors fail.** Extrapolation and velocity are worse than identity at
   almost every layer: consecutive residual updates are close to uncorrelated in direction
   (cos ≈ 0.08). Depth does not behave like a smooth trajectory in this model.
2. **A learned linear map has real but limited signal.** Full-rank ridge explains about half the
   update energy (Δrel 0.70 → 1 − 0.70² ≈ 51%) and cuts next-block projection error from 26% to
   18% (median). Signal survives on held-out text, so it is not memorisation.
3. **Cheap predictors keep little of it.** Rank 256 recovers only part of the gain (proj 0.22);
   the update is not low-rank in `h_l`.
4. **State error is the wrong metric.** At layers 17–21 identity is within 8–10% of the next state
   in L2, yet the projection error stays ~25%. Likely cause (not yet verified): a few
   huge-magnitude channels dominate ‖h‖ there and the RMSNorm before `q_proj` rescales them
   away. Either way, prediction must be judged after the norm.
5. **Outlier layers.** Layers 0, 16 and 35 rewrite most of the state (identity rel_err
   0.86–0.99), layer 34 about half (0.49). Linear prediction is best at 34–35 (Δrel 0.40 and
   0.17), so the last layers are the most predictable.

### Implication for H3 (correction economics)

Even the best predictor leaves ~18% relative error at the projection input, so speculation must
be corrected. The exact correction `W e` costs as much as the projection itself, and the
predictor adds 3–63% of a projection on top. PLE only pays if the error `e` is itself
structured — low-rank, sparse or channel-concentrated — so that an approximate correction is
cheap and accurate. That is H2, and it is the next measurement.

### Not yet done

Single model, single dataset, sequence length 512, batch 1. No timing. Gemma/Qwen3.5 contrast,
token-position breakdown and error-structure (SVD/sparsity) analysis pending.
