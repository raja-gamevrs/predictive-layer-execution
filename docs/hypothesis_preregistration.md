# Hypothesis pre-registration

Status: **draft** — finalise before any test-set results are viewed.

**Primary RQ:** Can the runtime predict a layer's intermediate activation, execute selected
downstream projections early, then reconcile them at a lower marginal cost than recomputation,
while lowering end-to-end latency at matched output quality?

| ID | Hypothesis | Test |
|---|---|---|
| H1 | Hidden-state deltas are predictable | Identity, linear, low-rank, MLP predictors; held-out activation error |
| H2 | Error has exploitable structure | SVD spectrum, channel/block sparsity, token correlation, residual norms |
| H3 | Structured correction is cheaper than recomputation | Exact vs speculate+full vs low-rank/block/sparse correction microbenchmarks |
| H4 | Real overlap reduces critical-path latency | CUDA events + Nsight timeline; serialized vs concurrent vs exact |
| H5 | Prefill is more favourable than decode | Prefill sweep; decode sweep as a separate benchmark |
| H6 | Adaptive policy beats always-on speculation | Confidence / layer / op / resource gate ablations |

Predictor families (H1), in order: identity / zero-delta, residual extrapolation, velocity
(α, β sweep), linear, low-rank, tiny MLP, then multi-state (depth/trajectory, gated, shared)
only if justified. See [`techniques.md`](techniques.md).

Gates and tolerances: see [`configs/quality_thresholds.yaml`](../configs/quality_thresholds.yaml).

## Change log

Any change after results are visible: date, rationale, what reran. Label results exploratory.
