# Layer manipulation techniques

How PLE manipulates an existing small model **without changing pretrained weights** at first.
Predicting an activation is not a speedup: predictor, speculative work, correction,
synchronisation and contention must cost less than the work saved or overlapped.

Residual view of a decoder block: `h_(l+1) = h_l + Δ_l`, with `Δ_l = F_l(h_l)`.

## Five approaches

| | Approach | What changes | When |
|---|---|---|---|
| A | Predict the residual stream | Nothing in the model; predictors read captured states | First |
| B | Speculative branch | Downstream ops start from a predicted state while exact execution continues | Only at operation level; full suffix duplication only if profiling justifies it |
| C | Attention / MLP internals | Predict submodule inputs; run QKV or MLP projections early | After A shows signal |
| D | Lightweight PLE adapters | Small trained predictors between frozen blocks | If A finds predictable structure |
| E | Execution scheduling | Speculative kernels on separate CUDA streams | Only after correction economics pass |

## A — Zero-parameter predictors first

Implemented in [`src/ple/predictors`](../src/ple/predictors/__init__.py).

| Predictor | Formula | Tests whether |
|---|---|---|
| Identity | `ĥ_(l+1) = h_l` | The state barely changes across the transition |
| Residual extrapolation | `ĥ_(l+1) = h_l + Δ_(l-1)` | The next update resembles the previous one |
| Velocity | `v_l = h_l − h_(l-1)`; `ĥ_(l+1) = h_l + α v_l + β v_(l-1)` | Depth behaves like a trajectory with first/second-order dynamics |
| Learned | `P(h_(l-k) … h_l) → Δ̂` | Linear, low-rank, gated or tiny-MLP predictors over several prior states (Approach D) |

With α = 1, β = 0, velocity reduces to residual extrapolation.

**Residual-dynamics direction:** characterise the model's own update dynamics before training
anything. Measure predictability by layer, token position, prompt length and workload; some
transitions or layer groups may be cheap to forecast.

## B — Speculative branches

Exact path runs layers normally; the predictor estimates a downstream boundary state;
selected downstream ops start from it; at reconciliation, correct, reuse or discard.
Unverified tensors must never reach the exact KV cache or exact downstream execution.

## C — Operation-level speculation

Pre-norm block: `a = Attn(Norm₁(h))`, `u = h + a`, `m = MLP(Norm₂(u))`, `h' = u + m`.

- **C1 MLP input:** predict `û`, compute `W_up û` early, correct with `W_up (u − û)`.
- **C2 QKV input:** predict `x̂`, compute `Q̂, K̂, V̂` early, correct each with `W (x − x̂)`.

Because of the norm, the speculated input is `Norm(ĥ)`, not `ĥ`; the error that the linear
correction must cover is `Norm(h) − Norm(ĥ)`.

Economics test: correction must be materially cheaper than recomputing, or the speculative
compute must be genuinely overlapped. Otherwise PLE only adds work.

## D — Adapters

Loss `‖ĥ_(l+1) − h_(l+1)‖²` is a start, not the criterion. Also evaluate cosine similarity,
Q/K/V and MLP projection error, final-logit divergence and calibration — low hidden-state MSE
can still give unacceptable downstream error.

| Adapter | Inputs → output | Use |
|---|---|---|
| Linear | `h_l → Δ̂` | Minimal baseline |
| Low-rank | `h_l → Δ̂` | Lower predictor cost |
| Depth / trajectory | `h_(l-2), h_(l-1), h_l → Δ̂` | Use several prior states |
| Gated | several prior states → `Δ̂` | Adaptive combination |
| Shared | layer-group inputs → next state | Parameter sharing |

## E — Scheduling order

1. Load the model and identify blocks and submodules.
2. Hook selected outputs and record exact activations.
3. Generate predictions at chosen boundaries.
4. Benchmark speculative QKV / MLP projections on their own.
5. Measure correction / reuse / discard economics and kernel timing.
6. Only then add CUDA streams, events and dependency tracking.
7. Never commit unverified speculation to exact downstream state.

## Working research question

Can a pretrained transformer use fine-grained, confidence-gated predictive execution of selected
internal operations, with projection reuse and selective correction, to cut real inference
critical-path latency, without wholesale layer-parallel approximation or retraining the base
model? This is a working hypothesis, not a novelty or performance claim.
