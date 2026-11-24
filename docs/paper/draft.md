# Predictive Layer Execution: Can Transformers Speculate Their Own Next Layer?

*Working draft. Numbers are copied from `docs/results_report.md` and the CSV/JSON files under
`results/processed/`; every table names its source. Sections marked TODO are not yet supported by
experiments. Related work is in [`../research_review.md`](../research_review.md) and
[`references.bib`](references.bib).*

## Abstract (draft — to be rewritten when results are final)

Transformer inference is sequential in depth: block l+1 cannot start until block l finishes. We
ask whether a pretrained model's own residual stream is predictable enough to *speculate* part of
block l+1 — its attention input projections — from block l's input, overlap that work with block
l, and reconcile it against the exact state afterwards (Predictive Layer Execution, PLE). On
Qwen3-4B-Base we find that (i) simple trajectory extrapolation of the residual stream fails,
(ii) a per-layer linear predictor explains about half of each block's update but leaves ~18%
error at the next projection, (iii) that error is neither low-rank nor token-concentrated, so no
cheap correction recovers exactness, (iv) uncorrected speculation of a single late layer costs
only ~0.25% perplexity, while speculating every layer collapses the model, and (v) on an RTX 5090
overlap saves up to 16% of a block at decode-sized inputs (1–64 tokens) but nothing from ~512 tokens
up, and the quality budget limits how many blocks it can be applied to. TODO: extend to
Qwen3.5 (hybrid linear attention), Ornith-1.5 and Qwen3.8-27B under BF16/INT8/INT4/NVFP4.

## 1. Introduction (outline)

- Depth-sequential dependency as a latency floor for prefill and decode.
- Existing ways around it: speculative decoding (across tokens), early exit / layer skipping
  (drop work), layer-parallel and parallel-block architectures (change the model or approximate).
- PLE's question: keep the exact model, speculate *within* a forward pass at operation
  granularity, and pay a reconciliation cost. When, if ever, does that pay?

**Positioning after the prior-art review (2026-11-23, see §6).** The mechanism is not new.
Running block l+1's attention on block l's input without retraining is Layer Parallelism (LP,
Calvo González et al.), CQIL and EasySpec; LP evaluates the same Qwen3-4B model. Applying the next
layer's query weights to the current normalised input is InfiniGen. A per-layer closed-form ridge
map from an earlier state into a frozen later layer's projections is RidgeFill (Nov 2026, on
Qwen3.5-9B and Qwen3.8-27B). Predict-then-correct across depth is SNLP and, earlier, Song et al.;
our exact correction is one Jacobi/Newton step. The paper is therefore framed as an **empirical
study**, and claims only:

1. a measurement framework that separates predictability, error structure, output quality and
   real overlap, with errors measured **after the next block's norm**, at the projection input;
2. a **per-boundary sensitivity map** of single-layer speculation (consistent with LP's advice to
   keep the final layers sequential, and finer-grained than LP's contiguous groups);
3. **negative results on reconciliation economics**: trajectory extrapolation is worse than
   identity, the error is not low-rank or token-concentrated, and no correction cheaper than `W e`
   reaches exactness;
4. the **BF16 reformulation floor** as the correct reference for any "exact" claim;
5. a **single-GPU overlap** measurement (LP, CQIL and EasySpec obtain their speedups from
   multiple GPUs), currently negative;
6. the same analysis across architectures and precisions: on hybrid linear attention
   (Qwen3.5-4B) the picture replicates, linear-attention blocks tolerate speculation better than
   softmax blocks, and **exact reconciliation survives weight-only quantisation (INT8, NVFP4
   W4A16) but not activation quantisation (NVFP4 W4A4)** (§4.6). TODO: Ornith-1.5-9B and
   Qwen3.8-27B on real quantised checkpoints.

**Name.** "PLE" collides with Gemma's Per-Layer Embeddings (same on-device-inference community)
and with Progressive Layered Extraction (recommendation). Spell out "Predictive Layer Execution"
or choose a new name before submission.

## 2. Method

### 2.1 Notation

A decoder block maps the residual stream `h_l ∈ R^{T×d}` to `h_{l+1} = h_l + Δ_l`, with
`Δ_l = F_l(h_l)` (attention, MLP, norms). Block l+1 starts with an RMSNorm `N_{l+1}` and its
attention input projections `y = W N_{l+1}(h_{l+1})` (q/k/v for softmax attention; the fused
input projections for Gated-DeltaNet linear-attention layers).

### 2.2 Speculate and reconcile

1. **Predict** `ĥ_{l+1} = P_l(h_l, h_{l-1}, …)` before block l runs.
2. **Speculate** `ŷ = W N_{l+1}(ĥ_{l+1})` concurrently with block l.
3. **Reconcile** when `h_{l+1}` is known: `e = N_{l+1}(h_{l+1}) − N_{l+1}(ĥ_{l+1})`, and
   `y = ŷ + W e` exactly. Because `W e` costs one full projection, PLE needs either a cheap
   approximate correction `ỹ = ŷ + C(W, e)` or no correction at all.

The norm matters: the projection sees `N(h)`, not `h`. An error that is small relative to `‖h‖`
can be large after normalisation (§4.1, finding 4).

### 2.3 Predictors (`src/ple/predictors`)

| Name | Definition | Parameters | Cost / token |
|---|---|---|---|
| identity | `ĥ_{l+1} = h_l` | 0 | 0 |
| extrapolation | `ĥ_{l+1} = h_l + (h_l − h_{l−1})` | 0 | O(d) |
| velocity(α) | `ĥ_{l+1} = h_l + α(h_l − h_{l−1})` | 0 | O(d) |
| linear | `ĥ_{l+1} = h_l + h_l A_l + b_l`, ridge regression on traces | d² + d | d² MACs |
| linear rank r | SVD truncation of `A_l` to rank r | 2rd + d | 2rd MACs |

Ridge: normal equations accumulated over training tokens (per-batch products in FP32, running
sums in FP64), `λ = 10⁻³ · tr(XᵀX)/d`, bias unpenalised (`src/ple/predictors/linear.py`).

### 2.4 Corrections (`src/ple/correction`)

| Name | Definition | Cost vs one exact projection |
|---|---|---|
| none | `ỹ = ŷ` | 0 |
| full | `ỹ = ŷ + W e` (BF16) | 1 |
| full_fp32 | as full, `e` and `W e` in FP32 | 1 |
| basis r | `ỹ = ŷ + (W U_r)(U_rᵀ e)`, `U_r` = top-r principal directions of `e` on held-in data | r(d + d_out)/(d·d_out) |
| top-k channels | per token, keep the k channels of `e` with largest `|e_i|·‖W_{:,i}‖` | k/d (+ gather) |
| token-selective | full correction for the fraction f of tokens with largest `‖e‖` | f |

### 2.5 Measurements

- **Predictability (H1)** — `scripts/predictability_study.py`. Per layer and predictor:
  `Δrel = ‖ĥ − h‖/‖Δ‖` (identity = 1), cosine of predicted vs true update, and relative error at
  the next block's normed input and first input projection.
- **Error structure / correction economics (H2–H3)** — `scripts/error_structure_study.py`.
  Relative projection error left by each correction, against its cost.
- **Output quality** — `scripts/approx_quality_study.py`. Hooks replace block l+1's attention
  input projections inside the real forward pass with speculated-and-corrected values
  (`src/ple/speculation_policy`); everything downstream runs normally. Compared with the exact
  model: mean KL(p_exact ‖ p_approx) per token, top-1 agreement, perplexity on the actual next
  token. Position 0 (attention-sink token) is always kept exact and excluded. A rerun of the
  exact model gives the run-to-run determinism floor.
- **Overlap (H4)** — `scripts/bench_overlap.py`. Block l alone, block l+1's norm+QKV alone, the
  sequential pair, and the pair with QKV launched on a second CUDA stream from `N(ĥ)`; eager and
  CUDA-graph modes; CUDA-event timing, warm-up, ≥ 50 repeats, median / p95 / CV.

### 2.6 Correctness tests (`tests/`)

- Activation capture equals Transformers' `hidden_states` (and the final norm of the last state);
  hooks are removed afterwards.
- Correction identity `W h = W ĥ + W(h − ĥ)` in FP32 and BF16 tolerances; each structured
  correction is exact in its limiting case (full basis, all channels, all tokens) and reduces to
  ŷ in the empty case.
- The in-model speculation simulator with full correction reproduces the exact model (FP64 test
  model), without correction differs, keeps position 0 exact, and leaves no hooks behind.

## 3. Experimental setup

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 5090 32 GB (compute capability 12.0), driver 617.42; i7-13700KF; 64 GB RAM; Windows 11 Pro 26300 |
| Software | Python 3.11.14, PyTorch 2.14.1+cu130, Transformers 5.24.0 (full lock: `requirements-lock.txt`) |
| Models | Qwen3-4B-Base `906bfd4` (36 layers, d = 2560, BF16); TODO: Qwen3.5-4B-Base `1001bb4`, Ornith-1.5-9B `489cb97` (BF16 / INT8 / NVFP4), Qwen3.8-27B (INT4 W4A16 `71dface`, NVFP4 W4A4 `482ca0f`) |
| Data | WikiText-2 raw v1 (`Salesforce/wikitext` `b08601e`, CC BY-SA). Predictors fit on train chunks 0–255 (512 tokens each, 130,816 scored tokens); error bases on train chunks 256–383; all scores on the test split (64 chunks, or 16 for output quality). No test text is used for fitting. |
| Precision | BF16 weights and activations; metrics accumulated in FP32 / FP64 |
| Shared GPU | The 5090 also drives the desktop (5–16% utilisation from other processes during runs); clocks not locked. Affects timing variance, not the quality metrics. |

## 4. Results

### 4.1 The residual stream is only partly predictable (H1)

Source: `results/processed/Qwen3-4B-Base_predictability.csv`.

| Predictor | Median Δrel | Median proj. error | Median cos(Δ̂, Δ) |
|---|---|---|---|
| identity | 1.000 | 0.260 | — |
| extrapolation | 1.295 | 0.330 | 0.08 |
| velocity α = 0.5 | 1.050 | 0.284 | 0.08 |
| linear (full) | 0.702 | 0.178 | 0.72 |
| linear r = 256 | 0.859 | 0.221 | 0.53 |
| linear r = 64 | 0.924 | 0.241 | 0.40 |

1. Consecutive updates are nearly uncorrelated in direction (cos ≈ 0.08); depth is not a smooth
   trajectory, so zero-parameter extrapolation is worse than identity.
2. A linear map explains ~51% of update energy on held-out text; most of the gain needs full rank.
3. Errors must be judged after the norm: at layers 17–21 identity is within 8–10% in L2 but 25%
   off at the projection.
4. Layers 0, 16 and 35 rewrite most of the state; the last two layers are the most predictable.

### 4.2 The error is unstructured, so correction is not cheap (H2–H3)

Source: `results/processed/Qwen3-4B-Base_error_structure.csv`. Median relative `q_proj` error
over layers 0–34 (linear predictor): none 0.178; basis r = 512 (cost 0.33) 0.101; top 25%
channels (cost 0.25) 0.077; worst 50% tokens (cost 0.5) 0.106. A rank-256 basis captures only
38–65% of error energy. No correction reaches exactness below full cost.

### 4.3 Output quality of approximate speculation

Source: `results/processed/Qwen3-4B-Base_approx_quality.csv`. Exact perplexity 11.40.
Single-layer uncorrected speculation: +0.0–1.2% perplexity (layers 2–30), +0.25% at layers 26–30,
+4% at layer 34 (identity). Every layer at once: identity 54.0, linear 13.25, linear + top-25%
channels 11.50.

**Reformulation floor.** The exact model is bit-deterministic (rerun KL 0.0). Exact correction
`ŷ + W e`, whether accumulated in BF16 or FP32, gives KL 0.0004–0.0012 and 98.1–99.2% top-1
agreement at every layer, and the same when all layers are reconciled at once. The difference is
the BF16 rounding of the projection output along two algebraically equal paths (≈ 1 ulp); a
single-ulp perturbation of one layer's q/k/v flips ~1.7% of near-tie argmax tokens. We therefore
report all approximate results relative to this floor, and define "exact" as "within the BF16
reformulation floor", which any fused or re-tiled kernel would also incur.

### 4.4 Overlap: a per-block gain at decode-sized inputs only (H4)

Source: `results/processed/Qwen3-4B-Base_bench_overlap_l26_{eager,graphs}.json`, boundary 26.

| Tokens | Block l (ms, graphs) | QKV path (ms) | Sequential (ms) | Overlap, identity | Upper bound |
|---|---|---|---|---|---|
| 16 | 0.280 | 0.060 | 0.335 | +4.6% | 16.5% |
| 128 | 0.268 | 0.061 | 0.324 | −0.6% | 17.4% |
| 512 | 0.732 | 0.123 | 0.910 | +0.5% | 19.6% |
| 2048 | 3.276 | 0.426 | 3.733 | +0.9% | 12.2% |
| 8192 | 18.952 | 1.759 | 21.473 | −0.7% | 11.7% |

In eager mode overlap is −15% to +2% (launch overhead dominates below 2k tokens). With the linear
predictor's extra GEMM on the side stream, overlap is 2–4% *slower*. Sequential CV with graphs is
0.05–0.13, above the 5% baseline gate, so differences under ~5% are not significant.
**Kernel concurrency (CUPTI profile, eager).** Below ~512 tokens the speculative QKV kernels finish
before block l's kernels are launched (0–3% concurrent): eager execution is launch-bound, so there
is nothing to overlap into. From 512 to 8192 tokens, 24–47% of the QKV kernel time does run
concurrently with block l, but contention cancels the gain: at 8192 tokens the overlapped QKV
kernels take 2.23 ms versus 1.86 ms alone (+20%). Real concurrency exists; useful concurrency does
not, at single-block granularity on one GPU.

**Idle-GPU rerun with amortised timing (supersedes the table above below 512 tokens).** With the
GPU otherwise idle and each sample timing 50 back-to-back graph replays (sequential CV < 1%;
`results/processed/quiet_gpu/batched/`), overlap with the identity predictor saves +6.9% (1 token),
+10.5% (4), +15.7% (16), +9.7% (32), +6.6% (64), +4.2% (128), +2.8% (512) and +0.7% (2048) of the
sequential time, against upper bounds of 12–18%. The single-replay figures above were dominated by
launch overhead around 20–70 µs paths. So overlap is useful per block at decode-sized inputs and not
beyond ~512 tokens. Fusing the speculative projections into block l's own QKV GEMM (gains folded
into the weights; identity predictor only) saves a reproducible +1.9% to +5.5% of a block per
speculated block below 128 tokens. End to end the gain is limited by the quality budget (≈ 9 blocks
with the linear predictor); a measured decode test is pending.

### 4.5 Second architecture: hybrid linear attention (Qwen3.5-4B-Base)

Source: `results/processed/Qwen3.5-4B-Base_*` and `results/processed/qwen35_linear_blocks/`;
details in `docs/results_report.md`. The predictability and error-structure results replicate
(extrapolation worse than identity, cos(Δ̂, Δ) = −0.08; linear Δrel 0.71; best cheap correction
top-25% channels leaves 9% projection error). Uncorrected identity speculation is costlier than on
Qwen3 (+1–4% perplexity per layer; +21% at the last block; all-layer 462 vs exact 11.82), and the
linear predictor matters more (+0.2–1.7% per layer, +0.6% at the last block). The linear predictor
is far more accurate for full-attention blocks (proj err 0.149) than linear-attention blocks
(0.253), yet linear-attention blocks are the more tolerant at mid depth (KL 0.004–0.007 vs
0.009–0.025 for identity), consistent with the recurrent state absorbing input error.

### 4.6 Precision: exact correction survives weight-only quantisation, not activation quantisation

Source: `results/processed/Qwen3.5-4B-Base_{int8_weight_only,nvfp4_w4a16_emulated,nvfp4_w4a4_emulated}_approx_quality.*`.
The exact correction is applied through each projection's own kernel, `ỹ = P(N(ĥ)) + (P(e) − b)`,
and compared with that variant's own exact path. NVFP4 is emulated (`src/ple/quant`: E2M1 values,
E4M3 per-16 scales, FP32 global scale, dequantised BF16 matmul) because no NVFP4 kernel build
exists for this platform.

| Variant | Exact ppl | Corrected KL (10 boundary sets) | Top-1 agreement |
|---|---|---|---|
| BF16 | 11.824 | 0.0002–0.0006 | 98.8–99.6% |
| INT8 W8A16 (torchao) | 11.820 | 0.0002–0.0006 | 98.6–99.5% |
| NVFP4 W4A16 (emulated) | 12.383 | 0.0002–0.0006 | 98.7–99.6% |
| NVFP4 W4A4 (emulated) | 13.055 | 0.009–0.061 | 86.6–93.4% |

Weight-only quantisation keeps the projection linear in its input, so exact reconciliation stays at
the BF16 reformulation floor at both 8 and 4 bits. Quantising activations quantises `N(ĥ)` and `e`
separately; the "exact" correction then sits 15–100× above the floor and is no better than
uncorrected linear speculation at late layers (boundary 26: 0.020 vs 0.022). Because NVFP4 kernels
in Transformers are W4A16 for decode but W4A4 for prefill, exact correction would hold in decode
and fail in prefill. Any exact-reconciliation scheme on activation-quantised hardware paths needs
a higher-precision correction path or recomputation.

**On a vendor checkpoint.** Ornith-1.5-9B (Qwen3.5 architecture, d = 4096) in BF16, INT8 W8A16,
and NVIDIA's ModelOpt release, whose attention projections are FP8 W8A8 with static input scales
(MLPs NVFP4 W4A16). Exact correction through the module: BF16 KL 0.0002–0.0011 per boundary
(floor), INT8 0.0002–0.0009 (at floor), ModelOpt 0.0008–0.0091, i.e. 3–10× above the model's own
floor, worst at early boundaries; 6–10× when all boundaries are reconciled. The emulated and
vendor-checkpoint results agree in direction and shape; FP8 activations break exactness less than
FP4.

**At 27B.** Qwen3.8-27B with Red Hat's INT4 W4A16 checkpoint (kept packed on the GPU and dequantised
per layer, `ple.quant.packed`): exact correction through the module stays at the floor (KL
0.0001–0.0006, also with all 63 boundaries reconciled). Per layer this is the most tolerant model
tested (uncorrected identity ≤ +1% except the last block; linear ≤ +0.3%), yet all-layer identity
speculation collapses hardest (perplexity 234k), and all-layer linear costs +13%.

### 4.7 Further down the stack and further ahead

Source: `results/processed/*_{depth_start,lookahead_quality,lookahead_proj}.csv`. Uncorrected,
BF16. **Start depth (cumulative).** Keeping blocks before `s` and the last two blocks exact and
speculating everything in between, the linear predictor stays within 1% perplexity for 9 of 36
blocks on Qwen3-4B (s = 24, +0.9%) and 3 of 32 on Qwen3.5-4B (s = 26, +0.8%); from s = 16 the cost
is +4.1% / +5.5%, and from s = 0 +16% / +29%. Cost accumulates roughly per speculated block and is
far higher per block early in the stack. **Lookahead.** Speculating block l+k from h_l, the
projection error roughly doubles from k = 1 to 8 (linear 0.18 → 0.37 on Qwen3), but the
single-block quality cost stays small (median +1.5% / +1.6% at k = 8). Far lookahead of a few
blocks is cheap in quality; speculating many blocks is not.

## 5. Discussion (outline)

- Where PLE's assumptions break: unstructured errors, norm amplification, compounding across depth,
  and no idle GPU resources to overlap into on a single consumer GPU.
- Where it might still work: memory-bound decode with weights of block l+1 prefetched; multi-GPU
  pipeline settings with idle devices; architectures with explicit parallel branches.
- Relation to parallel-block architectures: uncorrected late-layer speculation is equivalent to
  feeding block l+1's attention with block l's input — a post-hoc parallel attention block, i.e.
  LP's attention branch applied at one boundary.
- Agreement with SNLP's negative result: on off-the-shelf models, depth-parallel prediction plus
  correction matches quality but does not beat sequential execution without training.
- RidgeFill shows a ridge predictor *can* pay when it is used to **skip** work (late-layer KV
  fill during long prefill) rather than to overlap it; the difference is that skipping removes
  FLOPs, while PLE-style overlap only moves them.

## 6. Related work (summary; full review in `docs/research_review.md`, BibTeX in `references.bib`)

| Work | Mechanism | Overlap with this study |
|---|---|---|
| Layer Parallelism (Calvo González et al., arXiv 2502.02790) | Pairs of layers; both attentions read the shared input; no retraining; multi-GPU | Uncorrected single-boundary speculation is its attention branch; also evaluates Qwen3-4B |
| CQIL (Zou et al., ACL 2024) | Groups of layers on the group's input, one GPU per layer, cheap "bypass" correction | Identity predictor + partial correction |
| EasySpec (Wu et al., arXiv 2502.02493) | Draft-model attention layers in parallel on a shared input; KV recomputed exactly after each round | Same schedule; exactness by discard-and-recompute |
| InfiniGen (Lee et al., OSDI 2024) | Next layer's query weights applied to the current normed input, for KV prefetch | Our QKV launch with the identity predictor |
| RidgeFill (Johnston & Smith, arXiv 2611.09709) | Closed-form per-layer ridge map predicts late-layer inputs; KV filled from prediction to skip prefill | Our linear predictor; used to skip, not overlap |
| SNLP (Han et al., arXiv 2605.17842) | All layer inputs predicted, structured Newton corrections | General predict-then-correct; no gain on untrained models |
| Song et al. (ICML 2021) | Jacobi / Gauss-Seidel over depth, exact at convergence | Our exact correction is one Jacobi step |

## 7. Limitations

One model so far, one dataset, sequence length 512 for quality metrics, teacher-forced
evaluation only, no task benchmarks, single shared GPU without locked clocks, eager/graph PyTorch
only (no custom kernels). Thresholds were proposed but tolerances not pre-registered before the
first results; all current results are labelled exploratory.

## 8. Reproducibility

Every script records model revision, dataset revision, token counts, GPU, and library versions in
a JSON next to its CSV. Commands:

```
python scripts/verify_model.py --config configs/primary_qwen3_4b.yaml
python scripts/predictability_study.py --config configs/primary_qwen3_4b.yaml --train-seqs 256 --eval-seqs 64
python scripts/error_structure_study.py --config configs/primary_qwen3_4b.yaml
python scripts/approx_quality_study.py --config configs/primary_qwen3_4b.yaml
python scripts/bench_overlap.py --config configs/primary_qwen3_4b.yaml --boundary 26 --tokens 16 128 512 2048 8192 --repeats 200 [--graphs]
```
