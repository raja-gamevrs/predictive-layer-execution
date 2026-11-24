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

Single model, single dataset, sequence length 512, batch 1. No timing. Gemma/Qwen3.5 contrast
and token-position breakdown pending.

## 2026-11-23 — H2/H3 first pass: structure of the speculation error, Qwen3-4B-Base

Script: `scripts/error_structure_study.py`. Speculate the next block's `q_proj` from `N(ĥ)`; when
the exact state arrives, `e = N(h) − N(ĥ)` is cheap, and `W e` is the exact fix. Ridge predictors
fit on train chunks 0–255, error bases (PCA of `e`) on unseen train chunks 256–383, scored on 64
test chunks. Cost is the correction's MACs as a fraction of one exact `q_proj` (2560 → 4096).
Raw: [`results/processed/Qwen3-4B-Base_error_structure.csv`](../results/processed/Qwen3-4B-Base_error_structure.csv).

Median `q_proj` relative error over layers 0–34 after correction:

| Correction | Cost | Identity | Linear r256 | Linear full |
|---|---|---|---|---|
| none | 0 | 0.260 | 0.221 | 0.178 |
| fixed basis r = 64 | 0.04 | 0.194 | 0.180 | 0.151 |
| fixed basis r = 256 | 0.16 | 0.153 | 0.145 | 0.123 |
| fixed basis r = 512 | 0.33 | 0.126 | 0.119 | 0.101 |
| per-token top 5% channels | 0.05 | 0.198 | 0.168 | 0.136 |
| per-token top 10% channels | 0.10 | 0.168 | 0.144 | 0.116 |
| per-token top 25% channels | 0.25 | 0.108 | 0.095 | **0.077** |
| exact fix, worst 25% tokens | 0.25 | 0.211 | 0.180 | 0.139 |
| exact fix, worst 50% tokens | 0.50 | 0.162 | 0.137 | 0.106 |

### Findings

1. **The error is not low-rank.** A 256-dimensional basis holds only 38–65% of the error energy
   (train), and a rank-512 correction at a third of the projection cost still leaves ~10% error.
2. **Channel concentration is the strongest structure, and it is moderate.** The top 25% of
   channels per token carry ~80% of the correction energy (0.178 → 0.077), but picking a
   different 25% per token is an irregular gather that is unlikely to beat a dense GEMM on GPU.
3. **No token concentration.** Errors are spread evenly over tokens; fixing the worst half removes
   only ~65% of the error energy.
4. **Per layer, the picture is the same everywhere:** best case (layers 27–34, full linear +
   top 25%) still leaves 6.4–7% error.

### Gate assessment (exploratory — fidelity tolerance not yet pre-registered)

- **Exact PLE fails the correction-cost gate here.** Any correction that is cheap leaves 8–20%
  projection error; reaching exactness costs a full projection. Adding the predictor (3–63% of a
  projection) and the speculative projection itself, total work exceeds the exact path.
- **Approximate PLE is still open, but unproven.** Whether ~8–15% error at one projection is
  tolerable depends on its effect on the model's output, which has not been measured. That is the
  next test: logit KL / top-1 agreement when a single layer's QKV is speculated and
  approximately corrected.
- One model, one projection, one dataset. Repeat on Qwen3.5-4B-Base (hybrid linear attention),
  Ornith-1.5-9B and Qwen3.8-27B before drawing a model-general conclusion.

## 2026-11-23 — Output quality of approximate PLE, Qwen3-4B-Base

Script: `scripts/approx_quality_study.py`. In the real forward pass, block l+1's `q/k/v_proj`
outputs are replaced by `W N(ĥ)` plus a correction; everything downstream runs on the result.
Compared with the exact model on 16 held-out test sequences (8,160 scored tokens). Exact
perplexity 11.400. Position 0 kept exact. Raw:
[`results/processed/Qwen3-4B-Base_approx_quality.csv`](../results/processed/Qwen3-4B-Base_approx_quality.csv).

| Speculated at | Predictor | Correction (cost) | KL (nats/token) | Top-1 agreement | Perplexity |
|---|---|---|---|---|---|
| any one layer | — | full (exactness check) | 0.0005–0.0011 | 98.1–99.2% | 11.39–11.41 |
| one layer, 2–22 | identity | none (0) | 0.011–0.016 | 93.2–95.0% | 11.42–11.54 |
| one layer, 2–22 | linear | none (0) | 0.005–0.009 | 95.0–96.6% | 11.38–11.46 |
| one layer, 26 or 30 | identity | none (0) | 0.003–0.004 | 97.0–97.5% | 11.43 |
| one layer, 34 | identity | none (0) | 0.050 | 92.9% | 11.89 |
| one layer, any | linear | top 25% channels (0.25) | 0.0012–0.0023 | 97.5–98.3% | 11.39–11.41 |
| **every layer** | identity | none | 1.61 | 46.9% | 54.0 |
| every layer | linear | none | 0.217 | 77.8% | 13.25 |
| every layer | linear | rank-256 basis (0.16) | 0.081 | 86.1% | 11.91 |
| every layer | linear | top 25% channels (0.25) | 0.027 | 92.1% | 11.50 |

### Findings

1. **"Exact" correction is not bit-exact in BF16.** Computing `W N(ĥ) + W e` instead of `W N(h)`
   changes rounding enough to flip 1–2% of argmax tokens (KL ≈ 0.001). This is the noise floor
   for every other number here, and any exactness claim must state its precision.
   *Follow-up (same day, rerun):* the exact model is bit-deterministic run to run (KL 0.0, top-1
   100%), and forming `e` plus accumulating `W e` in FP32 (`full_fp32`) leaves the floor unchanged
   (KL 0.0004–0.0012, top-1 98.1–99.2%). The floor therefore comes from rounding the projection
   output to BF16: ŷ is rounded once, the exact `W N(h)` along a different path, so the two
   differ by about one BF16 ulp. A one-ulp perturbation of a single layer's q/k/v already flips
   ~1.7% of near-tie argmax tokens, and the effect does not grow when every layer is perturbed
   (KL 0.0012). Any reformulated BF16 computation (different GEMM tiling, fused kernel) would sit
   on the same floor, so "exact" here means "within the BF16 reformulation floor".
2. **A single speculated layer is cheap in quality.** Even free, uncorrected speculation from
   h_l raises perplexity by ≤ 1.2% at layers 2–30, and by only ~0.25% at layers 26–30.
3. **Errors compound across layers.** Speculating every layer with no correction collapses the
   model (perplexity 54). With the linear predictor plus a top-25% channel correction it stays
   usable (+0.9% perplexity, 92% top-1), but that correction is an irregular gather.
4. **The plausible regime is selective:** a few late layers, identity or linear prediction, little
   or no correction. Speculating block l+1's QKV from h_l in parallel with block l's work is,
   however, close to existing layer-parallel ideas (e.g. parallel attention/MLP blocks,
   "Leveraging the true depth of LLMs"); novelty has to be checked against them before claiming it.

### What this does not show

No timing (see next entry). The quality numbers are prefill-style teacher forcing on WikiText-2
only; no task benchmarks, no decode.

## 2026-11-23 — H4 first pass: does QKV overlap shorten the critical path? Qwen3-4B-Base, RTX 5090

Script: `scripts/bench_overlap.py --boundary 26 --tokens 16 128 512 2048 8192 --repeats 200`,
eager and `--graphs`. Block 26 run on its real input with the exact mask / rotary arguments
captured from a model forward pass; block 27's norm + q/k/v launched on a second CUDA stream
from `N(ĥ)` while block 26 runs on the main stream. Raw:
`results/processed/Qwen3-4B-Base_bench_overlap_l26_{eager,graphs}.json`.

| Tokens | Sequential, graphs (ms) | Overlap identity | Overlap linear | Upper bound | Seq. CV |
|---|---|---|---|---|---|
| 16 | 0.335 | +4.6% | +4.9% | 16.5% | 0.10 |
| 128 | 0.324 | −0.6% | −2.8% | 17.4% | 0.13 |
| 512 | 0.910 | +0.5% | −2.3% | 19.6% | 0.07 |
| 2048 | 3.733 | +0.9% | −4.3% | 12.2% | 0.09 |
| 8192 | 21.473 | −0.7% | −4.3% | 11.7% | 0.05 |

Eager mode: −15.2% to +1.8% (identity), −15.8% to −1.7% (linear); CV 0.26–0.43 below 2k
tokens because CPU launch overhead dominates a ~1 ms block. *(The eager JSON was later overwritten
by the profiling rerun below, which measured −8.9% to +2.0% (identity) and −13.9% to +0.3%
(linear), same conclusion; the graphs JSON is unchanged.)*

### Findings

1. **No real overlap.** Of a 12–20% theoretical saving, at most ~5% appears, and only at 16 tokens,
   within noise. The linear predictor's extra GEMM makes overlap a net loss at ≥ 128 tokens.
2. **Baseline variance fails the 5% gate** at most sizes (shared desktop GPU, unlocked clocks), so
   any effect below ~5% is not resolvable on this setup.
3. Likely cause (not yet profiled): at small T both streams are bound by reading weights from DRAM
   and compete for the same bandwidth; at large T block 26's GEMMs already fill the SMs.

### Gate assessment (exploratory)

The overlap gate fails for single-block QKV speculation on this GPU. Combined with the quality
result, the best case is: a few late layers, identity prediction, no correction, saving at most the
QKV share (~12–17%) of each speculated block, i.e. ≲ 0.5% of a full forward per layer, and that
saving does not materialise in measurement. Next checks before closing this branch: an Nsight
timeline to confirm (non-)concurrency, and the same benchmark on Qwen3.5 / larger models.

### Follow-up: kernel concurrency profile (same day)

`scripts/bench_overlap.py --profile`: PyTorch profiler (CUPTI) over 5 eager runs of
`overlap_identity`; for each side-stream kernel, the time it overlaps any main-stream kernel.
Nsight Systems is not installed; this uses the same CUPTI kernel timestamps. The GPU was also
serving the desktop (5–26% utilisation from other processes). Raw:
`results/processed/Qwen3-4B-Base_bench_overlap_l26_eager.json` (field `concurrency_overlap_identity`).

| Tokens | Side-stream kernel time per run | Concurrent with main stream | Standalone QKV path | Eager net effect |
|---|---|---|---|---|
| 16 | 0.061 ms | 3% | — | −8.9% |
| 128 | 0.049 ms | 0% | — | −5.6% |
| 512 | 0.143 ms | 35% | — | −3.6% |
| 2048 | 0.528 ms | 47% | 0.506 ms (event-timed) | +2.0% |
| 8192 | 2.230 ms | 24% | 1.858 ms (event-timed) | −0.3% |

**Findings.** Below ~512 tokens the side-stream kernels run almost entirely *before* block l's
kernels start (launch-bound: the GPU is idle waiting for the CPU), so there is no concurrency to
exploit in eager mode. From 512 tokens up, 24–47% of the speculative QKV work does run
concurrently, but contention cancels it: at 8192 tokens the QKV kernels take 20% longer when
overlapped than alone, and block l slows correspondingly. Real concurrency exists; useful
concurrency does not, at this granularity on one GPU. This explains the H4 timing result.

## 2026-11-23 — Second architecture: Qwen3.5-4B-Base (hybrid linear attention), BF16

Same three studies, same data splits and settings as Qwen3-4B-Base. Qwen3.5-4B-Base has 32 blocks:
24 Gated-DeltaNet linear-attention blocks and a full-attention block every fourth position
(3, 7, …, 31). Linear-attention blocks read the normed residual through four projections
(`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`); all four are speculated. "proj" error is
measured at the first one (`in_proj_qkv`) for linear blocks and `q_proj` for full blocks. Linear
attention ran on the Transformers reference kernels (`flash-linear-attention` not installed), which
changes speed only. Raw: `results/processed/Qwen3.5-4B-Base_*`, linear-block quality runs in
`results/processed/qwen35_linear_blocks/`.

**Predictability (H1)**, median over boundaries:

| Predictor | Δrel | proj err, all | proj err, next block linear-attn (n = 23) | proj err, next block full-attn (n = 8) |
|---|---|---|---|---|
| identity | 1.000 | 0.341 | 0.351 | 0.336 |
| extrapolation | 1.411 | 0.461 | | |
| velocity α = 0.5 | 1.125 | 0.398 | | |
| linear (full) | 0.706 | 0.187 | 0.253 | 0.149 |
| linear r = 256 | 0.871 | 0.241 | 0.301 | 0.198 |

**Error structure (H2)**, median proj error over boundaries 0–30, linear predictor: none 0.187;
basis r = 512 (cost 0.26) 0.123; top 25% channels (0.25) 0.090; worst 50% tokens (0.5) 0.109.

**Output quality.** Exact perplexity 11.824; exact rerun KL 0.0 (deterministic).

| Boundary (speculated block, type) | identity, none | linear, none | linear, top 25% | full |
|---|---|---|---|---|
| 1 (2, linear) | +4.9% ppl, KL 0.059 | +1.7%, 0.026 | | KL 0.0005 |
| 2 (3, full) | +4.1%, 0.058 | +0.9%, 0.019 | +0.0%, 0.004 | 0.0005 |
| 5–25 (linear blocks) | +0.3–1.4%, KL 0.004–0.017 | +0.3–0.9%, 0.003–0.012 | | 0.0003–0.0005 |
| 6–26 (full blocks) | +1.0–3.2%, KL 0.009–0.032 | +0.2–1.3%, 0.002–0.010 | +0.0–0.4%, 0.001–0.002 | 0.0003–0.0005 |
| 29 (30, linear) | +4.5%, 0.041 | +0.6%, 0.006 | | 0.0002 |
| 30 (31, full, last block) | **+20.9%, 0.180** | +0.6%, 0.008 | +0.1%, 0.002 | 0.0002 |
| all 31 boundaries | 462 ppl (collapse) | +31%, 0.288 | +2.7%, 0.036 | 0.0006 |

### Findings (Qwen3.5 vs Qwen3)

1. **The H1/H2 picture replicates.** Extrapolation is again worse than identity (cos(Δ̂, Δ) is now
   slightly *negative*, −0.08); a full-rank linear map again explains ~half the update energy
   (Δrel 0.71); the error is again neither low-rank nor token-concentrated.
2. **Qwen3.5 is less tolerant of uncorrected speculation.** Identity costs +1–4% perplexity per
   layer (vs +0.0–1.2% on Qwen3), and speculating the last block costs +21%. All-layer identity
   collapses harder (462 vs 54).
3. **The linear predictor matters more here.** It cuts the single-layer cost to +0.2–1.7% and
   rescues the last block (+0.6%). Prediction is much better for full-attention blocks (proj err
   0.149) than for linear-attention blocks (0.253).
4. **Linear-attention blocks are the more tolerant ones at mid depth,** despite their larger
   projection error: identity at mid-depth boundaries 13–26 costs KL 0.004–0.007 when the
   speculated block is linear-attention (blocks 14, 18, 22, 26) vs 0.009–0.025 when it is
   full-attention (blocks 15, 19, 23, 27).
   The recurrent Gated-DeltaNet state appears to absorb input error better than softmax attention.
5. **Reformulation floor replicates** (KL 0.0002–0.0006, top-1 98.8–99.6%).

## 2026-11-23 — Precision: does exact correction survive quantisation? Qwen3.5-4B-Base

Script: `scripts/approx_quality_study.py --corrections none full_module`, configs
`configs/qwen35_4b_base_{int8,nvfp4_w4a16,nvfp4_w4a4}.yaml`. Each variant is quantised at load
from the same BF16 snapshot; predictors are fit on that variant's own activations; every metric
compares against that variant's own exact path. `full_module` applies the exact correction through
the projection's own kernel: `ỹ = P(N(ĥ)) + (P(e) − b)`. INT8 = torchao `Int8WeightOnlyConfig`
(per-channel INT8 weights, BF16 activations). NVFP4 = emulation in `src/ple/quant` (E2M1 values,
E4M3 scale per 16 elements, FP32 global scale; dequantised BF16 matmul), with activations either
left in BF16 (W4A16) or fake-quantised the same way (W4A4); applied to every Linear inside the
decoder blocks. No NVFP4 kernel build exists for Windows / PyTorch 2.14 / CUDA 13.0, and
Transformers' own NVFP4 path is W4A4 at prefill and W4A16 at decode.
Raw: `results/processed/Qwen3.5-4B-Base_{int8_weight_only,nvfp4_w4a16_emulated,nvfp4_w4a4_emulated}_approx_quality.*`.

| Variant | Exact ppl (vs BF16 11.824) | `full_module` KL, range over 10 boundary sets | top-1 agreement | Exact rerun |
|---|---|---|---|---|
| BF16 (`full`) | 11.824 | 0.0002–0.0006 | 98.8–99.6% | deterministic |
| INT8 W8A16 | 11.820 (−0.03%) | 0.0002–0.0006 | 98.6–99.5% | deterministic |
| NVFP4 W4A16 | 12.383 (+4.7%) | 0.0002–0.0006 | 98.7–99.6% | deterministic |
| NVFP4 W4A4 | 13.055 (+10.4%) | **0.009–0.061** | **86.6–93.4%** | deterministic |

Uncorrected speculation (identity / linear) costs about the same in every variant as in BF16
(e.g. boundary 26: identity KL 0.012–0.014, linear 0.0020–0.0022 for BF16, INT8 and W4A16).

### Findings

1. **Weight-only quantisation preserves exact correction** to the BF16 reformulation floor, at both
   8 and 4 bits: the projection stays linear in its input, so `P(N(ĥ)) + P(e) = P(N(h))` up to
   rounding.
2. **Activation quantisation breaks it.** With W4A4, `N(ĥ)` and `e` are quantised separately (each
   with its own per-16 scales), so their sum is not the quantisation of `N(h)`. The "exact"
   correction is 15–100× above the floor and flips 7–13% of top-1 tokens. It is worst early
   (boundary 2: KL 0.060) and smallest at the last block (0.009).
3. **Under W4A4 the correction buys little.** At boundary 26, linear speculation without any
   correction (KL 0.022) is about as close to the exact path as the "exact" correction (0.020).
   Exact-reconciliation designs therefore need a correction path computed at higher activation
   precision than the speculative path, or must recompute.
4. **Practical consequence for NVFP4 deployments:** with a kernel that is W4A16 for decode and W4A4
   for prefill, exact correction would hold during decode but not during prefill, the opposite of
   where PLE's overlap has the most compute to hide.
5. These are emulated numerics (dequantise, then BF16 matmul); a real NVFP4 kernel accumulates
   differently, but the non-linearity comes from quantising activations, which a real W4A4 kernel
   does too. To confirm on real NVFP4 checkpoints (Ornith-1.5-9B-NVFP4 is W4A16, nvidia
   Qwen3.8-27B-NVFP4 is W4A4) once loaded.

## 2026-11-23 — Precision on real weights: Ornith-1.5-9B in BF16, INT8 and NVIDIA ModelOpt

Same script (`--corrections none full_module`, plus `full` for BF16), same splits. Three variants of
one set of weights (Qwen3.5 architecture, d = 4096, 32 blocks):
- **BF16**: `ornith-ai/Ornith-1.5-9B` `489cb97`.
- **INT8 W8A16**: the same snapshot quantised at load with torchao.
- **ModelOpt**: the vendor checkpoint `ornith-ai/Ornith-1.5-9B-NVFP4` `155f200`, which is
  MIXED_PRECISION: every attention input/output projection (`q/k/v/o_proj`, `in_proj_qkv`,
  `in_proj_z`, `out_proj`) is FP8 W8A8 with a static per-tensor input scale; MLPs and `lm_head` are
  NVFP4 W4A16; `in_proj_a/b`, norms and embeddings are BF16. Loaded with `ple.quant.modelopt`, which
  applies the checkpoint's own scales (weights dequantised; activations quantised to E4M3 with the
  stored `input_scale`). So the projections PLE speculates are activation-quantised here.

Raw: `results/processed/{Ornith-1.5-9B,Ornith-1.5-9B_int8_weight_only,Ornith-1.5-9B-NVFP4}_approx_quality.*`.

| Variant | Exact ppl | Exact fix through the module: KL per single boundary (2…30) | all 31 boundaries | Top-1 (single) |
|---|---|---|---|---|
| BF16 | 12.105 | 0.0002–0.0011 | 0.0014–0.0015 | 98.5–99.5% |
| INT8 W8A16 | 12.094 | 0.0002–0.0009 | 0.0011–0.0012 | 98.5–99.5% |
| ModelOpt (FP8 W8A8 attention) | 12.764 | 0.0008–0.0091 | 0.0090–0.0156 | 96.0–98.6% |

Per boundary, ModelOpt / BF16 KL ratio: 10× / 6× (identity / linear) at boundary 2, 5.9× at 6, ~4×
at 10–14, 3.3–3.4× at 18–26, 4× at 30; 6–10× for all boundaries at once.

Uncorrected speculation on the 9B model is far more damaging than on the 4B models: all-boundary
identity reaches perplexity ~12,000 in every variant (BF16 12,277; INT8 12,143; ModelOpt 10,990),
linear all-boundary 15.9 / 15.9 / 16.8.

### Findings

1. **Replicates the emulated result on a vendor checkpoint.** Weight-only INT8 keeps exact
   reconciliation at the model's own BF16 floor; FP8 W8A8 attention projections put it 3–10× above
   the floor (6–10× when every layer is reconciled).
2. **Smaller break than emulated FP4 activations** (15–100× on Qwen3.5), as expected from FP8's
   larger mantissa; same shape: worst early, smallest late.
3. **`full` and `full_module` agree exactly in BF16** (identical KL and ppl to 4 decimals), so the
   module-path correction introduces no error of its own.
4. **Scale makes uncorrected speculation worse, not better**: the 9B model collapses ~25× harder
   than Qwen3.5-4B under all-layer identity speculation.

## 2026-11-23 — Scale: Qwen3.8-27B, INT4 weight-only (W4A16)

Config `configs/qwen38_27b_int4.yaml`: `RedHatAI/Qwen3.8-27B-INT4` `71dface` (compressed-tensors
pack-quantized, symmetric INT4, group 128, every Linear in the decoder except `linear_attn.in_proj_a/b`;
BF16 embeddings, norms, lm_head). 64 blocks (48 Gated-DeltaNet, 16 full attention), d = 5120. A BF16
copy (~54 GB) does not fit the 5090, and this Transformers / compressed-tensors version decompresses
INT4 to BF16 on first forward, so the model is loaded with `ple.quant.packed`: weights stay packed on
the GPU and each layer dequantises to BF16 per call (peak 19.5 GB). Script:
`approx_quality_study.py --corrections none full_module --fit-group 8 --predictor-dtype bfloat16`
(ridge fit 8 layers per pass; fitted maps stored in BF16). Same data splits. Exact perplexity 9.335;
exact rerun deterministic. Raw: `results/processed/Qwen3.8-27B-INT4_approx_quality.*`. Qwen3.8-27B is a
post-trained model (no base checkpoint is published).

| Boundaries | identity, none | linear, none | exact fix through the module (full_module) |
|---|---|---|---|
| single, 2–58 (15 of them) | KL 0.0028–0.0247, −0.04% to +1.0% ppl | KL 0.0013–0.0080, −0.04% to +0.3% | KL 0.0002–0.0006, top-1 98.7–99.6% |
| single, 62 (last block) | KL 0.099, +10.3% | KL 0.0070, +0.6% | KL 0.0001–0.0002 |
| all 63 | perplexity 234,355 (collapse) | KL 0.179, +13.4%, top-1 82.2% | KL 0.0006, top-1 98.7–98.9% |

### Findings

1. **INT4 weight-only keeps exact reconciliation at the floor at 27B**, at every boundary and with all 63
   reconciled at once (KL ≤ 0.0006), extending the INT8 / NVFP4-W4A16 result to a third weight-only format
   and a 7× larger model.
2. **Per layer, the 27B model is the most tolerant model tested**: uncorrected identity costs ≤ 1% at
   every boundary but the last, and the linear predictor ≤ 0.3% (0.6% at the last block). Boundaries
   10–58 are all cheap.
3. **But all-layer identity collapses hardest** (234k vs 12k for Ornith-9B, 462 for Qwen3.5-4B): 63
   small per-layer errors compound. With the linear predictor, all-layer speculation costs +13%, the
   smallest relative cost of any model (Qwen3-4B +16%, Qwen3.5-4B +31%, Ornith-9B +32%).
4. **The last block remains the sensitive one** (+10% with identity, consistent with every other model
   and with LP's sequential tail).

## 2026-11-23 — Further down the stack, and further ahead (BF16, both 4B models)

Script: `scripts/depth_sweep_study.py` (defaults: fit 256 train chunks per lookahead, quality on 16
test chunks, projection error on 32 test chunks). All speculation uncorrected. Raw:
`results/processed/{Qwen3-4B-Base,Qwen3.5-4B-Base}_{depth_start,lookahead_quality,lookahead_proj}.csv`.

**Part A — start depth, cumulative.** Blocks before `s` stay exact, and every boundary from `s`
onward is speculated (k = 1), keeping the last two blocks exact (`tail = 2`; `tail = 0` is in the
CSV and is uniformly worse, mostly for identity).

| Start s | Qwen3-4B: blocks, identity ppl, linear ppl (exact 11.400) | Qwen3.5-4B: blocks, identity, linear (exact 11.824) |
|---|---|---|
| 0 | 33, 44.9, 13.22 (+16%) | 29, 249.7, 15.30 (+29%) |
| 8 | 25, 17.7, 12.68 (+11%) | 21, 16.35, 12.94 (+9.4%) |
| 16 | 17, 12.71, 11.87 (+4.1%) | 13, 13.69, 12.47 (+5.5%) |
| 20 | 13, 12.29, 11.71 (+2.7%) | 9, 12.91, 12.24 (+3.6%) |
| 24 | 9, 11.74, 11.50 (+0.9%) | 5, 12.38, 12.01 (+1.6%) |
| 28 | 5, 11.59, 11.46 (+0.5%) | 1, 11.91, 11.86 (+0.3%) |

**Part B — lookahead k.** Block l+k's attention inputs speculated from h_l, one source at a time
(sources 2, 6, 10, … with l + k < L). Projection error is the median over all valid l.

| k | Qwen3: proj err identity / linear | Qwen3: median ppl identity / linear (max) | Qwen3.5: proj err identity / linear | Qwen3.5: median ppl identity / linear (max) |
|---|---|---|---|---|
| 1 | 0.265 / 0.178 | 11.470 / 11.413 (11.461) | 0.342 / 0.190 | 12.171 / 11.893 (11.973) |
| 2 | 0.342 / 0.234 | 11.531 / 11.426 (11.486) | 0.425 / 0.253 | 11.992 / 11.944 (11.996) |
| 3 | 0.396 / 0.266 | 11.535 / 11.451 (11.550) | 0.477 / 0.295 | 11.983 / 11.914 (11.958) |
| 4 | 0.456 / 0.293 | 11.731 / 11.481 (11.564) | 0.501 / 0.328 | 12.078 / 12.009 (12.027) |
| 6 | 0.550 / 0.334 | 11.730 / 11.529 (11.570) | 0.575 / 0.382 | 12.102 / 12.010 (12.213) |
| 8 | 0.614 / 0.366 | 11.897 / 11.574 (11.642) | 0.607 / 0.420 | 12.187 / 12.017 (12.156) |

(Qwen3.5 k = 1 sources 2, 6, … all target full-attention blocks; other k mix block types, so its
k = 1 identity row is not directly comparable to k ≥ 2.)

### Findings

1. **Keeping the early layers exact helps a lot, but the cost is cumulative.** Perplexity cost
   grows roughly with the number of speculated blocks: on Qwen3-4B with the linear predictor,
   about +0.1% per block over the last 9, rising to about +0.25% per block over the last 17; early
   blocks are much more expensive per block (start 0 vs start 8: 8 extra blocks cost +5 points).
2. **The budget is small.** For ≤ 1% perplexity: 9 blocks (of 36) on Qwen3-4B (+0.9%), 3 (of 32) on
   Qwen3.5-4B (+0.8%; 5 blocks cost +1.6%), both linear-predicted, last two exact. Identity
   costs 3–4× more at the same depth (start 24: +3.0% vs +0.9% on Qwen3, +4.7% vs +1.6% on Qwen3.5).
3. **Predicting further ahead degrades gracefully.** Projection error roughly doubles from k = 1 to
   k = 8 (linear: 0.18 → 0.37), yet speculating one block 8 ahead costs only +1.5% (Qwen3) / +1.6%
   (Qwen3.5) median perplexity with the linear predictor. Far lookahead is the regime RidgeFill
   exploits (predicting late-layer inputs from an early split point) and is where more work could
   in principle run in parallel.
4. **The linear predictor's advantage grows with distance** (identity / linear proj err ratio
   1.5 at k = 1, 1.7 at k = 8 on Qwen3).

## 2026-11-23 — H4 rerun on an idle GPU: overlap does help at decode-sized inputs (Qwen3-4B-Base)

The user stopped using the PC, so the GPU was idle (0% utilisation, 0.7 GB in use before each run;
ComfyUI loaded but inactive). Same benchmark, boundary 26 → 27, CUDA graphs.

**Single-replay timing (as before)**, `results/processed/quiet_gpu/Qwen3-4B-Base_bench_overlap_l26_graphs.json`
and `quiet_gpu/small/…`: sequential CV fell to 0.7–4.8% (inside the 5% gate at every size). From 128
tokens up the result is unchanged (identity −2.3% to +1.1%). Below 128 tokens it was not reproducible:
at 16 tokens two runs gave +9.9% and +5.2%. Each path takes 20–70 µs, so launch gaps and event
overhead around a single graph replay dominate.

**Batched timing** (`--inner 50`: each sample times 50 back-to-back replays and divides; a real decode
step replays one graph for the whole model, so launch overhead is amortised the same way), 100 samples,
`results/processed/quiet_gpu/batched/Qwen3-4B-Base_bench_overlap_l26_graphs.json`:

| Tokens | Sequential (ms) | Most possible | Overlap, identity | Overlap, linear | Sequential CV |
|---|---|---|---|---|---|
| 1 | 0.245 | 15.0% | +6.9% | +5.4% | 0.009 |
| 4 | 0.267 | 15.1% | +10.5% | +7.4% | 0.009 |
| 16 | 0.330 | 18.3% | +15.7% | +13.7% | 0.007 |
| 32 | 0.274 | 15.9% | +9.7% | +4.9% | 0.009 |
| 64 | 0.294 | 16.3% | +6.6% | +2.4% | 0.008 |
| 128 | 0.342 | 15.5% | +4.2% | −1.3% | 0.008 |
| 512 | 1.011 | 14.1% | +2.8% | −2.4% | 0.004 |
| 2048 | 4.046 | 12.3% | +0.7% | −4.3% | 0.002 |

### Fused speculative projections (new: `scripts/bench_fused_qkv.py`)

With the identity predictor, block l+1's normed input is h_l / rms(h_l) times its own gain, so its
q/k/v can be computed as extra output columns of block l's own q/k/v GEMM after folding each block's
gain into its weights: one GEMM, no second stream. Compared with the production layout (one packed
QKV GEMM and a single-kernel RMSNorm per block), batched timing, two independent runs
(`quiet_gpu/batched_run{1,2}/Qwen3-4B-Base_bench_fused_qkv_l26.json`), saving per speculated block
as a share of block l's full forward time, k = blocks speculated:

| Tokens | k = 1 (run 1 / run 2) | k = 2 | k = 4 |
|---|---|---|---|
| 1 | +2.0% / +1.9% | +2.9% / +2.6% | +2.2% / +2.3% |
| 4 | +4.7% / +4.7% | +5.5% / +5.4% | +3.0% / +2.9% |
| 16 | +3.9% / +4.3% | +3.1% / +3.1% | +2.3% / +2.1% |
| 64 | +1.9% / +2.4% | +4.8% / +4.6% | +2.9% / +2.8% |
| 128 | +1.2% / +1.3% | +4.8% / +4.5% | +3.4% / +3.1% |
| 512 | +2.9% / +2.7% | +1.8% / +1.7% | +1.8% / +1.8% |
| 2048 | −0.4% / −0.3% | +2.6% / +2.6% | +2.1% / +2.1% |
| 8192 | +0.9% / +0.8% | +0.8% / +0.8% | +0.6% / +0.6% |

Fused outputs match the separate path to 2.7–3.3 × 10⁻³ relative error (rounding of the folded
weights in BF16), far below the speculation error itself. Unit test: `tests/test_fused_qkv.py`.

### Findings

1. **The earlier H4 verdict was partly a measurement artefact.** At decode-sized inputs (1–64 tokens)
   a second stream recovers 40–86% of the possible saving for one speculated block (+6.9% to +15.7%
   with identity). The single-replay benchmark could not see this. From 512 tokens up the earlier
   conclusion holds: block l's GEMMs fill the GPU and overlap gives ≤ 2.8%.
2. **The linear predictor's extra GEMM costs 1–5 points at small T and makes overlap a loss from 128
   tokens up**, as before.
3. **Fusion is reproducible but smaller than overlap at small T** (+1.9% to +5.5% per block); it is the
   only variant that does not lose at any size, and it needs the identity predictor.
4. **Per block is not end to end.** A rough estimate, not a measurement: at 16 tokens, linear overlap on
   the 9 blocks that fit the 1% perplexity budget (start 24, last two exact) would save about
   9 × 0.045 ms ≈ 0.4 ms of a ≈ 10 ms forward, about 4%; at 1 token about 1.6%. Identity overlap fits
   only 1–2 blocks in the budget. Two caveats cut against this: the baseline here is Hugging Face's
   block (three q/k/v GEMMs, multi-kernel norms), which is slower than a production block, and the
   1% budget was measured in teacher-forced prefill, not decode. A measured decode test is needed (H5).

## 2026-11-23 — Phase II, D1/D3: sensitivity atlas and static allocation (Qwen3-4B, Qwen3.5-4B, BF16)

Plan: `docs/phase2_plan.md`. Scripts: `scripts/sensitivity_atlas.py`, `scripts/allocation_study.py`.
Linear stand-ins fitted on WikiText-2 train chunks 0–255; atlas and allocation measured on train
chunks 400–415 (8,160 scored tokens); greedy sets re-measured on train chunks 600–615 (holdout). The
test split was not used. Position 0 kept exact. Perplexity differences under about ±0.7% are within
the noise of 8,160 tokens (one attention stand-in measured −0.67%). Exploratory: the Phase II gate is
proposed, not yet fixed.

**Atlas.** Every block, its attention and its MLP, each removed (`skip`) or replaced by a ridge map
(`linear`, a d × d GEMM). Raw: `results/processed/{Qwen3-4B-Base,Qwen3.5-4B-Base}_sensitivity_atlas.{csv,json}`.

| Model | Operation | Compute removed per op | Best single op (ppl) | Median, blocks 1 to L−2 | Ops ≤ 1% ppl | Ops with KL ≤ 0.01 |
|---|---|---|---|---|---|---|
| Qwen3-4B | block skip / linear | 2.8% / 2.6% | +3.4% (L16) / +1.5% (L2) | +11.0% / +7.9% | 0 / 0 | 0 / 0 |
| Qwen3-4B | attention skip / linear | 0.76% / 0.59% | +0.8% (L10) / −0.7% (L2, noise) | +2.6% / +0.6% | 1 / 25 | 0 / 0 |
| Qwen3-4B | MLP skip / linear | 2.0% / 1.8% | +2.2% (L16) / +1.3% (L19) | +8.1% / +6.5% | 0 / 0 | 0 / 0 |
| Qwen3.5-4B | block skip / linear | 3.2% / 3.0% | +2.4% (L12) / +1.5% (L12) | +11.6% / +7.8% | 0 / 0 | 0 / 0 |
| Qwen3.5-4B | attention skip / linear | 1.2% / 1.0% | +1.0% (L12) / +0.6% (L12) | +3.2% / +1.2% | 0 / 13 | 0 / 1 |
| Qwen3.5-4B | MLP skip / linear | 2.0% / 1.8% | +1.6% (L14) / +0.9% (L14) | +6.5% / +6.3% | 0 / 2 | 0 / 0 |

Block 0 is catastrophic for every skip (Qwen3-4B block skip: ppl × 4,800); the last block costs
+18% to +296%. The most tolerant region for blocks and MLPs is the middle (Qwen3-4B 14–21, Qwen3.5-4B
11–18), not the late-middle region that suited attention-input speculation in Phase I. Qwen3-4B
layer 16 is an outlier: its MLP's linear stand-in leaves 101% state error (+21.8% ppl), against
~20% for its neighbours. Qwen3.5-4B: replacing a linear-attention block's attention costs a median
+1.2% against +1.7% for full-attention blocks, repeating Phase I's pattern.

Measured op times (Qwen3-4B, block 26, CUDA graphs, 50 replays per sample): at 1 token, block
0.209 ms, attention 0.086 ms, MLP 0.106 ms, linear stand-in 0.014 ms (FP32 weights); at 512 tokens
0.878 / 0.292 / 0.537 / 0.188 ms. At decode size attention is 41% of a block's time but only 22% of
its MACs, so compute share understates what attention removal saves in time at small T.

**Error measures (D2, first look).** Spearman rank correlation with KL over all atlas rows except the
first and last block: local state error (relative L2 at block l's output) 0.82 (Qwen3-4B) / 0.69
(Qwen3.5-4B); within linear block replacements only 0.59 / 0.44. Error at the last block's output 0.98
/ 0.98, logits 0.98 / 0.98. The size of the local error is a weak predictor of damage; what predicts
it well is only available after running the rest of the network. Early positions are slightly more
sensitive: mean KL in positions 1–63 is 15–20% above positions 256+.

**Additivity and static greedy allocation (D3).** Single ops ranked by KL per unit of compute removed
(best mode per op); every pair among the top 8 measured together; greedy cumulative sets measured
after each addition. Raw: `results/processed/*_allocation_skip_linear.{csv,json}` (all ops) and
`*_allocation_linear.{csv,json}` (attention stand-ins only).

- Pairs: KL is close to additive (pair KL within −6% to +15% of the sum of singles; 27 and 25 pairs
  over all ops, 15 and 15 over attention stand-ins). Perplexity change is mildly super-additive
  (−0.2 to +1.4 points against the sum of singles).
- Larger sets are super-additive in KL too: measured / summed KL 1.15 at 5 ops and 1.47 at 24 ops
  (Qwen3-4B, all ops); 1.33 at 5 and 2.39 at 24 (Qwen3.5-4B).
- Holdout chunks track calibration closely (e.g. Qwen3-4B 5 ops +17.0% vs +15.4%), so the ranking
  generalises within WikiText.

| Model, candidate ops | Largest greedy set ≤ 1% ppl (calibration / holdout) | Compute removed | KL |
|---|---|---|---|
| Qwen3-4B, all | none (first op already +1.52% / +1.34%) | 0% | — |
| Qwen3-4B, attention stand-ins | 2 ops: +0.49% / +0.82% | 1.2% | 0.020 |
| Qwen3.5-4B, all | 1 op (L14 MLP): +0.93% / +0.72% | 1.8% | 0.013 |
| Qwen3.5-4B, attention stand-ins | 1 op (L13): +0.94% / +0.65% | 1.0% | 0.0097 |

The first greedy set to remove ≥ 10% of compute has 5 ops on both models: 12.2% at +17.0% ppl
(Qwen3-4B) and 12.5% at +12.6% (Qwen3.5-4B).

### Findings

1. **Static removal or linear replacement misses the Phase II gate by an order of magnitude** at 4B
   scale without retraining: at most 1.8% of decoder compute fits a 1% perplexity budget, against a
   10% target. Under the KL ≤ 0.01 bound only one op on one model qualifies. In time at decode size
   the best case is about 2% (two attention stand-ins at 0.072 ms each in a ≈ 7.5 ms step; an
   estimate from measured op times, not an end-to-end run).
2. **This meets the plan's stop rule for the selective-compute route at this scale.** Not covered:
   larger models (deeper models are reported to be more redundant; Qwen3.8-27B untested), healing by
   fine-tuning after removal (as layer-pruning papers do), per-token decisions (D6), precision as a
   strategy, and a second text domain.
3. **Greedy allocation is adequate for small sets** (pairs nearly additive in KL) but its estimate
   from singles becomes optimistic beyond a handful of ops.
4. **Local error size is a weak damage signal** (rank correlation 0.44–0.82), which supports the D2
   premise; a useful sensitivity estimator would need to be both cheap and much better than this.
