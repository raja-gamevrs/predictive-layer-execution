# Related work and prior-art review (Stage A)

Checked 2026-11-23. I verified every entry against its arXiv abstract page, its HTML full text
or a proceedings page. Entries marked **[unverified]** could only be confirmed from secondary
sources. The structured comparison is in [`novelty_matrix.csv`](novelty_matrix.csv), and the
BibTeX is in [`paper/references.bib`](paper/references.bib).

**What PLE is, for reference.** For block boundary l → l+1 of a frozen pretrained decoder:
predict ĥ_(l+1) from h_l, launch block l+1's input projections (q/k/v) on N(ĥ_(l+1)) while block l
is still running, then reconcile when h_(l+1) arrives. Reconciliation can be the exact
y = W·N(ĥ) + W·e (with e = N(h) − N(ĥ)), a cheap approximate correction, or nothing.

## Corrections to the previous bibliography

| Previous entry | Problem | Fix |
|---|---|---|
| SNLP (2026), arXiv 2605.17842 | **Exists and is correct.** Han, Xu, Wang, Srivastava. v1 18 May 2026, v3 8 Jul 2026. Preprint, no venue. | Kept. |
| "ParaFormer (2025)", arXiv 2510.15425 | The paper is now titled **TeamFormer: Shallow Parallel Transformers with Progressive Approximation** (v2, 24 Feb 2026). "ParaFormer" was the v1 title. It is a trained vision architecture, so it is weak prior art. | Retitled. |
| "Leveraging the true depth of LLMs (2025)" | Correct id. v3 (6 Jan 2026) adds Qwen3-4B/14B and says "Reviewed on OpenReview". The venue (reportedly TMLR 01/2026) is **[unverified]** because OpenReview blocked automated access. | Cited as arXiv, venue flagged. |
| CQIL, LayerSkip (ACL Anthology links only) | Correct, but the arXiv ids were missing: CQIL is 2404.06709, LayerSkip is 2404.16710. | Added. |
| EAGLE "(ICML 2024)" | Correct: PMLR 235:28935–28948. | — |
| Parallel Track Transformers (Apple page) | Correct. arXiv 2602.07306 (Feb 2026), no venue. | arXiv id added. |
| CLaSp "(2025)" | arXiv says "accepted to ACL 2025". | Venue added. |
| Missing | The closest works were not listed at all: EasySpec, InfiniGen, RidgeFill, DejaVu, Ladder-Residual, Pre-gated MoE. | Added below. |

## 1. Layer-parallel inference on approximate inputs (closest prior art)

**Calvo González, Paliotta, Pagliardini, Jaggi, Fleuret. "Leveraging the true depth of LLMs."**
arXiv [2502.02790](https://arxiv.org/abs/2502.02790), v3 2026 (venue [unverified], OpenReview
`JccJ6YfWd4`). *Layer Parallelism* (LP) rewrites a pair of consecutive layers k, k+1 as
ũ = x + A_k(x) + A_(k+1)(x) and out = ũ + F_k(ũ) + F_(k+1)(ũ). Both attentions read the same input,
with QKV stacked and fused across two GPUs under tensor parallelism. The method is post hoc and
needs no training. It reports 1.19× throughput on Llama 2 7B for a 1.5% average accuracy drop and
evaluates **Qwen3-4B** too. Quality collapses beyond about 7 merged layers on Qwen3-4B, and the
authors keep a sequential tail (for Qwen3, LP is applied only "until the 4th to last decoder
layer") and avoid early layers. *Relation:* PLE's uncorrected "QKV of block l+1 from h_l" is LP's
attention branch on the same model. PLE adds a learned predictor and an exact or approximate
correction. PLE's per-layer data agrees with LP's tail recommendation: boundary 34 is the least
tolerant, while 26–30 are the most tolerant.

**Zou, Wang, Zhao, Kong, Yang, Deng. "CQIL: Inference Latency Optimization with Concurrent
Computation of Quasi-Independent Layers."** ACL 2024 (Long), pp. 7293–7307,
doi:10.18653/v1/2024.acl-long.394, arXiv [2404.06709](https://arxiv.org/abs/2404.06709).
Groups of p middle layers share the input of the group's first layer and run concurrently, one
GPU per layer. A "bypass" forwards earlier layers' attention outputs into later layers' FFN
inputs to recover lost information. It reaches up to 48.3% lower latency on LLaMA-33B, and LoRA
on about 0.5B tokens nearly closes the quality gap. *Relation:* the same identity-predictor
speculation as PLE, plus a cheap partial correction (the bypass). The parallelism is across
devices rather than overlapped on one device.

**Wu, Gao, Li, Wu. "EasySpec: Layer-Parallel Speculative Decoding for Efficient Multi-GPU
Utilization."** arXiv [2502.02493](https://arxiv.org/abs/2502.02493), v2 Dec 2025 (no venue).
In the *draft* model, groups of N consecutive **attention** layers (excluding the first and last)
all take the group's input hidden state and run in parallel across GPUs, while MLPs stay
sequential ("fuzzy speculation"). After each draft round the fuzzy KV entries are discarded and
recomputed exactly in one pass ("bonus calibration"), and the target model verifies the tokens.
The authors cite InfiniGen as the inspiration. *Relation:* this is the "start block l+1's
attention from h_l, keep the rest sequential" regime, already published, with exactness restored
by discard-and-recompute.

**Han, Xu, Wang, Srivastava. "SNLP: Layer-Parallel Inference via Structured Newton
Corrections."** arXiv [2605.17842](https://arxiv.org/abs/2605.17842), v3 Jul 2026 (preprint).
SNLP treats the full depth trajectory as the root of a nonlinear residual system. All suffix
layers are evaluated in parallel on the previous iterate, then a structured surrogate Jacobian
is applied. In the Identity-Newton variant the correction is
h_l^(k+1) = f_l(h_(l−1)^(k)) + h_(l−1)^(k+1) − h_(l−1)^(k), a prefix sum of corrections over depth.
Initialisation is the prefix state, a batched forward, or "a lightweight predictor". It reaches
2.3–2.58× wall-clock on 0.5B nanochat models when trained with an SNLP regulariser. On
off-the-shelf Qwen2.5-0.5B and TinyLlama-1.1B it matches perplexity but runs *slower* than
sequential, and on 3B it reaches 1.20× for +3.9% PPL. It can also act as a self-speculative
drafter. *Relation:* SNLP is the general "predict all layer inputs, then correct" framework.
PLE's one-boundary, one-iteration, exact-correction variant is a special case: a Jacobi/Newton
step with a learned initial guess. SNLP's finding that untrained models do not benefit is direct
negative evidence for PLE's frozen-weights premise. A follow-up applies SNLP to encrypted
inference: Han et al., arXiv [2607.04819](https://arxiv.org/abs/2607.04819).

**Sun, Pickett, Nain, Jones. "Transformer Layers as Painters."** AAAI 2025, 39:25219–25227,
arXiv [2407.09298](https://arxiv.org/abs/2407.09298). On frozen Llama2-7B and BERT, middle layers
can run in parallel on the same input with outputs averaged (looped variants work better). First
and last layers are special. *Relation:* tolerance evidence for approximate-input execution, with
no prediction or correction.

**Lad, Lee, Gurnee, Tegmark. "The Remarkable Robustness of LLMs: Stages of Inference?"** arXiv
[2406.19384](https://arxiv.org/abs/2406.19384) (NeurIPS 2025 poster). Deleting or swapping
adjacent layers keeps 72–95% of accuracy, and the damage concentrates at the first and last
layers. The paper proposes depth stages that end in "residual sharpening". *Relation:* context
for why late layers might be predictable, but it does not test speculation.

**Freiberger, Kun, Løvlie, Risi. "LayerShuffle."** arXiv
[2407.04513](https://arxiv.org/abs/2407.04513). Training ViTs under random layer order makes them
order-robust at about a 20% accuracy cost. *Relation:* weak. It works through training, not
speculation.

**Solver-based depth parallelism.** Song, Meng, Liao, Ermon. "Accelerating Feedforward
Computation via Parallel Nonlinear Equation Solving." ICML 2021, PMLR 139:9791–9800, arXiv
[2002.03629](https://arxiv.org/abs/2002.03629). Jacobi and Gauss-Seidel iterations reproduce the
feedforward result exactly in at most N parallel steps. Danieli et al. "DeepPCR." NeurIPS 2023,
arXiv [2309.16318](https://arxiv.org/abs/2309.16318), uses parallel cyclic reduction. Jiang et
al. "Layer-Parallel Training for Transformers." arXiv
[2601.09026](https://arxiv.org/abs/2601.09026), uses MGRIT over depth during training. *Relation:*
PLE's exact correction is one Jacobi step with a learned initial guess, a well-established idea.

## 2. Architectures that remove sequential dependencies

**Parallel attention + MLP blocks.** GPT-J-6B (Wang & Komatsuzaki, 2021,
[mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax); model release, no
paper) and PaLM (Chowdhery et al., JMLR 24(240):1–113, 2023, arXiv
[2204.02311](https://arxiv.org/abs/2204.02311)) compute y = x + Attn(LN(x)) + MLP(LN(x)) and fuse
the input projections, which gives about 15% faster training. PaLM reports small degradation at
8B and none at 62B. *Relation:* the same dependency removal as uncorrected PLE, but trained in.

**Cutler et al. "StagFormer."** arXiv [2501.15665](https://arxiv.org/abs/2501.15665). The upper
stack at step i consumes lower-stack outputs only up to step i−1, so the stacks run in parallel
(about 33% simulated decode speedup). Trained.

**Zhang, Mishra, et al. "Ladder-Residual."** ICML 2025 (PMLR 267), arXiv
[2501.06589](https://arxiv.org/abs/2501.06589). Each module reads the stale residual r_(i−2), so
its compute overlaps the previous module's tensor-parallel all-reduce. This gives about 30%
end-to-end speedup on 70B/TP8, and Llama-3.1-8B can be retrofitted with about 3B tokens.
*Relation:* conceptually the closest "use a one-step-stale state to overlap work" design. PLE
attempts the same without retraining and on a single device.

**Prabhakar, Zhang, Wentzlaff. "Kraken."** NeurIPS 2024, arXiv
[2408.07802](https://arxiv.org/abs/2408.07802). **Wang et al. "Parallel Track Transformers."**
arXiv [2602.07306](https://arxiv.org/abs/2602.07306) (Apple, 2026). **Wang, Wei, Li.
"TeamFormer" (v1: "ParaFormer").** arXiv [2510.15425](https://arxiv.org/abs/2510.15425). These are
trained architectures that reduce cross-device synchronisation or use parallel branches.
*Relation:* weak. None of them is post-hoc speculation.

## 3. Predicting later-layer states or ops from the current state

**Lee, Lee, Seo, Sim. "InfiniGen."** OSDI 2024, pp. 155–172, arXiv
[2406.19707](https://arxiv.org/abs/2406.19707). During decoding at layer i−1, InfiniGen multiplies
layer i−1's *layer-normalised attention input* by a partial (SVD-skewed, top-30% columns) query
weight of layer i and a partial key cache to speculate layer i's attention. It then prefetches
only the important KV from CPU. The method rests on consecutive block inputs having cosine
similarity 0.89–0.97 because of large-magnitude residual outlier channels. The real layer-i
attention uses the exact query. *Relation:* this is PLE's C2 with the identity predictor and a
low-rank Q, run one layer early, with the speculation used for selection rather than values. It
also anticipates PLE's observation that outlier channels dominate ‖h‖.

**Liu et al. "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time."** ICML 2023,
PMLR 202:22137–22176, arXiv [2310.17157](https://arxiv.org/abs/2310.17157). Asynchronous
"lookahead" predictors read layer l's input and predict layer l+1's head and neuron sparsity,
running in parallel with layer l. The justification is consecutive-layer cosine similarity of
about 0.99 on OPT. *Relation:* the same "predict the next block from the current input,
concurrently" schedule, but it predicts masks, not activations.

**Hwang et al. "Pre-gated MoE."** ISCA 2024, arXiv [2308.12066](https://arxiv.org/abs/2308.12066).
Block N's (fine-tuned) gate selects block N+1's experts so that expert migration overlaps block
N. **Madan et al. "Speculating Experts Accelerates Inference for Mixture-of-Experts."** arXiv
[2603.19289](https://arxiv.org/abs/2603.19289) (2026). It speculates future experts from current
internal representations, optionally with lightweight estimators. *Relation:* speculative launch
of a next-layer operation from the current state, limited to routing.

**Johnston & Smith. "RidgeFill: Training-Free Late-Layer Cache Fill for Faster Prefill."** arXiv
[2611.09709](https://arxiv.org/abs/2611.09709), 6 Nov 2026. For each late layer ℓ ≥ S, a
**per-layer, full-rank, closed-form ridge map** x̂_ℓ = A_ℓ x_S + b_ℓ predicts that layer's input
from the split state. The prediction goes through the layer's own norm and K/V projections to
write the cache, so filled rows never wait for layers S…ℓ−1. There is no per-token exact check:
an exact tail and selected rows carry quality, and the authors found decode-time checking too
costly. It is training-free, evaluated on Llama-3.1-8B and **Qwen3-8B**, Qwen3.5 and Qwen3.8,
with 1.25–1.60× TTFT. The ridge map recovers far more than the identity fill (+49.6 RULER points
on filled regions). *Relation:* the closest neighbour of PLE's *predictor*: the same estimator
class (per-layer full-rank ridge on a frozen Qwen3-family model), the same "feed the prediction
through the layer's own projections", and the same identity baseline. It differs in skipping
multi-layer spans during prefill instead of overlapping one boundary, and in having no exact
reconciliation. It appeared 17 days before this review.

**Related cache-fill work.** He, Fu, Wang, Li. "POP: Prefill-Only Pruning." arXiv
[2602.03295](https://arxiv.org/abs/2602.03295): deep layers skipped in prefill, with KV written
from an earlier state via independent KV projections. Qiao et al. "SwiftKV." arXiv
[2410.03960](https://arxiv.org/abs/2410.03960): later layers' KV prefilled from an earlier layer's
output, with distillation. Horton et al. "KV Prediction for Improved Time to First Token." arXiv
[2410.08391](https://arxiv.org/abs/2410.08391): an auxiliary model predicts the base KV cache.
Wang. "RISK-KV." arXiv [2610.29544](https://arxiv.org/abs/2610.29544): cross-layer ridge
prediction of a layer's attention-projection input from neighbouring exact KV, with NLL-based
quality gating.

**Linear inter-layer maps and lenses.** Yom Din, Karidi, Choshen, Geva. "Jump to Conclusions:
Short-Cutting Transformers with Linear Transformations." LREC-COLING 2024, pp. 9615–9625, arXiv
[2303.09435](https://arxiv.org/abs/2303.09435). Fitted linear maps from layer l to later layers
beat the identity baseline and are used for early exit. Belrose et al. "Eliciting Latent
Predictions from Transformers with the Tuned Lens." arXiv
[2303.08112](https://arxiv.org/abs/2303.08112): per-layer affine translators to the final layer.
Razzhigaev et al. "Your Transformer is Secretly Linear." ACL 2024, pp. 5376–5384, arXiv
[2405.12250](https://arxiv.org/abs/2405.12250): Procrustes similarity of about 0.99 between
consecutive layers (after normalisation), used for linear replacement and pruning. *Relation:*
PLE's ridge predictor is this predictor class. PLE's measured ceiling (about 51% of update
energy, 18% post-norm projection error on Qwen3-4B) is a quantitative caveat to "secretly
linear", not a new predictor.

## 4. Speculation with correction inside a forward pass, and the correction algebra

Apart from SNLP, EasySpec, InfiniGen, DejaVu and the MoE prefetchers above, I found no paper that
speculatively computes a *next-layer projection's values* on one device and then applies an exact
W·e reconciliation. The algebra itself is standard incremental computation. **Dutson, Li, Gupta.
"Eventful Transformers."** ICCV 2023, pp. 16911–16923, arXiv
[2308.13494](https://arxiv.org/abs/2308.13494). It computes W·x as a cached W·x_ref plus updates
for only the tokens whose change exceeds a threshold, which is PLE's "exact fix for worst-k%
tokens" correction applied over time instead of depth. Speculative-execution-with-rollback
applied to layers is in effect what EasySpec (discard fuzzy KV, recompute) and SNLP (sequential
verifier) do.

## 5. Token-level speculative decoding (terminology only)

Leviathan, Kalman, Matias. "Fast Inference from Transformers via Speculative Decoding." ICML
2023, PMLR 202:19274–19286 ([link](https://proceedings.mlr.press/v202/leviathan23a.html)).
Hu et al. "Speculative Decoding and Beyond: An In-Depth Survey of Techniques." arXiv
[2502.19732](https://arxiv.org/abs/2502.19732). Li, Wei, Zhang, Zhang. "EAGLE." ICML 2024, PMLR
235:28935–28948, arXiv [2401.15077](https://arxiv.org/abs/2401.15077), predicts second-to-top
features of *future tokens*. Elhoushi et al. "LayerSkip." ACL 2024, pp. 12622–12642, arXiv
[2404.16710](https://arxiv.org/abs/2404.16710). Chen et al. "CLaSp." ACL 2025, arXiv
[2505.24196](https://arxiv.org/abs/2505.24196). These speculate across tokens and verify with the
full model. They are not intra-forward-pass speculation, and PLE borrows only their verification
discipline.

## 6. Name collision: "PLE"

* **Exact phrase "Predictive Layer Execution":** 0 hits in the arXiv API (all fields). A web
  search found no ML paper. The only GitHub repository is this project
  (`raja-gamevrs/predictive-layer-execution`, created 2026-11-23). The USPTO and patent results
  were unrelated ("predictive model execution layer" in request-processing patents). The
  Semantic Scholar API was rate-limited and Google Patents was not queried directly, so treat
  those two as **[not checked]**.
* **Gemma 3n "Per-Layer Embeddings (PLE)"** ([Google docs](https://ai.google.dev/gemma/docs/gemma-3n)):
  per-layer embedding parameters that can be cached off-accelerator ("PLE caching"). This is a
  direct collision in on-device LLM inference, the same community PLE targets.
* **"Progressive Layered Extraction (PLE)"** (Tang et al., RecSys 2020,
  doi:10.1145/3383313.3412236): a widely used multi-task recommendation model, also a "layer"
  name.
* **Piecewise linear encoding** (Gorishniy, Rubachev, Babenko, NeurIPS 2022, arXiv
  [2203.05556](https://arxiv.org/abs/2203.05556)): tabular feature encoding. The abstract does
  not use the acronym, and whether the paper body uses "PLE" is **[unverified]**.
* **Recommendation:** rename before publication, or always write it out in full. "PLE" alone
  will be confused with Gemma 3n's Per-Layer Embeddings.

## Novelty boundary

**Already covered by prior work:**

* *Running block l+1's attention (or the whole block) on block l's input, in parallel, without
  retraining.* This is published as LP (Calvo González et al., including Qwen3-4B), CQIL, and
  EasySpec (attention-only, MLP sequential, the exact "QKV of l+1 from h_l" regime). The
  plausible PLE regime is therefore **not novel as a mechanism**. LP also documents the same
  collapse when too many layers are parallelised.
* *Computing the next layer's (partial) query from the current layer's normalised input, one
  layer early.* Covered by InfiniGen, which also notes the outlier-channel explanation. DejaVu
  and Pre-gated MoE do the same for sparsity masks and routers.
* *A per-layer full-rank ridge map from an earlier state to a later layer's input, fed through
  the frozen layer's own norm and projections.* Covered by RidgeFill (Nov 2026, Qwen3-family),
  with RISK-KV and Jump-to-Conclusions as further linear-map precedents.
* *Predicting all layer inputs and correcting them.* Covered by SNLP, and in its exact-iteration
  form by Song et al. (2021). PLE's exact correction W·N(ĥ) + W·e is one Jacobi step with a
  learned initial guess. Token-gated delta correction is Eventful Transformers' algebra.

**What PLE as currently measured does that none of these report:**

1. Error is measured **at the next block's post-RMSNorm projection input** (q_proj relative
   error), not in hidden-state L2. The finding that a small L2 error (8–10%) still gives about 25%
   projection error at layers 17–21 is a useful metric correction.
2. A **per-boundary, single-layer sensitivity map** of uncorrected next-block QKV speculation on
   a modern model (≤1.2% PPL at any one layer, 0.25% at late layers), compared against
   identity, extrapolation, velocity and ridge predictors. LP and CQIL report only contiguous
   multi-layer groups. PLE's per-layer data is consistent with LP's guidance to keep the final
   layers sequential: boundary 34 (next to the output) is the least tolerant (+4% PPL with
   identity), while boundaries 26–30 are the most tolerant (+0.25%). *(Corrected 2026-11-23
   after checking the LP full text; an earlier draft of this review claimed a conflict.)*
3. **Negative results on correction economics:** zero-parameter extrapolation is worse than
   identity (cos ≈ 0.08 between consecutive updates). The speculation error is neither low-rank
   nor token- or channel-concentrated enough for a correction much cheaper than W·e (cheap
   corrections leave 8–20% error). None of the prior works quantify this.

**Narrowest defensible contribution:** an empirical characterisation, on frozen modern LLMs
(Qwen3-4B, extended to more models), of next-block-input predictability and of the cost and
accuracy of reconciling speculative projections, measured after the norm. It should be framed
explicitly as an analysis of when LP, CQIL and EasySpec-style approximate-input execution and
InfiniGen and RidgeFill-style predicted-input projections are safe, and why exact
reconciliation does not pay. A latency claim would be new only for **single-device overlap**
(CUDA streams at batch 1) of block l+1's QKV with block l, because LP, CQIL and EasySpec all get
their speedups from multi-GPU tensor or layer parallelism. That claim needs a measured
wall-clock win over the fused sequential baseline before it can be made. PLE should not claim
the speculate-and-reconcile idea, the ridge predictor, or the "QKV of l+1 from h_l" schedule as
novel.
