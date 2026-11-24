# Phase II — Sensitivity-Aware Predictive Execution

Status: **draft plan**, 2026-11-23; steps 1–3 run the same day (exploratory, see "Results so far"). Written from the user's Phase II proposal, with the corrections and
additions listed in "Changes from the proposal". Gates in "Pre-registered gate" must be fixed (dated,
committed) before any Phase II policy is evaluated on the test split.

## Where Phase I left it

Models: Qwen3-4B-Base, Qwen3.5-4B-Base (hybrid linear/softmax attention), Ornith-1.5-9B (BF16, INT8,
and NVIDIA's ModelOpt FP8 W8A8 checkpoint), Qwen3.8-27B (Red Hat INT4 W4A16). RTX 5090, WikiText-2
held-out text, prefill with teacher forcing. Full numbers: `docs/results_summary.md`.

- Predictive signal exists, but only a full-rank linear predictor beats reusing the current state
  (median projection error 17.8% vs 26.0%, Qwen3-4B); low-rank and extrapolating predictors do not.
- Cheap correction fails: the error is not low-rank, channel-sparse or token-sparse enough
  (best: top 25% channels at 0.25× a projection's cost leaves 7.7%).
- Speculating many layers accumulates damage; tolerance is highest in **late-middle** layers. The first
  blocks and the **last** block are the most sensitive (last block alone: +21% ppl on Qwen3.5-4B,
  +10% on Qwen3.8-27B), so "later is safer" is wrong at the very end.
- Lookahead degrades the state quickly but output quality slowly (+1.5% median ppl at k = 8).
- Second-stream overlap does not shorten the critical path from 128 tokens up, including on an idle
  GPU (2026-11-23 rerun: timing CV 0.7–4.8%, saving −2.3% to +1.1%). Below 128 tokens, single-replay
  timings are not stable enough to conclude; see `results_report.md` for the batched rerun.
- Exact reconciliation survives weight-only quantisation and breaks under activation quantisation
  (NVFP4 W4A4 emulated 48–116× the BF16 floor per boundary; real FP8 W8A8 checkpoint 2.9–10×).

The original mechanism (predict, speculate projections, correct cheaply, overlap) did not pass its
gates. Phase II tests a different mechanism; the precision result stands on its own for the paper.

## Thesis

Can predictive information extracted from a pretrained transformer estimate where computation is
sensitive, and allocate computation across layers and operations so that measured compute or latency
falls while output degradation stays within an explicit, pre-registered budget?

Prediction becomes an execution-control signal (which layers and operations run exactly, which are
approximated or removed, when speculation is worth enabling), not only a tensor to reuse.

## Changes from the proposal

1. Context corrected: four models, not two; the real-checkpoint Ornith run is done (P0 item removed);
   the isolated-GPU overlap rerun is done for ≥ 128 tokens; "later layers" made "late-middle layers".
2. **Only mechanisms that remove work count as savings.** Phase I showed overlap does not; the
   candidate strategies are: run exactly, remove a whole block, remove its attention or its MLP, replace
   either with a linear map (a d×d GEMM, cost known), lower its precision, and per-token versions of
   these. Each gets a measured cost before it enters any allocation.
3. **Signal-availability rule** (new, binding on Directions 5–6): a per-token decision may only use
   signals available *before* the computation it gates. The predictor's true error needs the exact next
   state, which is the work being skipped, so it cannot gate that work. Admissible signals: statistics
   of states already computed, a cheap learned estimate of downstream damage, offline per-layer
   sensitivity. Every signal's cost is charged against the saving.
4. **Strongest static baseline named:** the best fixed set of removed or replaced blocks/operations,
   chosen greedily on calibration data under the same budget (the ShortGPT / SLEB style). An adaptive
   policy must beat it at equal measured cost.
5. **Data splits fixed:** predictors fitted on WikiText-2 train chunks 0–255; atlas and policy
   calibration on train chunks from 400 on (disjoint); evaluation on the WikiText-2 test split plus a
   second domain (C4 or code, to be downloaded with the user's approval) that the policy never sees.
6. Direction 4 limited to one new measurement before a stop decision (below).
7. Prior art to differentiate against listed explicitly (below); to be verified in
   `docs/research_review.md` before any novelty claim.

## Directions

**D1 — Layer Sensitivity Atlas.** Per block and per operation (whole block, attention or linear
attention, MLP): output damage when removed or replaced by a linear map, measured as KL, top-1 agreement,
perplexity, logit relative error, damage by token position, and next-state relative error and cosine,
against the operation's cost (MACs per token; measured ms where kernels are representative).
Qwen3-4B and Qwen3.5-4B first; Ornith-1.5-9B and Qwen3.8-27B for generality. *Atlas v1 = this.*

**D2 — Sensitivity-aware error.** Test whether a sensitivity-weighted error, eᵀJᵀJe (approximated with
the output Fisher via gradients of sampled tokens, or a per-layer projection onto sensitive directions),
predicts per-token KL better than ‖e‖/‖h‖, cosine, or projection error. Needs per-token logging. Success:
a clearly higher rank correlation with per-token KL on calibration data, at a stated cost.

**D3 — Budgeted allocation.** Minimise Σ C_l(a_l) subject to ΔQ ≤ ε. First test **additivity**: for pairs
of operations, is damage(a, b) ≈ damage(a) + damage(b)? If yes, greedy allocation is near-optimal; if
not, measure interactions and use a search. Related: HAWQ-V3 solves this as an integer program for
mixed precision with Hessian sensitivity.

**D4 — Error geometry, one last test.** Measure how much of the linear predictor's error lies in the top
singular directions of the next block's input projections (the only part that can change their output).
If that share is not much larger than chance, close the correction branch.

**D5/D6 — Prediction as control; adaptive policy.** Policy a_l = π(signals available before l). Classes in
order: fixed rules, thresholds, greedy budget allocation, a small learned controller. Compare with the
exact model, the strongest static baseline, uniform approximation and late-layer-only approximation.

**D7 — Runtime, only when economics justify it.** Conditions: isolated GPU (met for ≥ 128 tokens),
reproducible baseline, an operation with measured headroom, favourable cost after all overheads, a
contention profile, critical-path reduction shown end to end with matched implementations.

## Pre-registered gate (proposed; fix before evaluating any Phase II policy)

- Quality budget ε: mean KL ≤ 0.01 nats per token **and** perplexity increase ≤ 1.0%, on the
  WikiText-2 test split and on the second domain, separately.
- Benefit: ≥ 10% fewer decoder MACs per token (counted from executed operations, predictor and policy
  included), **or** ≥ 5% lower end-to-end latency with baseline CV < 5% over 3 runs.
- Must beat the strongest static baseline at the same budget by a margin larger than run-to-run
  variation (3 seeds of calibration data).
- No change to ε, the metrics or the splits after test-split results are seen; any change is dated
  here with its reason and every affected run is redone.

## Order of work

1. **Atlas v1** (D1) on Qwen3-4B and Qwen3.5-4B: remove / linear-replace each block, attention and MLP.
2. **Additivity** (D3) on pairs drawn from the atlas's most tolerant operations.
3. **Static greedy allocation**: compute saved at ε on calibration data; this is the baseline.
4. **D4 single test**; stop the correction branch if negative.
5. **D2 per-token sensitivity** and admissible signals; then **D6 policy** against step 3.
6. Runtime work only if step 5 clears the gate on compute.

Stop rule: if step 3 finds that no allocation saves ≥ 10% of MACs within ε, the selective-compute route
is documented as a negative result with the atlas and the branch ends.

## Prior art to differentiate (verify before claiming novelty)

Layer removal and importance: ShortGPT (Block Influence), Gromov et al. "The Unreasonable
Ineffectiveness of the Deeper Layers", SLEB, LaCo. Adaptive depth per token: CALM, LayerSkip,
Mixture-of-Depths. Predictor-filled layers: RidgeFill. Budgeted sensitivity allocation: HAWQ-V3.
Layer-parallel speculation: Layer Parallelism, CQIL, EasySpec. Candidate openings: per-operation
sensitivity in hybrid linear/softmax-attention models; one budget covering removal and precision
together (Phase I shows activation-quantised layers behave differently); a measured test of additivity.

## Results so far (2026-11-23, exploratory)

Steps 1–3 on Qwen3-4B and Qwen3.5-4B (details: `results_report.md`, "Phase II, D1/D3"):
- Atlas: no single block fits a 1% perplexity budget on either model; attention stand-ins are the
  most tolerant operation (median +0.6% / +1.2%) but each removes only 0.6% / 1.0% of compute.
- Static greedy allocation within 1% perplexity: 1.2% (Qwen3-4B) and 1.8% (Qwen3.5-4B) of decoder
  compute, on calibration and holdout chunks alike. The 10% target needs +12.6% to +17% perplexity.
- KL near-additive for pairs, super-additive for larger sets; local state error ranks damage only
  moderately (Spearman 0.44–0.82).
- The proposed KL ≤ 0.01 bound binds before the perplexity bound (no single Qwen3-4B op meets it).

**The stop rule above is met for static removal and linear replacement at 4B scale.** Whether the
branch ends or continues with one of the uncovered extensions (Qwen3.8-27B atlas, healing by
fine-tuning, per-token policies, precision as a strategy) is the user's decision; the gate must not be
loosened to rescue it.
