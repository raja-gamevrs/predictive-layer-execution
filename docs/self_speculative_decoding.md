# Self-speculative decoding with atlas-guided layer skipping (pre-registered 2026-11-24)

Written before any run. Bars do not change after results are seen; changes get a dated amendment.

## Idea

Phase II found that statically skipping operations costs far too much quality to *replace* the model
(≤ 1.8% compute within 1% ppl). Here a skipped-operation copy of the same model only *drafts*: it
proposes the next k tokens greedily, the full model checks all k + 1 positions in one forward pass,
and the longest agreeing prefix plus the full model's next token is kept. The output is the full
model's greedy output, so the draft may be lossy; what matters is how many drafted tokens are
accepted and how cheap a draft step is. Prior art: Draft & Verify (Zhang et al. 2023), SWIFT
(2024), LayerSkip (trained, 2024). What is ours: the per-operation (attention / MLP) skip choice
from the Phase II sensitivity atlas, measured on a CUDA-graph decoder.

## Design

- Model: Qwen3-4B-Base, BF16, batch 1, RTX 5090, decoder `src/ple/decode.py` (checked against
  Transformers in E1). Draft and verify share one KV cache; verify rewrites every position it covers.
- Draft skip sets (from `results/processed/Qwen3-4B-Base_sensitivity_atlas.csv`, skip mode, ranked
  by KL per MAC, block 0 and the last block never skipped): ~26%, ~41% and ~55% of decoder MACs.
  Baseline: skip whole blocks 12–23 (33%), the contiguous-middle choice common in prior work.
- Draft length k ∈ {2, 3, 4, 6}.
- Prompts: 8 calibration prompts (WikiText-2 train chunks 400–407, 512 tokens) to choose the best
  (skip set, k); report on 8 holdout prompts (chunks 600–607) and 8 code prompts (512-token windows
  of this repo's own `.py` files, a second domain). 256 generated tokens per prompt.
- Timing: whole round on the GPU with CUDA graphs for the draft step and the verify pass, one host
  sync per round for the acceptance count; exact baseline = E1's graph-captured greedy decode.
  Timed only when no other job is using the GPU.

## Bars

- **Correctness:** generated tokens identical to exact greedy decoding at every position, except
  where the exact model's top-2 logit gap is below 0.05 (BF16 near-ties between the k + 1-token verify
  kernel and the 1-token decode kernel); any such divergence is reported with its gap.
- **Speed:** ≥ 1.3× tokens per second over exact greedy decode on the holdout prompts **and** ≥ 1.1× on
  code, with the configuration chosen on calibration prompts.

## Amendments

- 2026-11-24, after a 2-prompt correctness smoke test and before any timing: the correctness bar's
  0.05 gap is below BF16 logit resolution (one ulp is 0.125 at |logit| 16–32), and the exact
  single-token decode itself disagrees with a full-sequence pass at such near-ties (1–4 per 64
  tokens). Correctness is therefore checked teacher-forced: one full-model pass over prompt + output,
  and every output token must be that pass's argmax unless the top-2 gap is ≤ 0.25 (two ulps).
  Exact greedy decoding is held to the same check. A no-skip draft (draft = full model) is added as a
  control: its acceptance should be near 100%. The speed bar is unchanged.
- 2026-11-24, timing protocol: the first timed run timed each prompt's exact baseline once, at the
  start, and another GPU job ran during part of it (exact decode went from 8.7 to as much as
  75 ms/token on some prompts), so its speedups (up to 3.6×) are artefacts. That run is kept as
  `Qwen3-4B-Base_self_spec_timed_contended.json` and not used. Timing is redone with
  `--time-only`: for every (set, k, prompt), exact and self-spec decode alternate on the same prompt
  (2 runs each after a warm-up, medians), prefill excluded from both, speedup = exact / self-spec
  ms per generated token; a pair whose exact time is > 5% above the fastest exact seen is re-timed
  up to 3 times and flagged if still slow. Correctness and acceptance come from the untimed run
  (deterministic). Bars unchanged.

## Results (2026-11-24): correct, but slower than exact decoding at every setting — **fail**

Raw files: `results/processed/self_spec/Qwen3-4B-Base_self_spec.json` (correctness, acceptance) and
`Qwen3-4B-Base_self_spec_timed.json` (interleaved timing; `scripts/self_spec_bench.py --time-only
--runs 2`). Exact decode 8.70–9.05 ms/token across the 480 timed pairs (3 re-timed once, none left
contended).

**Correctness: met.** 0 greedy violations in 480 runs (20 settings × 24 prompts × 256 tokens). Every
output token that differs from the teacher-forced argmax is a near-tie (gap ≤ 0.25): 128–137 per
setting, set by k alone (the verify kernel's shape) and not by the draft, against 117 for exact
one-token greedy decoding itself.

**Speed: not met.** Mean per-prompt speedup over exact greedy decoding (tokens per round and
acceptance on holdout):

| Draft | k | tokens/round | drafts accepted | speedup cal | speedup hold | speedup code |
|---|---|---|---|---|---|---|
| atlas ~26% | 2 | 2.47 | 73% | 0.883× | **0.938×** | **0.874×** |
| atlas ~26% | 3 | 2.92 | 64% | 0.794× | 0.861× | 0.793× |
| atlas ~26% | 4 | 3.28 | 57% | 0.715× | 0.776× | 0.699× |
| atlas ~26% | 6 | 3.78 | 47% | 0.600× | 0.659× | 0.573× |
| atlas ~41% | 2 | 2.04 | 52% | 0.832× | 0.918× | 0.885× |
| atlas ~41% | 3 | 2.25 | 42% | 0.722× | 0.807× | 0.768× |
| atlas ~41% | 4 | 2.44 | 36% | 0.633× | 0.712× | 0.649× |
| atlas ~41% | 6 | 2.54 | 26% | 0.491× | 0.561× | 0.507× |
| atlas ~55% | 2 | 1.48 | 24% | 0.675× | 0.741× | 0.728× |
| atlas ~55% | 3 | 1.53 | 18% | 0.556× | 0.622× | 0.603× |
| atlas ~55% | 4 | 1.53 | 13% | 0.467× | 0.510× | 0.505× |
| atlas ~55% | 6 | 1.53 | 9% | 0.359× | 0.392× | 0.389× |
| blocks 12–23 (33%) | 2 | 2.07 | 53% | 0.760× | 0.841× | 0.786× |
| blocks 12–23 (33%) | 3 | 2.31 | 44% | 0.642× | 0.737× | 0.657× |
| blocks 12–23 (33%) | 4 | 2.39 | 35% | 0.550× | 0.612× | 0.547× |
| blocks 12–23 (33%) | 6 | 2.55 | 26% | 0.413× | 0.485× | 0.413× |
| none (control) | 2 | 2.98 | 99% | 0.960× | 0.964× | 0.966× |
| none (control) | 3 | 3.94 | 98% | 0.962× | 0.963× | 0.969× |
| none (control) | 4 | 4.96 | 99% | 0.947× | 0.958× | 0.951× |
| none (control) | 6 | 6.81 | 98% | 0.945× | 0.957× | 0.963× |

Chosen on calibration: atlas ~26%, k = 2 → 0.938× holdout, 0.874× code, against bars of 1.3× and
1.1×. No skipping setting beats exact decoding on any domain. The atlas sets beat the contiguous
blocks 12–23 baseline at equal or lower acceptance cost (atlas ~41% at 0.918× vs blocks at 0.841×,
k = 2 holdout), so the per-operation choice helps, but not enough.

**Why.** Fitting round time against k per set (slope = one draft step, intercept = verify plus
loop overhead, in exact-step units): a draft step costs 82% / 60% / 48% of an exact step with
~26% / 41% / 55% of MACs skipped (69% for blocks 12–23), and verify plus overhead costs 0.99–1.05
steps. At k = 2 with every draft accepted, the best possible speedups would be 1.12× (atlas ~26%),
1.35× (~41%), 1.52× (~55%) and 1.26× (blocks): only the larger skip sets could clear 1.3× even in
principle, and those are exactly the drafts the full model rejects most (52% and 24% accepted). The
no-skip control (0.96×) is the cost of the draft-and-verify loop itself. Skipping sub-layers in
place makes a draft that is either too expensive or too inaccurate; a usable self-draft would need
to cost ≲ 40% of a step at ≳ 80% acceptance, which no static skip set here comes near.
