# E3b: healing scaling curve, "PLE as architecture" (pre-registered 2026-11-24)

Written before any E3b training run. The design, bar and stop rule below do not change after
results are seen; any change is added as a dated amendment with its reason.

## Question

E3 (docs/experiments_e1_e5.md) healed all-block speculation with a 1,000-step LoRA and cut holdout
damage from KL 0.218 / +18.2% ppl to KL 0.138 / +8.8%. If the model is instead trained for much
longer to *expect* a one-block-stale attention input, does quality approach the project bar, or does
the curve flatten well above it?

## Design

- **Model**: Qwen3-4B-Base, BF16, `configs/primary_qwen3_4b.yaml`, RTX 5090, batch 1.
- **Speculation (fixed, train and eval)**: identity predictor at every boundary l = 0..34 (35 of 35):
  block l+1's q/k/v are computed from RMSNorm_(l+1)(h_l) instead of RMSNorm_(l+1)(h_(l+1)),
  uncorrected (`gated_speculation(layers, range(35), identity, d)`, tau = None), position 0 exact.
  Note E3 used the ridge predictor; E3b uses identity (no fitted component at all).
- **Trainable part (settings, in order)**:
  - (a) **main curve**: LoRA rank 16 on every projection (q/k/v/o, gate/up/down) of every block, as
    E3: the delta lives inside each projection's own forward, so the speculative path goes through it.
    LoRA scale alpha/r = 2 (E3's alpha 32).
  - (b) capacity test, if time allows: LoRA rank 64 (or 128), scale 2·sqrt(16/r) (rank-stabilised so
    the effective step matches rank 16; = 1.0 at r = 64).
  - (c) full-rank, only if it fits under the memory cap: a memory-lean variant (for example a full-rank
    trainable delta on q/k/v with LoRA rank 16 elsewhere and low-precision optimiser state). Exploratory.
- **Objective**: KL(p_exact ‖ p_student) over all next-token positions of the chunk, teacher = the
  original exact model (LoRA disabled, no speculation), recomputed each step.
- **Optimiser**: AdamW, lr 2e-4, weight decay 0, grad-norm clip 1.0, LoRA params FP32.
  **Schedule: linear warmup over 200 steps, then constant lr to 16,000 steps** (constant so that
  every checkpoint of one run is comparable; none is annealed). One extra, secondary point: from the
  16k state, 1,000 more steps with lr decaying linearly to 0 ("16k + 1k cooldown"); it is reported but
  is **not** used by the stop rule.
- **Data**: WikiText-2 **train**, 512-token chunks (`e_common.chunks`; 4,916 chunks in all). Excluded:
  chunks 0–255 (predictor fitting), 400–415 (calibration), 600–615 (holdout); the remaining 4,628
  chunks are shuffled each epoch with a fixed seed (seed 0 + epoch). 16k steps × 512 tokens = 8.2M
  tokens ≈ 3.5 epochs. **Multi-epoch caveat**: later checkpoints have seen each training chunk up to
  4 times; train KL may fall partly from memorisation, so holdout KL is the measure. Test split unused.
- **Checkpoints**: step 0 (untrained), 1k, 2k, 4k, 8k, 16k (+ cooldown). At each: calibration and
  holdout KL vs the original exact model, top-1 agreement, perplexity change, **with speculation on**,
  and the same **with speculation off** (drift of the healed weights from the original model).
- **Memory**: process capped at 50% of the 32 GB GPU (`set_per_process_memory_fraction(0.5)`); the
  exact model's reference log-probs are recomputed per evaluation chunk instead of being held.

## Bar (the Phase II gate, unchanged)

Mean KL ≤ 0.01 nats/token **and** perplexity increase ≤ 1.0%, vs the original exact model,
speculation on, on holdout (calibration reported alongside and must also meet it for a full pass).

## Stop rule (fixed before any result)

Applied to setting (a), rank 16, holdout, speculation on, at the constant-lr checkpoints:

1. **Bar met**: some checkpoint meets the bar on holdout (and calibration) → PLE-as-architecture is
   viable at this coverage; report the first such checkpoint.
2. **Stop for good (flattening above the bar)**: at 16k steps holdout KL > 0.03 (3× the bar) **and**
   holdout KL fell by < 30% from 4k to 16k steps, i.e. KL_16k > 0.7 · KL_4k. Then no longer run of this
   recipe is justified (a further 4× of training would at best take ~30% off a KL that is already > 3×
   the bar).
3. **Still improving**: neither 1 nor 2. Report a power-law fit of holdout KL vs tokens over the 4k,
   8k and 16k points and the extrapolated tokens to KL 0.01, with the caveat that WikiText-2 train
   cannot supply them without more epochs (a larger corpus would be needed; none is downloaded here).

Secondary readings, also fixed now:

- **Capacity** (setting b vs a): if the higher rank's holdout KL at 16k (or at the furthest checkpoint
  both reach) is ≤ 0.8× rank 16's, adapter capacity is a limiting factor; otherwise it is not.
- **Overfitting**: if holdout KL rises from 8k to 16k while train KL keeps falling, report it as
  multi-epoch overfitting (rule 2 still applies as written).

## Results (2026-11-24): stop rule 2 fires, flattening far above the bar

Script: `scripts/e3b_healing_scaling.py`. Raw: `results/processed/e1_e5/e3b/Qwen3-4B-Base_e3b_lora_r{16,64}.json`
(every checkpoint, both splits); plot `results/processed/e1_e5/e3b/e3b_healing_scaling.svg` (SVG, not
PNG: matplotlib is not installed in the venv and nothing was downloaded). Logs and adapter weights:
`results/raw/e1_e5/e3b/`. Exact-model holdout ppl recomputed per chunk at every checkpoint.

**Step 0.** Identity speculation at all 35 boundaries, untrained: holdout KL 1.647, +396% ppl
(calibration 1.508, +333%). This is ~8× worse than E3's ridge starting point (KL 0.218, +18.2%): the
identity guess is very poor at the early boundaries.

**(a) LoRA rank 16** (33.0M params), holdout (calibration in brackets), vs the original exact model:

| Step | Tokens | Epochs | Spec on: KL | top-1 | ppl change | Spec off (drift): KL | ppl change | Train KL (last 100) |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 1.647 (1.508) | 48.5% | +396% (+333%) | 0 | 0 | |
| 1k | 0.51M | 0.22 | 0.2145 (0.2315) | 79.0% | +14.9% (+16.1%) | 0.094 (0.098) | +2.9% | 0.230 |
| 2k | 1.02M | 0.43 | 0.2062 (0.2197) | 79.8% | +13.9% (+15.0%) | 0.102 (0.107) | +3.3% | 0.207 |
| 4k | 2.05M | 0.86 | 0.1956 (0.2122) | 79.8% | +11.8% (+14.3%) | 0.109 (0.110) | +3.6% | 0.210 |
| 8k | 4.10M | 1.73 | 0.1869 (0.2152) | 80.6% | +10.1% (+14.7%) | 0.116 (0.124) | +3.5% | 0.157 |
| 16k | 8.19M | 3.46 | **0.1887** (0.2150) | 80.7% | **+10.4%** (+15.2%) | 0.125 (0.144) | +5.1% | 0.111 |
| 16k + 1k cooldown | 8.70M | 3.67 | 0.1744 (0.2000) | 81.3% | +9.4% (+13.2%) | 0.114 (0.136) | +4.6% | 0.099 |

**(b) LoRA rank 64** (132M params, scale 1.0), holdout (calibration):

| Step | Spec on: KL | ppl change | Spec off: KL | ppl change | Train KL |
|---|---|---|---|---|---|
| 1k | 0.2295 (0.2435) | +17.2% (+17.9%) | 0.109 | +3.9% | 0.242 |
| 2k | 0.2286 (0.2472) | +16.9% (+18.7%) | 0.124 | +5.3% | 0.228 |
| 4k | 0.2170 (0.2344) | +14.2% (+16.6%) | 0.128 | +4.3% | 0.234 |
| 8k | 0.2186 (0.2489) | +14.6% (+18.5%) | 0.146 | +6.3% | 0.178 |
| 16k | 0.2234 (0.2594) | +14.3% (+19.4%) | 0.161 | +6.8% | 0.136 |
| 16k + 1k cooldown | 0.2033 (0.2367) | +12.3% (+17.1%) | 0.146 | +5.9% | 0.113 |

Wall clock: ≈ 0.29 s per training step (512 tokens, teacher + student forward, backward) on an idle
GPU for both ranks; rank 16's first ~2.5k steps shared the GPU with another job (1.1–1.4 s/step), so
its JSON average is 0.385 s/step. Peak memory 12.4 GB (rank 16), 14.1 GB (rank 64), no gradient
checkpointing. Evaluation (2 × 16 chunks, three forwards each) ≈ 6–7 s per checkpoint.

**(c) Full-rank.** Full fine-tuning with Adam does not fit 16 GB for a 4B model. Tried: a full-rank
trainable BF16 delta on every q/k/v (566M params; LoRA rank 16 on o/MLP). With AdamW (two BF16 state
tensors per weight) it ran out of memory at step 2 under the 15.9 GiB cap. With an Adafactor-style
optimiser (factored second moment, no first moment; `FactoredRMS` in the script) it fits (peak 14.4 GB),
but with the other agent's job resident the GPU was full (31.9 / 32.6 GB) and it ran at 5.3 s/step, so
it was stopped after ~100 steps (train KL 0.72 at step 100, similar to LoRA's 0.76). No full-rank
checkpoint was evaluated.

### Verdict against the pre-registered rule

- **Rule 1 (bar met): no.** Best holdout point is KL 0.174 / +9.4% ppl (rank 16, after the cooldown),
  17× the KL bar and 9× the ppl bar; calibration is worse (KL 0.200, +13.2%).
- **Rule 2 (stop for good): fires.** Rank 16 at 16k: holdout KL 0.1887 > 0.03, and it fell only 3.5%
  from 4k (0.1956), far short of 30%. The power-law fit over 4k / 8k / 16k has slope −0.026 in log-log;
  extrapolating to KL 0.01 needs ~10^53 steps, i.e. never. Almost all the gain happens in the first
  1k steps (1.65 → 0.21); after that the curve is flat.
- **Capacity: not the limit.** Rank 64 is *worse* than rank 16 at every checkpoint (16k: 0.223 vs
  0.189, ratio 1.18, criterion was ≤ 0.8).
- **Overfitting: yes, mildly.** From 8k to 16k rank 16's train KL fell 0.157 → 0.111 while holdout
  KL rose 0.1869 → 0.1887 (calibration flat at 0.215); train KL drops visibly each time an epoch
  restarts (step 4,628: 0.21 → 0.14). Rank 64 shows the same more strongly (holdout 0.217 → 0.223
  from 4k to 16k).
- **Drift grows with training.** With speculation off, the healed weights move steadily away from
  the original model (rank 16: KL 0.094 at 1k → 0.125 at 16k, ppl +2.9% → +5.1%); the adapters are
  increasingly fitting the training text rather than the speculation error.
- **Cooldown** (secondary, outside the rule): decaying lr to 0 over 1k steps takes ~8% off holdout KL
  (0.189 → 0.174 at rank 16, 0.223 → 0.203 at rank 64). So some of the plateau is constant-lr noise,
  but annealing does not change the order of magnitude.

**Conclusion.** Training Qwen3-4B-Base to expect a one-block-stale attention input at every boundary
(identity predictor) recovers most of the damage within ~0.5M tokens and then flattens at holdout
KL ≈ 0.17–0.19 / +9–10% ppl, 17–19× the KL bar, with no benefit from 4× more adapter capacity and
signs of multi-epoch overfitting. With adapter healing on WikiText-2-sized data, PLE-as-architecture
at this coverage does not approach the bar; per the stop rule, no longer run of this recipe is
justified. Not tested here: pretraining-scale data (a fresh corpus, many more tokens and no repeated
epochs), full-rank training of all weights, and healing on top of the ridge predictor for longer
(E3's ridge start is already 0.138 after 1k steps).
