# Where does a Qwen3-4B token's time go, and what can each part give back? (pre-registered 2026-11-24)

Written before any of these runs. Bars do not change after results are seen; changes get a dated
amendment with every affected run redone.

## Why

Self-speculative decoding failed because a draft step costs most of a full step (0.48–0.82) and the
verify pass ~1 step (`docs/self_speculative_decoding.md`). The exact decode step itself is 8.7 ms/token
on Qwen3-4B BF16 (`src/ple/decode.py`, CUDA graph, 512-token prompt), while streaming its ~8 GB of
weights at the RTX 5090's 1.79 TB/s would take ~4.5 ms. So before any new speculation idea, find
where the step loses time, part by part, for decode (one token) and prefill (the 512-token prompt),
and test one fix per part.

## Method

1. **Profile.** CUDA kernel times inside the graph-captured step (torch.profiler / CUPTI), grouped
   by component: embedding, norms, q/k/v GEMM, q/k norm + RoPE, KV write, attention, o GEMM,
   gate/up GEMM, SiLU·mul, down GEMM, residual adds, final norm + LM head, argmax. Report each
   component's time, its bytes moved and achieved bandwidth (decode) or FLOP rate (prefill), and
   the gap between summed kernel time and wall time (launch / idle). Same for prefill.
2. **Kernel microbenchmarks** at decode shapes (1 row) and prefill shapes (512 rows) for every GEMM
   in the model: BF16 cuBLAS (current), weight-only INT8 and INT4 kernels available in this PyTorch
   build, and FP8 `_scaled_mm`. Reported: µs and achieved bandwidth.
3. **One lever per component**, each measured end to end in the E1 harness against the current
   decoder, interleaved (A B A B …, 5 pairs of 128-token runs), so GPU contention hits both.

## Levers

Lossless (output must stay identical to the current exact decoder under the self-spec teacher-forced
check: every token the full model's argmax unless the top-2 gap ≤ 0.25):

- **L1 launch overhead / fusion**: fewer, fused kernels (torch.compile of the step if it works on
  this Windows build, else hand-fused elementwise: norm+RoPE, SiLU·mul, residual+norm).
- **L2 attention**: attend over the filled cache length only (now the whole `max_len` with a mask),
  and a flash / efficient kernel instead of the masked SDPA path; prefill with `is_causal`.
- **L3 GEMV efficiency**: alternatives to the current cuBLAS BF16 GEMV at 1 row (layout, fused
  q/k/v/o or gate/up/down shapes, split-K), if the profile shows < 75% of peak bandwidth.
- **L4 self-spec loop**: remove the per-round host sync (check acceptance every few rounds).
- **L5 self-spec draft head**: draft LM head over a frequency-ranked vocabulary subset (verify keeps
  the full head, so output stays exact).

Lossy (quality against the original BF16 model, the Phase II bar: KL ≤ 0.01 and ≤ 1% perplexity on
WikiText-2 train chunks 400–415 and 600–615):

- **L6 weight-only INT8 / INT4 baseline** (every linear layer; LM head separately).
- **L7 quantized self-draft**: the L6 model as the self-spec draft, full BF16 model verifying, so the
  output stays exact; reported with the self-spec bars (≥ 1.3× holdout, ≥ 1.1× code).

## Bars

- A lossless lever **counts** if it lowers end-to-end decode ms/token (or prefill ms) by ≥ 5% versus
  the current decoder, with the pairwise ratio's spread across the 5 pairs < 2%, and output is exact.
- A lossy lever counts if it meets the quality bar on both splits and saves ≥ 5%.
- L7 is judged on the self-spec speed bars.
- Levers are reported one at a time against the current decoder and then stacked; the stack is
  reported but not judged.

## Amendments

- 2026-11-24, before the full runs: a 1-prompt-per-domain smoke test of `scripts/lever_bench.py`
  was seen (decode L1 1.07×, L2 1.29×, L1+L2 1.38×, all exact; INT4 +16–19% ppl; INT4 self-draft
  k = 3 1.45–1.56× over the L1+L2 decoder). The full runs below use 4 prompts per domain for the
  decoder levers and 8 for self-spec, as planned; L4 is run at every 1 and 2 rounds (every 4 was
  dropped after the smoke test showed no effect at 2 or 4). L3 (alternative BF16 GEMV kernels) is
  settled by the kernel microbenchmark: pre-transposed weights are slower and cuBLAS is at 78–87% of
  peak bandwidth already, so no end-to-end run. L5 (vocabulary-subset draft head) is not run: with an
  INT4 draft the LM head is ~0.14 ms of the draft step. Bars unchanged.

## Results

Raw files under `results/processed/decode_levers/`: `Qwen3-4B-Base_profile.json`,
`attention_kernels.json`, `gemm_kernels.json` and
`Qwen3-4B-Base_levers_decode_prefill_quality_selfspec.json` (`scripts/lever_bench.py`, run 2026-11-24
on an idle GPU; ComfyUI held ~2.6 GB and had an empty queue throughout). Speedups are means over
prompts of each prompt's median pairwise ratio; spread is (max − min) / median of a prompt's 5 pair
ratios, worst prompt reported.

### Profile (decode step, 512-token context, CUDA graph)

| Component | ms | Share | Note |
|---|---|---|---|
| GEMMs (q/k/v 0.81, o 0.53, gate/up 2.35, down 1.22, LM head 0.50) | 5.41 | 65% | 78–87% of peak bandwidth |
| Attention (masked SDPA over all `max_len` slots) | 1.78 | 22% | 3% of peak bandwidth |
| q/k norm + RoPE (~14 kernels / layer) | 0.64 | 8% | |
| Norms, KV write, SiLU·mul, adds | 0.40 | 5% | |
| Idle between kernels | 0.001 | 0% | nothing for L1 to win from launch overhead |

### Kernels

- Attention, 1 query row: cuDNN SDPA 10–11 µs per layer against 52 µs for the default masked path.
  2–8 rows (verify): batched-matmul GQA 16.5 µs, cuDNN 21 µs, default 51 µs. Flash attention is not
  built into this Windows PyTorch.
- GEMMs at 1 row: INT4 weight-only (tinygemm, group 128) 3–3.5× faster than BF16; INT8 weight-only
  4–9× **slower**; FP8 per-tensor ~2× faster on the large shapes. Pre-transposed BF16 weights are
  slower (L3 closed). At 512 rows INT4 is several times slower; FP8 reaches 220–289 TFLOP/s against
  150–180 for BF16 (1.5–1.7×, ~4% relative error, not run end to end).

### Decode (4 prompts × 3 domains, 128 tokens, 5 interleaved pairs)

| Lever | ms/token (original 8.71) | Speedup | Range over prompts | Worst spread | Violations | Verdict |
|---|---|---|---|---|---|---|
| L1 fused q/k norm + RoPE | 8.23 | **1.058×** | 1.054–1.063 | 0.8% | 0 | **counts** |
| L2 fast attention | 6.87 | **1.269×** | 1.265–1.280 | 0.9% | 0 | **counts** |
| L1 + L2 (stack, not judged) | 6.42 | 1.357× | 1.354–1.367 | 0.9% | 0 | |
| L6 INT4 every linear + head | 2.65 | 3.284× | 3.275–3.302 | 1.2% | 104 | fails quality |

Near-ties (mismatches with a top-2 gap ≤ 0.25): 47 for the original decoder itself, 45–53 for the
levers, so the fused and fast paths move ties no more than the original decode path does against the
prefill judge.

### Prefill (512 tokens, eager, 5 interleaved pairs)

| Lever | ms (original ~37.6) | Speedup | Range | Worst spread | KL vs original | Verdict |
|---|---|---|---|---|---|---|
| L1 fused RoPE | 36.27 | 1.034× | 1.027–1.042 | 4.9% | 1.4e-3 | below 5% |
| L2 causal SDPA over the prompt | 35.12 | 1.071× | 1.059–1.084 | 6.2% (12 of 12 prompts over 2%) | 0 | **does not count: spread** |
| L1 + L2 (stack) | 34.11 | 1.108× | 1.101–1.118 | 3.4% | 1.4e-3 | |
| L6 INT4 | 189.4 | 0.20× | | | 0.23 | fails |

L2's prefill gain is above 5% on every prompt, but eager timing of a single ~37 ms prefill varies
more than the 2% the bar allows, so by the bar it does not count. A CUDA-graph prefill would settle it
(not run).

### Quality of L6 (INT4 RTN, group 128, every linear and the LM head)

| Split | KL | Top-1 agreement | Perplexity BF16 → INT4 |
|---|---|---|---|
| Calibration (chunks 400–415) | 0.177 | 0.796 | 7.61 → 8.85 (+16.4%) |
| Holdout (chunks 600–615) | 0.186 | 0.796 | 10.53 → 12.53 (+19.0%) |

Both far outside KL ≤ 0.01 and ≤ 1% perplexity. **L6 fails** as a replacement model.

### Self-speculation: L4 (check every 2 rounds) and L7 (INT4 draft)

8 prompts per domain, 256 tokens, 2 interleaved timing pairs. Speedup against the L1 + L2 exact
decoder (6.45 ms/token) and, in brackets, the original E1 decoder (8.73 ms/token).

| Config | k | Cal | Holdout | Code | Holdout (orig) | Code (orig) | Accept | Violations | Near-ties |
|---|---|---|---|---|---|---|---|---|---|
| BF16 draft, every 1 | 2 | 0.948 | 0.950 | 0.950 | 1.28 | 1.28 | 99% | 1 | 137 |
| BF16 draft, every 1 | 3 | 0.952 | 0.954 | 0.963 | 1.29 | 1.30 | 99% | 1 | 137 |
| BF16 draft, every 1 | 4 | 0.943 | 0.946 | 0.947 | 1.28 | 1.28 | 99% | 0 | 133 |
| BF16 draft, every 1 | 6 | 0.944 | 0.943 | 0.952 | 1.28 | 1.29 | 98% | 0 | 133 |
| BF16 draft, every 2 | 2 | 0.949 | 0.951 | 0.950 | 1.29 | 1.28 | 99% | 1 | 137 |
| BF16 draft, every 2 | 3 | 0.952 | 0.958 | 0.964 | 1.30 | 1.30 | 99% | 1 | 137 |
| BF16 draft, every 2 | 4 | 0.946 | 0.945 | 0.947 | 1.28 | 1.28 | 99% | 0 | 133 |
| BF16 draft, every 2 | 6 | 0.945 | 0.946 | 0.953 | 1.28 | 1.29 | 98% | 0 | 133 |
| INT4 draft, every 1 | 2 | 1.358 | 1.391 | 1.438 | 1.88 | 1.94 | 88% | 1 | 137 |
| INT4 draft, every 1 | 3 | 1.406 | 1.466 | 1.522 | 1.98 | 2.06 | 83% | 1 | 137 |
| INT4 draft, every 1 | 4 | 1.397 | 1.446 | 1.555 | 1.96 | 2.11 | 82% | 0 | 133 |
| INT4 draft, every 1 | 6 | 1.372 | 1.468 | 1.593 | 1.99 | 2.16 | 75% | 0 | 133 |
| INT4 draft, every 2 | 2 | 1.360 | 1.394 | 1.439 | 1.89 | 1.95 | 88% | 1 | 137 |
| INT4 draft, every 2 | 3 | 1.402 | 1.469 | 1.524 | 1.99 | 2.06 | 83% | 1 | 137 |
| **INT4 draft, every 2** | **4** | **1.402** | **1.447** | **1.555** | **1.96** | **2.10** | 81% | **0** | 133 |
| INT4 draft, every 2 | 6 | 1.377 | 1.472 | 1.595 | 1.99 | 2.16 | 75% | 0 | 133 |

**The one violation.** It is the same token in all eight k = 2 and k = 3 settings: calibration prompt
2, generated position 73, where the verify pass emitted "0" and the prefill judge's argmax is "8" with
a top-2 gap of 0.375 (three BF16 steps at a logit of ~31; the allowance is 0.25). It does not depend
on the draft: BF16 and INT4 drafts give the same violation and identical near-tie counts (137 at
k = 2–3, 133 at k = 4–6), because in greedy speculation every emitted token is the verify pass's
argmax. It is rounding in the 3- and 4-row verify pass, not a draft error. Under the pre-registered
check those eight settings still fail exactness, so the pick is made among the settings with zero
violations.

**Pick.** As in `docs/self_speculative_decoding.md`, the setting is picked on calibration and judged
on holdout and code: among the zero-violation settings the best calibration speedup is the INT4 draft
checked every 2 rounds with k = 4 (1.402×; every-1 k = 4 is 1.397×). It gives **1.447× on holdout and
1.555× on code** over the L1 + L2 decoder (4.42 ms/token; per-prompt range 1.23–1.68×), and 1.96× /
2.10× over the original decoder.

**Verdicts.**

- **L7 passes** the self-spec bars (≥ 1.3× holdout, ≥ 1.1× code) with exact output at the picked
  setting. This is the speedup the skip-layer draft could not reach (0.94×): an INT4 draft step costs
  about a third of a full step, and 75–88% of its tokens are accepted.
- **L4 does not count**: checking acceptance every 2 rounds instead of every round changes speed by
  under 0.5% in every setting.
- The BF16 no-skip draft (a full-cost draft, as a control) runs at 0.94–0.96×, as it should.

### Summary

| Part | Lever | Result | Verdict |
|---|---|---|---|
| Decode | L1 fused q/k norm + RoPE | 1.058× | counts |
| Decode | L2 fast attention | 1.269× | counts |
| Decode | L1 + L2 | 1.357× (8.71 → 6.42 ms/token), exact | stack, not judged |
| Decode | L3 GEMV alternatives | cuBLAS already 78–87% of peak; alternatives slower | closed by microbenchmark |
| Decode | L4 sparse acceptance checks | < 0.5% | does not count |
| Decode | L5 vocabulary-subset draft head | not run | |
| Decode | L6 INT4 model | 3.28× but +16–19% perplexity | fails quality |
| Decode | L7 INT4 self-draft | 1.45× holdout, 1.56× code over L1 + L2 (≈2× over original), exact | **passes** |
| Prefill | L1 | 1.034× | below 5% |
| Prefill | L2 | 1.071×, spread 6% | does not count (spread) |

## Follow-up: better INT4, and replication (pre-registered 2026-11-24, before any of these runs)

Written after the results above and before any run below. Prefill is dropped (decode only from here,
at the user's request: no prefill lever cleared the bar).

**A. INT4 quality (L6b).** Every linear layer and the LM head in INT4, variants: round to nearest
(RTN) with groups of 128 (the L6 reference), 64 and 32; RTN with a per-group clipping search (range
shrunk by 0–30% in 2.5% steps, lowest squared rounding error) at 128 and 32; GPTQ at 128 and 32;
GPTQ with the clipping search at 32. Then the variant with the lowest calibration KL again with the
LM head kept in BF16. GPTQ statistics come from WikiText-2 train chunks 0–127 (512 tokens each),
disjoint from the quality chunks (400–415, 600–615) and the self-spec prompts. Quality as before on
both splits; decode ms/token timed for every variant (group size changes the kernel). L6 bar
unchanged: KL ≤ 0.01 and ≤ 1% perplexity on both splits, and ≥ 5% faster than the original decoder.

**B. INT4 draft (L7b).** The two variants with the lowest calibration KL run as self-spec drafts with
the L1 + L2 decoder verifying: k ∈ {3, 4, 6}, acceptance checked every round, 8 prompts per domain,
256 tokens, 2 interleaved pairs, the same exactness check. Pick on calibration speed among
zero-violation settings, judged on holdout and code. A variant **improves on L7** if its holdout and
code speedups over the L1 + L2 decoder both exceed L7's pick (1.447×, 1.555×) by ≥ 3%. It must also
pass the self-spec bars (≥ 1.3× holdout, ≥ 1.1× code).

**C. What is left in the step.** Profile the L1 + L2 decode step by component, as in the profile
above. Reported only; any new lever gets its own amendment before it runs.

**D. Replication.**
1. Qwen3-4B again in a fresh process: the decode part (L1, L2, L1 + L2; 4 prompts per domain, 5
   pairs) and L7's picked setting (INT4 RTN g128 draft, k = 4, every 2 rounds). It **replicates** if
   every speedup is within 3% of the first run and output is exact.
2. Another model. No other Qwen3-architecture model is on disk. Qwen3.5-4B and Ornith-1.5-9B use
   Gated DeltaNet linear attention in 24 of 32 layers, which this decoder does not implement.
   TinyLlama-1.1B (Llama, on disk) is run with the decoder extended to Llama blocks; Llama has no q/k
   norm, so L1 is the RoPE fusion alone. Same protocol and bars as the first run (decode levers and
   L7 at k ∈ {3, 4, 6}). Another Qwen3 size (1.7B or 8B) needs a download and runs only if the user
   approves it, under the same rules.

## Follow-up results (2026-11-24)

### A. INT4 quality (L6b)

`results/processed/decode_levers/Qwen3-4B-Base_int4_quality.json`. Violations: greedy INT4 decoding,
4 prompts per domain × 128 tokens (1,536 tokens), judged by the original BF16 decoder. ms/token:
INT4 decoding with fast attention and fused RoPE, mean over one prompt per domain.

| Variant | Cal KL | Cal ppl | Hold KL | Hold ppl | Hold top-1 | Violations | ms/token | × original |
|---|---|---|---|---|---|---|---|---|
| RTN g128 (L6) | 0.1769 | +16.40% | 0.1863 | +19.05% | 0.796 | 106 | 2.580 | 3.36× |
| RTN g64 | 0.1296 | +13.52% | 0.1338 | +13.78% | 0.813 | 92 | 2.721 | 3.23× |
| RTN g32 | 0.1014 | +8.84% | 0.1037 | +9.36% | 0.830 | 66 | 2.920 | 3.04× |
| Clip g128 | 0.1641 | +15.58% | 0.1709 | +15.89% | 0.789 | 87 | 2.646 | 3.31× |
| Clip g32 | 0.1035 | +9.95% | 0.1166 | +11.79% | 0.810 | 74 | 2.883 | 3.02× |
| GPTQ g128 | 0.0774 | +6.58% | 0.0647 | +4.34% | 0.887 | 70 | 2.612 | 3.33× |
| GPTQ g32 | 0.0462 | +4.24% | 0.0344 | +3.03% | 0.908 | 46 | 2.878 | 3.05× |
| GPTQ + clip g32 | 0.0420 | +3.42% | 0.0340 | +2.76% | 0.911 | 46 | 2.912 | 3.04× |
| GPTQ + clip g32, BF16 head | 0.0380 | +3.09% | 0.0308 | +2.26% | 0.917 | 43 | 3.236 | 2.72× |

GPTQ does most of the work: at group 128 it halves calibration KL and cuts holdout KL by 65%. Smaller
groups add about as much again. The clipping search alone barely helps (KL 0.177 → 0.164 at 128,
and slightly worse than plain RTN at 32), and adds only a little on top of GPTQ. Keeping the LM
head in BF16 helps quality a little and costs 0.32 ms/token. **L6b does not count:** the best
variant is still ~3× over the KL budget (0.031–0.038 against 0.01) and +2.3–3.1% perplexity against
1%.

### B. INT4 draft (L7b)

`results/processed/decode_levers/Qwen3-4B-Base_int4_draft.json`. Speedups over the L1 + L2 decoder
(and over the original decoder), medians over 8 prompts per domain, 256 tokens, acceptance checked
every round. *GPTQ g128 was added after the quality results were seen (it has the fastest kernel of
the GPTQ variants); it is exploratory and not judged against the rule.*

| Draft | k | Cal | Holdout | Code | Holdout (orig) | Code (orig) | Accept cal / hold / code | Violations |
|---|---|---|---|---|---|---|---|---|
| GPTQ + clip g32, BF16 head | 3 | 1.407× | 1.445× | 1.444× | 1.958× | 1.961× | 91 / 95 / 95% | 1 |
| | 4 | 1.400× | 1.440× | 1.496× | 1.955× | 2.020× | 90 / 93 / 97% | 0 |
| | 6 | 1.439× | 1.471× | 1.557× | 1.990× | 2.104× | 87 / 89 / 94% | 0 |
| GPTQ + clip g32 | 3 | 1.485× | 1.531× | 1.531× | 2.073× | 2.073× | 91 / 94 / 94% | 1 |
| | 4 | 1.475× | 1.519× | 1.572× | 2.059× | 2.130× | 88 / 91 / 95% | 0 |
| | **6** | **1.536×** | **1.572×** | **1.662×** | **2.130×** | **2.254×** | 85 / 87 / 94% | 0 |
| *GPTQ g128 (exploratory)* | 3 | 1.507× | 1.573× | 1.595× | 2.131× | 2.161× | 86 / 92 / 93% | 1 |
| | 4 | 1.509× | 1.593× | 1.643× | 2.158× | 2.226× | 84 / 90 / 94% | 0 |
| | 6 | 1.587× | 1.656× | 1.751× | 2.241× | 2.370× | 82 / 87 / 92% | 0 |

The one violation at k = 3 is the same token as in the first run (calibration prompt 2, the verify
pass's rounding at a near-tie), for every draft.

**Pick:** GPTQ + clip g32 at k = 6 (best calibration among the zero-violation settings of the two
pre-registered drafts). Holdout 1.572× is 8.6% above L7's 1.447×, and code 1.662× is 6.9% above
1.555×. Both clear the 3% margin, and the self-spec bars hold with exact output. **L7b improves on
L7:** 2.13× (holdout) and 2.25× (code) over the original decoder, against 1.96× / 2.10×.

Caveats. L7's reference numbers come from the first run's process; D1 re-measures L7 in a fresh
process. The best k is the largest tried, so k > 6 may be faster still (not tested). A better draft
raises acceptance but its own speed matters as much: the BF16 head adds 0.32 ms per draft token and
loses more than its extra acceptance gains, and the exploratory GPTQ g128 draft, with lower
acceptance but a faster kernel, is faster than both.

### C. What is left in the step

`results/processed/decode_levers/Qwen3-4B-Base_profile_fast.json`. One decode step at position ~512,
GPU time by kernel group.

| Component | L1 + L2 step (6.21 ms busy) | INT4 draft step, RTN g128 (2.60 ms busy) |
|---|---|---|
| Weight GEMMs | 5.25 ms (85%) | 1.66 ms (64%) |
| RMSNorm | 0.29 ms | 0.29 ms |
| Attention | 0.27 ms | 0.26 ms |
| Elementwise (residual adds, SiLU·mul, RoPE) | 0.26 ms | 0.26 ms |
| KV write / index | 0.14 ms | 0.14 ms |

Attention fell from 1.78 ms to 0.27 ms, so the verifier step is now mostly weight streaming: 0.96 ms
of non-GEMM work is left, a ceiling of ~15% for further fusion. In the draft step the same 0.95 ms
is 37% of the step, so fusing norm, residual and KV write would speed up the draft (and so L7b) more
than the verifier.

### D1. Replication on Qwen3-4B (fresh process)

`results/processed/decode_levers/Qwen3-4B-Base_levers_decode_selfspec_rep1.json`. Mean speedup over
the original decoder, 12 prompts (decode); L7's pick (INT4 RTN g128 draft, k = 4, every 2 rounds),
8 prompts per domain.

| | First run | D1 | Change | Violations (first / D1) |
|---|---|---|---|---|
| L1 fused RoPE | 1.058× | 1.060× | +0.2% | 0 / 0 |
| L2 fast attention | 1.269× | 1.273× | +0.3% | 0 / **1** |
| L1 + L2 | 1.357× | 1.361× | +0.3% | 0 / 0 |
| L1 + L2 + L6 INT4 (lossy) | 3.284× | 3.300× | +0.5% | 104 / 113 |
| L7 cal (over L1 + L2) | 1.402× | 1.402× | 0.0% | 0 / 0 |
| L7 holdout | 1.447× | 1.452× | +0.3% | 0 / 0 |
| L7 code | 1.555× | 1.606× | **+3.3%** | 0 / 0 |

**Verdict by the rule as written: D1 does not fully replicate.** Two items miss it:

1. **L7 code is 3.3% faster, outside 3%, but the prompts were not the same.** The code prompts are cut
   from this repo's own `.py` files, and scripts and tests were added between the runs, so every code
   prompt moved. Calibration and holdout prompts are fixed WikiText chunks and reproduce prompt by
   prompt (per-prompt L7 speedups agree to ±0.01). The first run's code prompts cannot be rebuilt
   (those files were not committed), so this item cannot be re-tested like for like.
2. **L2 had one violation** (code prompt 3), where the first run had none. Diagnosis (scratchpad
   scripts, not a judged run):
   - The original decoder is bitwise repeatable, run to run and replay to replay.
   - The fast path is not. Even replaying one CUDA graph gives different logits from step 1 or 3
     on, and eager steps vary too.
   - The cause is cuDNN's one-row SDPA (the kernel L2 uses for single-token steps). An
     attention-only repro of the decode loop, with identical inputs and no other kernels, gives
     different bits in 4 of 5 repeats. The math backend gives identical bits in 5 of 5. Masked
     cache slots do not matter, and the size of the variation is BF16 rounding (≤ 0.002).
     `cudnn.deterministic` has no effect.
   - With the batched-matmul kernel for one-row steps, all replays and fresh graphs are bitwise
     equal.
   - So at a near-tie the fast step can pick either token, and the continuation then differs.
     Where the lossless variants first leave the original's sequence changes between runs
     (e.g. calibration prompt 0, L2: token 58 in the first run, 14 in D1).
   - The D1 violation is one such flip that the 0.25 near-tie cutoff does not cover. Caught again
     in a repeat, it was a top-2 gap of 0.5 at a logit of 32.5, two BF16 ulps at |logit| 32–64.
     The cutoff stays at 0.25, so it counts.

What this means: L1, L2 and L1 + L2 are reproducible in speed (within 0.5% on identical prompts).
Their output is the original's up to BF16 rounding at near-ties, but with L2 it is not bitwise
reproducible run to run. Self-spec output (L7, L7b) does not depend on this: emitted tokens come
from the verify pass, which uses the batched-matmul kernel (2+ rows). cuDNN runs only in draft
steps, where it can change acceptance but not output.

## Amendment 2026-11-24 (after D1, before any run below)

1. **Frozen code prompts.** From now on code prompts are cut from the `.py` files in `src`, `scripts`
   and `tests` at commit `a118bda` (`--code-rev a118bda`), so edits to the repo no longer change
   them. Earlier runs keep their working-tree prompts; numbers before and after this change are not
   compared on code.
2. **L2d, deterministic fast attention (new decode variant, lossless).** L1 + L2 with one-row steps
   on the batched-matmul kernel instead of cuDNN (`Qwen3Decoder.step_attn = "bmm"`). Microbenchmark
   (Qwen3-4B shapes, CUDA graph, before the decode run): 17.9 µs per layer against 6.8 µs for cuDNN,
   so expect ~0.4 ms per token slower than L1 + L2. Run in the decode part on Qwen3-4B (frozen code
   prompts) and TinyLlama. Judged like L1 / L2: counts if ≥ 5% faster than the original, worst
   spread ≤ 2%, no violations. Also reported: bitwise equality of its tokens across the 5 timing
   pairs is not checked by the benchmark, so the claim of determinism rests on the diagnosis above.
3. **D2 (TinyLlama) runs both L7 configurations** of the first run's protocol (every 1 and every 2
   rounds), k ∈ {3, 4, 6}, plus L2d. Pick and bars as for L7.
4. **D3 (TinyLlama L7b).** The L7b pick (GPTQ + clip g32 draft, every round, k ∈ {3, 4, 6}) on
   TinyLlama, GPTQ statistics from WikiText-2 train chunks 0–127 as for Qwen3-4B. It **replicates
   L7b** if its pick (best calibration among zero-violation settings) beats D2's L7 pick by ≥ 3% on
   both holdout and code. Its own quality sweep is not run (cost); the variant is taken from the
   Qwen3-4B result.

## Results after the amendment (2026-11-24)

### L2d, deterministic fast attention

`results/processed/decode_levers/Qwen3-4B-Base_levers_decode_l2d.json` (Qwen3-4B, frozen code prompts,
12 prompts, 5 pairs) and the TinyLlama file below.

| Model | L1 + L2 | L1 + L2d | L2d cost | L2d spread | L2d violations | Verdict |
|---|---|---|---|---|---|---|
| Qwen3-4B | 1.361× (6.40 ms) | 1.288× (6.76 ms) | −5.4% (+0.37 ms) | 0.9% | 0 | counts |
| TinyLlama-1.1B | 1.603× (1.91 ms) | 1.431× (2.14 ms) | −10.7% (+0.23 ms) | 1.0% | 0 | counts |

L2d counts on both models (≥ 5% over the original, spread ≤ 2%, no violations). It costs what the
microbenchmark predicted on Qwen3-4B (0.37 ms against ~0.4). On TinyLlama, where attention is a larger
share of a much smaller step, it gives back about a third of L1 + L2's gain. **Use:** L1 + L2 when speed
matters most; L1 + L2d when greedy output must be bitwise repeatable. The same rerun also repeats
L1 / L2 / L1 + L2 on Qwen3-4B a third time: 1.058× / 1.274× / 1.361×, no violations (L2's D1 violation
did not recur).

### D2. Replication on TinyLlama-1.1B

`results/processed/decode_levers/TinyLlama-1.1B-intermediate-step-1431k-3T_levers_decode_selfspec.json`.
Llama blocks (22 layers, hidden 2048, 32 heads, 4 KV heads, head dim 64, no q/k norm), frozen code prompts.

| Lever | TinyLlama | Qwen3-4B (first run) | Spread | Violations | Verdict |
|---|---|---|---|---|---|
| L1 (RoPE fusion only) | 1.081× | 1.058× | 0.8% | 0 | counts |
| L2 fast attention | 1.433× | 1.269× | 0.9% | 0 | counts |
| L1 + L2 | 1.603× (1.91 ms/token) | 1.357× | 1.2% | 0 | counts |
| L1 + L2 + L6 INT4 (lossy, quality not run) | 3.051× | 3.284× | 0.8% | 55 | — |

| L7 config | k | Cal | Holdout | Code | Holdout (orig) | Code (orig) | Accept cal / hold / code | Violations |
|---|---|---|---|---|---|---|---|---|
| every 1 | 3 | 1.230× | 1.224× | 1.231× | 1.944× | 1.955× | 95 / 94 / 94% | 0 |
| | 4 | 1.291× | 1.281× | 1.303× | 2.033× | 2.069× | 92 / 91 / 93% | 0 |
| | 6 | 1.378× | 1.343× | 1.367× | 2.132× | 2.171× | 91 / 88 / 90% | 0 |
| every 2 | 3 | 1.235× | 1.226× | 1.233× | 1.948× | 1.961× | 95 / 94 / 95% | 0 |
| | 4 | 1.293× | 1.284× | 1.304× | 2.040× | 2.074× | 92 / 91 / 93% | 0 |
| | **6** | **1.380×** | **1.346×** | **1.369×** | **2.140×** | **2.175×** | 91 / 88 / 90% | 0 |

**D2 replicates on a second architecture.** Every lossless lever counts with exact output, and the gains
are larger than on Qwen3-4B: attention is a bigger share of the step in a 1.1B model. L7's pick (every 2,
k = 6) passes the self-spec bars (1.346× holdout ≥ 1.3, 1.369× code ≥ 1.1) with no violations: **2.14×
(holdout) and 2.18× (code) over the original decoder**. As on Qwen3-4B, checking every 2 rounds changes
speed by < 0.5% (L4 does not count), and the best k is the largest tried.

### D3. L7b on TinyLlama

`results/processed/decode_levers/TinyLlama-1.1B-intermediate-step-1431k-3T_int4_draft.json`. GPTQ + clip
g32 draft, every round.

| k | Cal | Holdout | Code | Holdout (orig) | Code (orig) | Accept cal / hold / code | Violations |
|---|---|---|---|---|---|---|---|
| 3 | 1.204× | 1.194× | 1.190× | 1.917× | 1.909× | 97 / 95 / 95% | 0 |
| 4 | 1.278× | 1.263× | 1.263× | 2.023× | 2.022× | 96 / 94 / 95% | 0 |
| **6** | **1.334×** | **1.336×** | **1.341×** | **2.138×** | **2.147×** | 92 / 92 / 93% | 0 |

**D3 does not replicate L7b.** Its pick (k = 6) is 0.7% below D2's L7 pick on holdout (1.336× against
1.346×) and 2.0% below on code (1.341× against 1.369×), not 3% above. The better draft raises acceptance
(92% against 88% on holdout at k = 6), but RTN INT4 on TinyLlama already accepts 88–95%, so there is
little left to gain, and the group-32 kernel is slower than group 128. L7b's gain is model-dependent: it
helps where the plain INT4 draft is the weak link (Qwen3-4B, acceptance 76–81% at L7's pick) and not where
it is already accurate.

### Where this leaves decode

| | Qwen3-4B | TinyLlama-1.1B |
|---|---|---|
| Original decoder | 8.71 ms/token | 3.05 ms/token |
| L1 + L2 (exact up to rounding at near-ties) | 1.36× | 1.60× |
| L1 + L2d (bitwise repeatable) | 1.29× | 1.43× |
| Best self-spec, exact (holdout / code) | 2.13× / 2.25× (L7b, GPTQ + clip g32, k = 6) | 2.14× / 2.18× (L7, RTN g128, every 2, k = 6) |

### Raw decode speed

Absolute speed of each setting, as measured: RTX 5090, batch 1, BF16 weights unless noted, 512-token
prompt, decode only (prefill excluded), CUDA-graph step. Exact levers: mean over 12 prompts
(4 per domain), 128 tokens, from the L2d run (Qwen3-4B) and D2 (TinyLlama). Self-spec: mean over
8 prompts per domain, 256 tokens. Tokens/s is 1000 / (ms/token). The original decoder is this repo's
own decoder (`src/ple/decode.py`), not a production engine. Rows
marked ‡ are stock Transformers `generate` and rows marked § are llama.cpp, each from a separate run
described below the table.

| Setting | Qwen3-4B ms/token | Qwen3-4B tokens/s | TinyLlama ms/token | TinyLlama tokens/s | Output |
|---|---|---|---|---|---|
| Hugging Face Transformers `generate`, dynamic cache ‡ | 31.43 | 32 | 15.07 | 66 | exact (0 violations) |
| Hugging Face Transformers `generate`, static cache, no compile ‡ | 37.60 | 27 | 17.89 | 56 | exact (0 violations) |
| llama.cpp b12359, BF16 GGUF § | 6.16 | 162 | 1.85 | 540 | exact (0 violations) |
| llama.cpp, BF16 + Q4_0 draft, k = 6 (holdout / code) § | 6.42 / 5.39 | 156 / 185 | 2.25 / 2.30 | 445 / 435 | exact (0 violations) |
| llama.cpp, Q4_0 model § | 2.74 | 365 | 1.10 | 905 | **lossy** (32 / 10 violations per 1,000 tokens) |
| llama.cpp, Q4_K_M model § | 3.00 | 333 | 1.16 | 861 | **lossy** (18 / 7 per 1,000) |
| Original decoder (SDPA over the full cache, reference RoPE) | 8.70 | 115 | 3.05 | 327 | reference |
| L1 fused RoPE | 8.22 | 122 | 2.82 | 354 | exact |
| L2 fast attention | 6.83 | 146 | 2.13 | 469 | exact |
| L1 + L2 | 6.40 | 156 | 1.91 | 525 | exact |
| L1 + L2d (bitwise repeatable) | 6.76 | 148 | 2.13 | 468 | exact |
| L1 + L2 + L6, INT4 RTN g128 model | 2.64 | 379 | 1.00 | 999 | **lossy** (+16–19% ppl on Qwen3-4B; fails the bar; 74 / 36 violations per 1,000 tokens) |
| L7 pick, INT4 RTN g128 draft (holdout / code) | 4.46 / 4.15 | 224 / 241 | 1.43 / 1.41 | 698 / 709 | exact |
| L7b pick, INT4 GPTQ + clip g32 draft (holdout / code) | 4.10 / 3.87 | 244 / 258 | 1.43 / 1.43 | 698 / 701 | exact |
| Exploratory: INT4 GPTQ g128 draft, k = 6 (holdout / code) | 3.89 / 3.68 | 257 / 271 | – | – | exact; not pre-registered |
| BF16 weight-streaming floor (weights ÷ 1.79 TB/s) | ~4.5 | ~222 | ~1.2 | ~860 | bound, not measured |

The L7 picks are k = 4, every 2 rounds (Qwen3-4B), and k = 6, every 2 rounds (TinyLlama). The L7b
picks are k = 6, every round. Calibration-domain self-spec speeds are 4.64 / 4.22 ms (Qwen3-4B, L7 /
L7b) and 1.39 / 1.44 ms (TinyLlama). On Qwen3-4B the INT4 drafts decode faster than the BF16
weight-streaming floor, which no exact one-token-per-step BF16 decoder can beat; on TinyLlama
(about 2.1 GB of weights read per step) they do not.

### Benchmark against Transformers (2026-11-24, reported only, no bar)

‡ Rows marked ‡ come from a separate run (`scripts/hf_baseline_bench.py`,
`results/processed/decode_levers/*_hf_baseline.json`): Transformers 5.24.0 `generate`, greedy, batch 1,
BF16, SDPA attention, on the same 12 prompts (code prompts frozen at a118bda), 128 tokens, 3 timed runs
per prompt after a warm-up, medians. Decode ms/token = (wall time for 128 tokens − wall time for 1
token) / 127, so prefill is excluded. The compiled static-cache path needs Triton, which is not
installed on this Windows build, so `generate` runs eager; this is the stock library, not a serving
engine. Every token Transformers emitted is the full model's greedy choice under the same check as the
levers (0 violations on both models), so the two decoders produce the same text.

In the same process and on the same prompts:

| | Qwen3-4B ms/token | Qwen3-4B tokens/s | TinyLlama ms/token | TinyLlama tokens/s |
|---|---|---|---|---|
| Transformers, dynamic cache | 31.43 | 32 | 15.07 | 66 |
| Transformers, static cache, no compile | 37.60 | 27 | 17.89 | 56 |
| Original decoder (this repo) | 8.25 | 121 | 2.82 | 355 |
| L1 + L2 | 6.26 | 160 | 1.90 | 526 |

The original decoder is faster here than in the lever runs (8.25 against 8.70 ms) because its cache
was allocated for 656 positions instead of 840, and it attends over the whole allocated cache; L1 + L2
attends over the filled part only and barely changes. Against stock Transformers `generate`, L1 + L2
is 5.0× faster on Qwen3-4B and 7.9× on TinyLlama, and, across runs, the L7b draft 7.7–8.1× (Qwen3-4B) and the L7 draft 10.5–10.7× (TinyLlama). Most of
that gap is Python and kernel-launch overhead that any CUDA-graph decoder removes, so it says little
about the levers themselves.

Published reference points, not comparable to these numbers and given only for scale:

| Source | Model, setup | Tokens/s |
|---|---|---|
| Qwen speed benchmark (Qwen team docs) | Qwen3-4B BF16, batch 1, input length 1, 2048 generated, **NVIDIA H20**; speed counts prompt and generated tokens | SGLang 133.13; Transformers 45.94 |
| Alpin Dale, "Hitting 1,000 tokens per second on a single RTX 5090" (Feb 2026) | **Qwen3-0.6B** BF16, batch 1, positions 0–200, one hand-written persistent megakernel; no exactness check stated | ~1,000 (~1,033 after an update) |

Neither is a measurement of these models on this GPU. llama.cpp is measured below; vLLM and SGLang
are not. Sources:
https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html,
https://blog.alpindale.net/posts/5090_decode_optimization/

### Benchmark against llama.cpp (2026-11-24, reported only, no bar)

§ Rows marked § come from llama.cpp b12359 (v1.0.0, the official Windows CUDA 13.5 build; release
checksums verified), with GGUFs converted locally from the same Hugging Face snapshots
(`convert_hf_to_gguf.py --outtype bf16`) and quantized with `llama-quantize`. `llama-server`
(`-ngl 99 -fa on -c 2048 -np 1`) on the self-spec prompts (8 per domain, code frozen at a118bda),
prompts sent as the same token ids, 256 tokens, greedy, one warm-up and 2 timed requests per prompt,
median; speed is the server's own decode timing (prompt excluded). Speculative rows use llama.cpp's
`--spec-type draft-simple` with the Q4_0 GGUF of the same model as draft, k ∈ {3, 4, 6, 8}; k = 6 is
the best calibration setting on both models (all settings had 0 violations). Every token is checked
against this repo's original BF16 decoder as for the levers (`scripts/llamacpp_bench.py`,
`results/processed/decode_levers/*_llamacpp.json`). `llama-bench` (`-p 0 -n 128 -d 512 -fa on`,
5 repetitions; `results/processed/decode_levers/llamacpp_bench/`) gives the same picture at 128
tokens: Qwen3-4B BF16 168.4 tokens/s, Q4_0 391.9, Q4_K_M 354.7; TinyLlama BF16 544.2, Q4_0 1,015.9,
Q4_K_M 968.5.

| tokens/s | Qwen3-4B | TinyLlama-1.1B |
|---|---|---|
| llama.cpp BF16 | 162 | 540 |
| This repo, L1 + L2 (same prompts, 128 tokens) | 156–160 | 525–526 |
| llama.cpp BF16 + Q4_0 draft, k = 6 (holdout / code); acceptance 75% / 88% and 91% / 89% | 156 / 185 | 445 / 435 |
| This repo, best INT4 self-draft (holdout / code); acceptance 87% / 94% and 88–95% | 244 / 258 (L7b) | 698 / 709 (L7) |
| llama.cpp Q4_0 model (lossy) | 365 | 905 |
| This repo, INT4 RTN g128 model (lossy) | 379 | 999 |

What this shows:
1. **Exact levers reach parity, not more.** L1 + L2 is 1–4% slower than llama.cpp's BF16 path on
   both models. The levers close the gap between this repo's original decoder and a tuned engine.
2. **The INT4 self-draft is the one clear gain.** With drafts of the same quality (similar
   acceptance), llama.cpp's speculative mode gives little or nothing over its own plain decoding
   (Qwen3-4B −4% / +15%, TinyLlama −18% / −20%), while this repo's graph-captured draft-and-verify
   round gives 1.5–1.6× (Qwen3-4B) and 1.3× (TinyLlama) over llama.cpp's plain BF16, with the same
   greedy output. The difference is the cost of a round, not the draft: here a whole round
   (k draft steps, the verify pass and the accept logic) is one CUDA graph with no host sync.
3. **llama.cpp's 4-bit models are more accurate than this repo's.** Its Q4_0 and Q4_K_M make 2–5×
   fewer greedy violations than RTN g128 at similar speed; as lossy models both still change the
   output, so neither is counted as a lever.

Caveats: one GPU, batch 1, greedy only, base models, Windows; llama.cpp's speculative settings other
than k were left at their defaults, so a tuned configuration may do better.

## Follow-up E: a larger model, Gemma 4 12B (pre-registered 2026-11-24, before any Gemma run)

Written before the Gemma decoder exists and before any Gemma measurement. The user asked for the
decoder to be benchmarked on larger models and chose Gemma 4 12B first (download approved);
Qwen3.8-27B follows once the decoder implements Gated DeltaNet. Gemma 4 26B A4B is not downloaded.
Inkling 2 has no "Mini" release; Inkling 2 Small (594B total, mixture of experts) does not fit on one
RTX 5090.

**Model.** `google/gemma-4-12B` (base, Apache-2.0), revision 1a2c7729392586f5ecddae8807ee2ef741eabb9a,
BF16. 48 layers: 40 sliding-window (1,024) layers with 8 KV heads of 256, 8 full-attention layers with
one shared K = V head of 512 and RoPE on a quarter of each head; q/k/v norms, sandwich norms, a
per-layer output scalar, attention scale 1, final-logit softcap 30, tied 262,144-token embedding.

**Decoder.** `src/ple/gemma4_decode.py` reimplements the text model for the same harness (prefill,
CUDA-graph step, INT4 copies, self-spec). The runs use at most 512 + 256 + 64 positions, inside the
1,024 sliding window, so sliding layers see the whole context; the decoder asserts this.

**Correctness gate (before any timing).** On the 12 decode prompts, the original Gemma decoder's
teacher-forced logits over prompt + 128 Transformers-greedy tokens against Transformers' own forward:
mean KL ≤ 0.005 and top-1 agreement ≥ 99%; and its 128-token greedy output has 0 violations under
the usual check. If the gate fails, nothing below runs until it passes; the gate does not change.

**Levers.** L1 (fused q/k norm + RoPE), L2 (fast attention; head-512 layers use batched matmuls, as
cuDNN does not take that head size), L1 + L2 and L1 + L2d, measured exactly as D2 (4 prompts per
domain, 128 tokens, 5 interleaved pairs). Same bars: ≥ 5% faster, pair spread < 2%, exact output.

**Self-spec.** The INT4 RTN g128 draft (L7) at k ∈ {3, 4, 6}, every 1 and 2 rounds, if the BF16
weights, the INT4 copy and the caches fit beside ComfyUI's ~3.3 GB; otherwise reported as not run
for memory. Same pick rule and bars as L7.

**References.** Transformers `generate` (dynamic and static cache) and llama.cpp b12359 (BF16, Q4_0,
Q4_K_M, and BF16 with a Q4_0 draft if it fits), as in the Qwen3-4B and TinyLlama comparisons.
Reported only.

### Follow-up E results so far (2026-11-24): the correctness gate fails

`scripts/decoder_gate.py`, `results/processed/decode_levers/gemma-4-12B_decoder_gate*.json`.

| Run | Mean KL (≤ 0.005) | Top-1 (≥ 99%) | Violations of our greedy output (0) | Verdict |
|---|---|---|---|---|
| 1: q/k/v and gate/up as one concatenated GEMM each (as the Qwen3 decoder) | 0.0160 | 98.96% | 20 | fail |
| 2: separate projections, as Transformers runs them | 0.0039 | 99.28% | 8 | **fail** (violations) |
| Control: Transformers' own cached `generate`, judged by Transformers' uncached forward | – | – | 31 | (reported only) |

**Diagnosis of run 1.** Embeddings match exactly and layers 0–21 differ only at rounding level
(relative error ~5 × 10⁻³). At layers 22–23 one prompt position (393 on code prompt 1) starts to
diverge and the error grows to more than 100% of the hidden state by the late layers; at one scored
position our top token was a different token entirely (24.1 against 20.0). Transformers' SDPA and
eager attention agree with each other there (KL 0.002). Rerunning our reference math with
Transformers' GEMM shapes (separate q/k/v, gate/up, `F.linear`) drops that prompt's KL from 0.153 to
0.001; attending over exactly T positions instead of the padded cache changes nothing. So the cause
is not a logic error but this model's sensitivity: a different GEMM kernel's rounding at some
positions is amplified into different logits. Even with matched GEMMs, single positions still drift
by up to 39% in hidden state at layer 30. The decoder was changed to separate projections (run 2);
the gate itself did not change.

**Run 2.** KL and top-1 pass. The greedy check does not: 8 tokens of our 128-token CUDA-graph output
on 12 prompts are not Transformers' argmax where Transformers' top-2 gap is above 0.25. The control
shows the same check fails for Transformers against itself, with 31 violations: its cached
`generate` (one token per step, KV cache) and its own uncached forward disagree more than our decoder
and its forward do. On this model the "0 violations" criterion cannot be met by any BF16
implementation measured here, Transformers included.

Per the rule above, nothing below the gate runs (levers, self-spec, Transformers and llama.cpp
references on Gemma) until the user decides how to proceed; the gate is not relaxed post hoc.

### Amendment 2026-11-24 (after gate run 2, decided by the user, before any run below)

The user chose to amend the gate's third criterion rather than stop or run unvalidated. New
criterion: the violations of our greedy output under Transformers' forward must not exceed
Transformers' own self-inconsistency on the same prompts (its cached `generate` judged by its
uncached forward, measured by `scripts/hf_self_consistency.py`: 31). KL ≤ 0.005 and top-1 ≥ 99% are
unchanged. Under the amended gate, run 2 passes (KL 0.0039, top-1 99.28%, 8 ≤ 31). Everything else in
follow-up E (levers, bars, self-spec, references) is unchanged and runs as written.

### Deviations found while running follow-up E (2026-11-24, before any Gemma timing)

- **L2 kernel.** The pre-registration expected only the head-512 layers to need batched matmuls.
  cuDNN SDPA on this build takes heads of at most 128, so it rejects Gemma's head-256 layers too
  (the first chain crashed there). Measured inside CUDA graphs, a one-row step per layer costs
  15.7 µs (head 256) / 15.9 µs (head 512) with batched matmuls, against 50–61 / 67–108 µs for the
  math and memory-efficient SDPA kernels and nothing from cuDNN. So on Gemma every one-row "fast"
  step uses batched matmuls, and L2 and L2d are the same code; both are still reported.
- **INT4 copy.** The BF16 weights (23.9 GB), the INT4 copy (~5.6 GB) and ComfyUI's ~3.3 GB exceed
  the 32.6 GB card (out of memory while packing the LM head), so the INT4 variant and the INT4
  self-draft are not run for memory, as pre-registered.
- **llama.cpp speculative.** BF16 target + Q4_0 draft also oversubscribes the card; Windows then
  pages GPU memory to system RAM, and the k = 3 draft run measured 29.9 ms/token against 16.8 for
  plain BF16 in the same attempt. Those timings measure paging, not speculation, so the run was
  stopped and is kept only as a record (`gemma-4-12B_llamacpp_attempt1_spilled.json`); llama.cpp is
  reported plain (BF16, Q4_0, Q4_K_M).

### Follow-up E results (2026-11-24): Gemma 4 12B

`gemma-4-12B_levers_decode_selfspec.json`, `gemma-4-12B_hf_baseline.json`, `gemma-4-12B_llamacpp.json`,
`llamacpp_bench/gemma-4-12B.bench.json`. RTX 5090, batch 1, BF16, 512-token prompt, decode only.

| Setting | ms/token | tokens/s | Speedup over original (spread) | Violations, 12 prompts × 128 tokens |
|---|---|---|---|---|
| Transformers `generate`, dynamic cache ‡ | 63.76 | 16 | – | 33 |
| Transformers `generate`, static cache, no compile ‡ | 64.10 | 16 | – | 42 |
| Original decoder | 23.60 | 42 | 1 | **9** (its own greedy output under its own full forward) |
| L1 fused q/k norm + RoPE | 22.02 | 45 | 1.071× (0.2%) | 13 |
| L2 fast attention (batched matmuls) | 21.22 | 47 | 1.112× (0.2%) | 35 |
| L1 + L2 (= L1 + L2d here) | 19.66 | 51 | 1.200× (0.2%) | 25 |
| llama.cpp BF16 § | 16.90 | 59 | – | 119 |
| llama.cpp Q4_0 § (lossy) | 6.70 | 149 | – | 300 |
| llama.cpp Q4_K_M § (lossy) | 7.20 | 139 | – | 230 |
| BF16 weight-streaming floor (23.9 GB ÷ 1.79 TB/s) | ~13.4 | ~75 | – | bound |

llama-bench (128 tokens after 512): BF16 60.3, Q4_0 156.5, Q4_K_M 142.5 tokens/s. Violations are
tokens that are not this repo's original decoder's argmax (top-2 gap above 0.25) under its
teacher-forced full pass; for llama.cpp, 24 prompts × 256 tokens (BOS not added: prompt lengths match).

**Verdict: the speed is real, exactness is not established, so by the bar no lever counts.** The
speedups clear 5% with spreads of 0.2%. But the exactness check fails for the original decoder
itself: 9 of its own greedy tokens disagree with its own full forward (one-row CUDA-graph steps
against one 640-row pass), as Transformers' generate disagrees with Transformers' forward (31 in the
gate). Every lever adds to that (13, 35, 25), so even a control-relative reading (the gate's
amendment, which the user approved for the gate only) would not pass them; the lever bar is not
amended. On this model, any change of kernel or rounding order changes some greedy tokens, and the
harness cannot separate "exact up to rounding" from "different". This is the BF16-fragility result
again, now on the speed side.

**Against llama.cpp**, Gemma 4 12B reverses the Qwen3-4B picture: llama.cpp's BF16 path is 17% faster
than L1 + L2 (59 against 51 tokens/s, 80% against 68% of the weight-streaming floor) and its output
departs further from the original decoder's (119 violations against 25). The INT4 self-draft could not
be measured: its INT4 copy does not fit beside the BF16 weights on this card.

## Follow-up F: cheaper drafts and fused kernels (pre-registered 2026-11-24, before any of these runs)

Written before any code or run below. The user decided to treat the results so far as the base for
empirical papers and to keep looking for speed. On Qwen3-4B the L7b round at k = 6 is six INT4 draft
steps (2.60 ms each, 37% of it outside GEMMs) and one BF16 verify pass (~6.4 ms), so the draft is most
of the round. Each method below makes the draft (or the step) cheaper; verification stays the full
BF16 model, so output stays exact.

- **F1 layer-skipped INT4 draft.** The L7b draft (GPTQ + clipping, groups of 32) also skips the
  attention / MLP sub-layers ranked cheapest by the Phase II atlas (skip-mode KL per MAC; block 0
  and the last block never skipped; `atlas_skip_sets`), at 5%, 10% and 15% of decoder MACs.
- **F2 reduced-vocabulary draft head.** The draft's INT4 LM head keeps only the V most frequent
  tokens of WikiText-2 train chunks 0–127 (the GPTQ calibration chunks; no code text), V ∈ {8192,
  32768}; its argmax is mapped back to the full vocabulary. Verify keeps the full head.
- **F3 compiled step.** With `triton-windows` 3.9.0.post30 (installed after follow-up E's runs, at the
  user's approval), `torch.compile` (Inductor, no CUDA graphs of its own; the result is captured in
  this repo's CUDA graphs) of the decoder forward, for the single-token step and for the self-spec
  round. If Inductor cannot compile it on this build, hand-written Triton kernels for fused residual
  add + RMSNorm and fused q/k norm + RoPE + KV write replace it.

**Measurement.** Self-spec as L7b: k ∈ {3, 4, 6, 8}, acceptance every round, 8 prompts per domain,
256 tokens, 2 interleaved pairs, the same exactness check; pick on calibration speed among
zero-violation settings, judged on holdout and code. A draft method **improves** if its pick beats
the L7b pick (1.572× holdout, 1.662× code over the L1 + L2 decoder, re-measured in the same run) by ≥ 3%
on both. F3 on the single-token step is an exact lever with the usual bar: ≥ 5% faster than L1 + L2,
pair spread < 2%, exact output. Methods that count are stacked and the stack reported (not judged).
Qwen3-4B first; replication on TinyLlama-1.1B (and Gemma 4 12B where the INT4 copy fits in memory)
under the same rules.

## Follow-up G: Qwen3.8-27B, INT4 (pre-registered 2026-11-24, before any Qwen3.8 run)

Written before the decoder has run. The user lifted the hold on Qwen3.8 and asked for it after Gemma.

**Model.** `RedHatAI/Qwen3.8-27B-INT4` (compressed-tensors W4A16: symmetric INT4, groups of 128;
LM head, embeddings and the DeltaNet a / b projections in BF16), revision 71dface3, on disk. BF16
weights would be ~54 GB, so the INT4 checkpoint *is* the model here: "exact" means equal to this
checkpoint's greedy output, not to a BF16 model. 64 layers: 48 Gated DeltaNet (linear attention,
fp32 recurrent state) and 16 gated full-attention layers (head 256, GQA 24 / 4, RoPE on 64 of 256
dims).

**Decoder.** `src/ple/qwen35_decode.py`, same harness interface; the checkpoint's INT4 codes are
repacked for torch's tinygemm kernel so each weight dequantises to exactly `q × scale`. Prefill uses
Transformers' own chunked delta rule and causal conv; one-token steps run the recurrent update in
place inside the CUDA graph. Multi-row verify steps (self-speculation) are not implemented yet.

**Gate.** As follow-up E after its amendment (KL ≤ 0.005, top-1 ≥ 99%, our greedy-output violations
≤ Transformers' own generate-vs-forward violations), against Transformers running the same
checkpoint through `ple.quant.packed` (dequantise to BF16 per forward). The two do not fit in memory
together, so `scripts/decoder_gate_split.py` runs it in three passes, saving Transformers' logits to
disk. If the gate fails, nothing below runs until it passes; the gate does not change.

**Levers.** L1 (fused zero-centred q/k norm + partial RoPE, 16 attention layers), L2 (batched-matmul
one-row attention; cuDNN SDPA here takes heads ≤ 128), L1 + L2, measured as D2 (4 prompts per domain,
128 tokens, 5 interleaved pairs), same bars (≥ 5%, spread < 2%, exact output). With only 16 of 64
layers doing attention, L1 and L2 are expected to be small; DeltaNet-step levers and MTP
self-speculation get their own pre-registration once this decoder passes.

**References.** llama.cpp b12359 on GGUFs converted from the same checkpoint (if its converter reads
compressed-tensors INT4) and quantized to Q4_0 / Q4_K_M; these are different 4-bit models, so their
speed is compared with this decoder's and their output checked against this checkpoint's, reported
only. Transformers through the packed loader is the gate's reference only: it dequantises every
weight on every forward, so its speed is not reported as a baseline.

### Follow-up G results so far (2026-11-24): the Qwen3.8 decoder passes the gate

`scripts/decoder_gate_split.py`, `results/processed/decode_levers/Qwen3.8-27B-INT4_decoder_gate.json`.

| Mean KL (≤ 0.005) | Top-1 (≥ 99%) | Our greedy-output violations | Control: Transformers vs itself | Verdict |
|---|---|---|---|---|
| 0.0011 | 99.41% | 1 | 1 | **pass** (first run, no changes) |

Per prompt the KL is 0.0004–0.0009 except hold 0 (0.0066, one position at 0.56); our CUDA-graph
greedy output equals Transformers' for all 128 tokens on 5 of 12 prompts, and its disagreements with
Transformers' forward are near-ties except one. Unlike Gemma 4 12B, this model is not fragile in this
sense: Transformers' own generate disagrees with its forward once in 1,536 tokens (Gemma: 31). The
gate as first written (0 violations) would have failed by that one token, as Transformers itself does.

### Follow-up G results (2026-11-24): Qwen3.8-27B levers and llama.cpp

`Qwen3.8-27B-INT4_levers_decode.json`, `Qwen3.8-27B-INT4_llamacpp.json`, `llamacpp_bench/Qwen3.8-27B.bench.json`.
RTX 5090, batch 1, 512-token prompt, decode only; this repo's decoder runs the RedHat INT4 checkpoint.

| Setting | ms/token | tokens/s | Speedup (spread) | Violations | Verdict |
|---|---|---|---|---|---|
| Original decoder | 18.41 | 54 | 1 | 0 / 1,536 | reference |
| L1 fused q/k norm + partial RoPE | 18.02 | 55 | 1.021× (0.1%) | 0 | below 5%: does not count |
| L2 fast attention (batched matmuls) | 17.73 | 56 | 1.038× (0.1%) | 0 | below 5%: does not count |
| L1 + L2 (= L1 + L2d) | 17.34 | 58 | 1.062× (0.2%) | 0 | stack, not judged |
| llama.cpp Q4_0 § (its own 4-bit model) | 12.3 | 81 | – | 87 / 6,144 | reference |
| llama.cpp Q4_K_M § | 13.7 | 73 | – | 65 / 6,144 | reference |

llama-bench (128 after 512): Q4_0 82.6, Q4_K_M 74.0 tokens/s. llama.cpp's violations are against
this repo's original decoder, i.e. against the RedHat INT4 checkpoint: its models are different
4-bit quantisations of the same weights (converted to BF16 and re-quantised), so they change the
output, which they would do against BF16 too.

**Verdict.** Output is exact, but with attention in only 16 of 64 layers neither lever clears 5%.
llama.cpp is 1.26–1.40× faster. The step reads ~15.5 GB (13 GB of INT4 layers and the 2.5 GB BF16 LM
head; weight-streaming floor ~8.7 ms), so this decoder is at ~50% of the floor, against llama.cpp's
~70%; candidate causes, not measured yet: the BF16 LM head (llama.cpp quantises it), the DeltaNet
step as ~20 small kernels per layer (a 150 MB fp32 state read and written every token), and the
tinygemm INT4 kernel at one row.

## Follow-up H: self-speculation with Qwen3.8's own MTP layer (pre-registered 2026-11-24, before any code)

Written before any of it exists. Qwen3.8 cannot use an INT4 copy of itself as draft (it is already
INT4), but its checkpoint ships a multi-token-prediction (MTP) module trained with it
(`model_mtp.safetensors`, BF16, 0.85 GB): `fc([norm_e(embed(t_(i+1))), norm_h(h_i)])` → one gated
attention layer + MLP → `mtp.norm` → the shared LM head, predicting t_(i+2) from the main model's
pre-norm hidden state h_i (layout confirmed against llama.cpp's `qwen35` MTP graph). A draft step costs
one layer and the LM head instead of 64 layers.

**Method (H1).** Each round, one CUDA graph: (1) the MTP layer runs over the previous verify pass's
rows with the main model's true hidden states (this refreshes its KV cache and yields draft d_1 from
the last accepted row); (2) d_2 … d_k chain the MTP layer on its own output hidden state; (3) the full
INT4 model verifies [last, d_1 … d_k] in one pass; the longest agreeing prefix plus the model's next
token is kept. The Gated DeltaNet layers run the verify rows one by one, keep the recurrent state after
each row and roll back to the last kept row; the conv state is re-cut from the verify inputs. Output
is the model's own greedy output (same exactness check as L7).

**Measurement.** k ∈ {1, 2, 3, 4}, acceptance checked every round, 8 prompts per domain, 256 tokens,
2 interleaved pairs against the exact one-token decoder (the best exact setting of follow-up G),
pick on calibration among zero-violation settings, judged on holdout and code. **Passes** at
≥ 1.3× holdout and ≥ 1.1× code over that decoder (the self-spec bars), exact output.

**Reference.** llama.cpp's own MTP speculation (`--spec-type draft-mtp`) on its Q4_K_M GGUF (which
includes the MTP layer), k ∈ {1, 2, 3, 4}, same prompts, against its own plain Q4_K_M decoding. Its
model is a different 4-bit quantisation, so its speedup over itself is compared with ours over our
exact decoder, and tokens/s side by side. Reported only.

### Amendment to follow-up H, 2026-11-25 (written while the first MTP run was in progress, before any MTP result was seen)

Follow-up J (amended) replaced the 8-prompt, three-domain protocol with stable tokens/s at 2k–128k on one
holdout prompt, and F3a made the fused one-token step the best exact setting. H is measured the same way:

- **Reference:** the fused exact step (F3a, `l12c`) on the same prompt and context, and llama.cpp's plain
  decoding of the matched Red Hat INT4 GGUF (Q4_0, same codes; the follow-up J runs).
- **MTP rounds use the fused helpers too**, including a fused multi-row DeltaNet verify pass
  (`_f_delta_verify`: the same recurrent rule unrolled over the k + 1 rows, states kept per row for
  `commit`), so the comparison is fused against fused.
- **Measured:** k ∈ {1, 2, 3}; a greedy check at 2k (256 tokens, against the same model's teacher-forced
  pass, unfused); then stable tokens/s at every context where it fits, 2 timed runs.
- **Counts** at a context if it is ≥ 1.1× the fused exact step with 0 violations (the self-spec bar for
  code, the lower of the two original bars, since there is now one domain). The bar is not relaxed later.
- **llama.cpp's MTP** (`--spec-type draft-mtp`, llama.cpp b12720) on the matched Q4_0 GGUF, which carries the
  MTP layer (`blk.64.nextn.*`, BF16 from the checkpoint), k ∈ {2, 3, 4}, with b12720's plain decoding of the
  same file as its own reference, same prompt and contexts, 2 runs. Reported only.

*Deviation found 2026-11-25, after the first 2k run and llama.cpp's 2k MTP run were seen:* our MTP layer was fed
the main model's hidden state **before** its final norm, and chained on its own pre-norm output. The pre-registration
above says "pre-norm", but llama.cpp's `qwen35` MTP graph (`h_nextn` = the output-normed hidden, and the MTP's
`shared_head_norm` output for chaining) and vLLM both use the **post-norm** states. Exactness does not depend on
this (the full model verifies every token); acceptance does. First run (pre-norm), 2k: k = 2 / 3 / 4 at 125.1 /
139.8 / 140.7 tokens/s, acceptance 0.89 / 0.86 / 0.79, violations 1 / 0 / 0 (k = 1: 98.7, 0.88, 0); kept as
`Qwen3.8-27B-INT4_tps_mtp_prenorm.json` and `..._prenorm_smoke.json`. Same-day llama.cpp b12720 MTP on the same
file, 2k: 134.8 / 147.8 / 157.5 tokens/s, acceptance 0.94 / 0.91 / 0.86. The code now uses the post-norm states
(`ple.mtp_spec`) and every context is rerun with it; both runs are reported.

*Note added 2026-11-25 after the 2k smoke run (k = 1–3 seen):* k = 4, which was in the original H list, is
added for the all-context run (with k = 2 and 3). k = 1 is not rerun past 2k. Nothing else changes.

### Follow-up H results (2026-11-25): Qwen3.8's MTP layer as the draft (post-norm states)

Check at 2k (256 tokens against the same INT4 model's teacher-forced pass): **0 violations for k = 2, 3 and
4** (1–2 near-ties each). Stable tokens/s, one holdout prompt per context, 256 tokens, median of 2 runs; the
draft acceptance in brackets. llama.cpp b12720 runs its own MTP on the same GGUF (same INT4 codes, same BF16
MTP layer and LM head), same prompts; its arithmetic is W4A8 and it was not checked for exactness.

| Context | Ours fused step | Ours MTP k = 2 | k = 3 | k = 4 | llama.cpp plain | llama.cpp MTP k = 2 | k = 3 | k = 4 |
|---|---|---|---|---|---|---|---|---|
| 2k | 73.1 | 126.6 (0.92) | 141.0 (0.88) | **150.2** (0.87) | 75.6 | 134.8 (0.94) | 147.8 (0.91) | **157.5** (0.86) |
| 4k | 71.6 | 111.8 (0.78) | **119.9** (0.72) | 117.8 (0.63) | 74.4 | 120.3 (0.76) | **126.7** (0.70) | 122.9 (0.62) |
| 8k | 68.8 | 92.7 (0.59) | 94.9 (0.52) | **101.9** (0.53) | 72.9 | 101.0 (0.59) | 101.4 (0.51) | **117.3** (0.59) |
| 16k | 64.7 | 107.3 (0.82) | **113.2** (0.73) | 113.2 (0.66) | 69.5 | 116.2 (0.82) | 124.6 (0.75) | **126.7** (0.70) |
| 32k | 57.8 | 85.0 (0.65) | **88.0** (0.57) | 83.0 (0.48) | 60.8 | **89.9** (0.62) | 86.9 (0.50) | 83.6 (0.42) |
| 64k | 47.3 | 75.1 (0.69) | **78.8** (0.61) | 78.8 (0.56) | 48.4 | 83.0 (0.73) | **87.2** (0.64) | 79.2 (0.49) |
| 128k | 34.7 | 56.2 (0.65) | **58.1** (0.57) | 50.5 (0.42) | 34.5 | 57.3 (0.61) | **59.9** (0.53) | 55.8 (0.40) |

- **H counts at every context** (≥ 1.1× the fused exact step, 0 violations): every k at every
  context is 1.35–2.05× faster; the best k per context 1.48–2.05×. Acceptance swings with the prompt (one prompt per context), the same way in
  both engines.
- **llama.cpp's MTP is faster than ours by 2–13%** (best k against best k: 2k 0.95×, 4k 0.95×, 8k 0.87×,
  16k 0.89×, 32k 0.98×, 64k 0.90×). Acceptance is level, so the difference is the cost of a round.
- Against llama.cpp's *plain* decoding (the comparison of follow-up J) our MTP is 1.40–1.99× faster.
- At 128k the first attempt ran out of memory in the MTP prefill (its per-chunk masks); rerun with 512-row chunks
  (`..._tps_mtp_128k.json`; 0.97× llama.cpp's MTP). llama.cpp's 128k MTP was first skipped by our memory guard,
  which counted the MTP as a second full KV cache; with the estimate fixed (one extra layer) it ran
  (`llamacpp_b12720/mtp128k/`).

### Follow-up H2: a 4-bit draft head for MTP (pre-registered 2026-11-25, before any H2 run)

Written after the post-norm MTP run at 4k (acceptance level with llama.cpp's, speed ~5% below it) and before
any H2 code ran. Each MTP draft step reads Qwen3.8's whole BF16 LM head (248,320 × 5,120, 2.5 GB, ~1.4 ms),
k times per round. **H2:** the draft's logits come from an INT4 copy of that head (groups of 32, clipping
search; `MTPLayer(head="int4")`, `tps_bench.py --mtp-head int4`); the verify pass keeps the BF16 head, so
the output stays the model's own (checked as in H). Measured exactly as H (amended): k ∈ {3, 4}, greedy
check at 2k (256 tokens), stable tokens/s at 2k–128k, 2 runs. **Counts** at a context if it is ≥ 5% faster
than H's post-norm MTP at the same k with 0 violations. Not relaxed later.

### Follow-up H2 results (2026-11-25): the 4-bit draft head

Check at 2k (256 tokens): **0 violations for k = 3 and 4**. Same prompts and protocol as H; tokens/s (acceptance).

| Context | Fused step | H2 k = 3 | H2 k = 4 | H2 / H (same k, better of the two) | llama.cpp MTP, best k | Ours (H2 best) / llama.cpp MTP |
|---|---|---|---|---|---|---|
| 2k | 72.7 | 158.7 (0.86) | **175.6** (0.87) | 1.17× (k = 4) | 157.5 | **1.11×** |
| 4k | 71.2 | **136.8** (0.72) | 135.6 (0.62) | 1.15× (k = 4) | 126.7 | **1.08×** |
| 8k | 68.8 | 107.8 (0.52) | **110.0** (0.47) | 1.14× (k = 3) | 117.3 | 0.94× |
| 16k | 64.8 | 127.3 (0.72) | **132.9** (0.67) | 1.17× (k = 4) | 126.7 | **1.05×** |
| 32k | 57.7 | 95.3 (0.54) | **96.8** (0.49) | 1.17× (k = 4) | 89.9 | **1.08×** |
| 64k | 47.2 | **85.8** (0.60) | 82.6 (0.51) | 1.09× (k = 3) | 87.2 | 0.98× |
| 128k | 34.7 | **62.1** (0.56) | 55.8 (0.43) | 1.10× (k = 4) | 59.9 | **1.04×** |

**H2 counts at every context measured** (≥ 5% over H at the same k, 0 violations): the draft's acceptance is
unchanged (the 4-bit head almost never changes the draft's top token), and each draft step reads 0.6 GB
instead of 2.5 GB. With it our MTP self-speculation is faster than llama.cpp's own MTP on the same weights at
2k, 4k, 16k, 32k and 128k, and 2–6% slower at 8k and 64k; against our fused one-token step it is 1.6–2.4×.
The 128k H2 run (512-row prefill chunks) paged ~0.2 GB of GPU memory to system RAM (Windows reported it
as shared usage of the process), so its numbers may read low.

## Follow-up I: long contexts (pre-registered 2026-11-24, before any code or run)

Written before any of it exists. The user pointed out that the real uses (chatbots, summarization,
production serving) run at long contexts, where every result so far (512-token prompts) may not hold:
attention and the KV cache grow with the context, the weights do not.

**Contexts.** Prompts of 4,096, 8,192, 16,384 and 32,256 tokens (32,256 + 256 generated + slack stays
inside Qwen3-4B's 32,768 positions); 2 prompts per domain per context: calibration and holdout text
from WikiText-2 train at token offsets 204,800 and 1,228,800 (the regions of the short-context
chunks 400 and 600), code from this repo at commit a118bda.

**Models.** Qwen3-4B (all contexts), Gemma 4 12B (all contexts; its decoder gets real sliding-window
attention for the 40 window-1,024 layers, checked against Transformers at 4k before timing),
Qwen3.8-27B INT4 (all contexts that fit). TinyLlama-1.1B (2,048 positions) is excluded.

**Harness change.** Prefill runs in chunks of 2,048 rows (an explicit causal mask per chunk) so a
32k prompt never builds a 32k × 32k mask; prefill stays untimed. Each mode prefills once per prompt
and snapshots its cache and recurrent state; the timed runs restore the snapshot instead of
re-prefilling. The greedy check scores the generated tokens as one teacher-forced extension of the
prompt's cache (not one 32k pass), with the usual rule.

**Measured.** For each model and context: decode ms/token and tokens/s (128 tokens) for the original
decoder, L1, L2, L1 + L2 and L1 + L2d, 3 interleaved pairs (fewer than 5, for time); a profile of
the L1 + L2 step (attention and KV-cache share); on Qwen3-4B also the L7b INT4 self-draft pick
(GPTQ + clip g32, k = 6), 256 tokens; and llama.cpp b12359 at the same contexts with its default F16
KV cache and with `q8_0` KV (`llama-bench -d`), plus its server on the same prompts for BF16 / Q4_K_M.
**Bars** unchanged: a lever counts at a context if ≥ 5% faster with spread < 2% and exact output
(on Gemma, as at short context, exactness is reported against the original decoder's own
self-inconsistency at that context, and the bar is not relaxed).

**Amendment 2026-11-24 (before any run of follow-up I).** This repo's code at a118bda is only
35k–42k tokens (by tokenizer), too short for two 32k code prompts. Long-context code prompts instead
come from the standard library of the interpreter that runs the benchmarks (CPython 3.11.14,
`sys.base_prefix/Lib`, every `.py` file sorted by path, excluding `test*`, `idlelib` and
`site-packages`), cut into consecutive non-overlapping prompts from its start. Nothing else changes.

**Amendment 2026-11-24, at the user's direction (before any timed run of follow-up I).** Two timed
pairs per lever (after one warm-up pair) instead of three, to shorten the runs. The spread bar
(< 2%) is computed over those two pairs. Nothing else changes.

**Next (not pre-registered yet).** Quantized-KV methods (a quantized-KV draft, TurboQuant-style
rotation + scalar quantisation) get their own pre-registration after these runs show how much of a
long-context step the KV cache takes.

### Follow-up I withdrawn (2026-11-24, at the user's direction)

The user withdrew the full sweep (every lever, every model, four contexts) in favour of follow-up J:
only this repo's best decode settings against llama.cpp, with the quantisation matched between the two
engines. No follow-up I run was completed; the 4k smoke test of the harness (1 prompt per domain,
16 tokens) is not a result. The harness work (chunked prefill, Gemma's ring-buffer sliding
attention, snapshots) is kept and used by J.

## Follow-up J: best settings against llama.cpp, quantisation matched (pre-registered 2026-11-24, before any run)

Written before the matched files exist. Every comparison gives both engines the **same weights**:

| Model | This repo | llama.cpp b12359 | How the weights match |
|---|---|---|---|
| Qwen3-4B | L1 + L2, BF16 weights, BF16 KV | BF16 GGUF, `-ctk bf16 -ctv bf16`, flash attention | GGUF tensors checked bit-identical to the checkpoint |
| Qwen3-4B | L7b self-spec: BF16 target + INT4 draft (GPTQ + clipping, groups of 32, incl. LM head), k = 6 | BF16 target + a draft GGUF whose linear layers and LM head are **Q4_1 blocks holding exactly the codes, scales and minima of this repo's draft** (`--spec-type draft-simple`), k ∈ {4, 6}, best reported | same 4-bit codes; Q4_1 is asymmetric groups of 32 like the draft; the draft's codes are saved once and loaded by both |
| Gemma 4 12B | L1 + L2, BF16 | BF16 GGUF, BF16 KV | as Qwen3-4B (no draft: it does not fit beside the BF16 weights in either engine) |
| Qwen3.8-27B | L1 + L2 on the RedHat INT4 checkpoint | a Q4_0 GGUF **written from the checkpoint's own codes and scales** (each group of 128 as four Q4_0 blocks with the same scale; the converter's V-head reordering applied to codes and scales), LM head, embeddings and MTP layer BF16 as in the checkpoint | dequantised tensors checked equal to the checkpoint's `q × scale` |
| Qwen3.8-27B | follow-up H's MTP self-spec (when implemented and passing its own checks) | `--spec-type draft-mtp` on the same GGUF | same MTP weights (BF16) |

Residual differences, stated in advance: llama.cpp stores block scales in FP16 and this repo's kernel
in BF16 (both hold BF16-derived values exactly, except Q4_1 minima, which are rounded to FP16);
kernels and accumulation order differ.

**Contexts.** 512 (the short-context prompts) and 8,192 and 32,256 (follow-up I's long prompts;
Qwen3.8 and Gemma as memory allows). 2 prompts per domain; this repo: one warm-up and 2 timed runs,
median; llama.cpp server: one warm-up and 2 requests, median of its own decode timing.

**Reported.** Decode tokens/s of both engines side by side, and for both the greedy violations
against this repo's original decoder on the same weights (teacher-forced, chunked at long context).
No bar: this is a measurement of where each engine stands with identical weights.

### Follow-up J: the matched files (2026-11-24)

- **Qwen3-4B draft.** `scripts/save_draft_codes.py` quantised the L7b draft once (GPTQ + clipping,
  groups of 32, LM head included; 145 matrices) and saved the exact codes, BF16 scales and BF16
  zero points the kernel uses (`G:/ple_cache/draft_qwen3_4b_gptq_clip_g32.pt`). This repo's runs
  load them (`Qwen3Decoder.load_int4_codes`); `scripts/make_matched_gguf.py draft-q4_1` wrote the
  same codes as Q4_1 blocks into a copy of the BF16 GGUF (252 matrices + an untied Q4_1
  `output.weight`; token embeddings BF16), `Qwen3-4B-draft-ours-Q4_1.gguf`, 3.3 GB. llama.cpp loads
  it and runs it alone at 353 tokens/s.
- **Qwen3.8.** The BF16 GGUF's attention and MLP tensors equal the checkpoint's `q × scale` rounded
  to BF16; its DeltaNet `attn_qkv`, `attn_gate` and `ssm_out` are the same values with the V heads
  reordered (whole 128-wide heads, the same width as the INT4 groups, so groups stay whole).
  `make_matched_gguf.py redhat-q4_0` writes the checkpoint's own codes and scales as Q4_0 with that
  reordering and checks every tensor. 336 of 190,054,400 scales (all below 6 × 10⁻⁵, in FP16's
  subnormal range) are not exact in FP16; those groups are rounded to FP16 and excluded from the
  equality check, which holds for every other weight.
- **KV cache.** llama.cpp runs with `-ctk bf16 -ctv bf16` (and the same for its draft), as this
  repo's BF16 cache; checked to work with flash attention (Qwen3-4B 167 tokens/s).

### Amendment to follow-up J, 2026-11-24, at the user's direction (before any J measurement)

The user asked for contexts of 2k, 4k, 8k, 16k, 32k, 64k and 128k tokens and only the stable decode
throughput. So, replacing J's measurement protocol:

- **Contexts.** Prompts of 2,048, 4,096, 8,192, 16,384, 32,768, 65,536 and 131,072 tokens minus 320
  (room for 256 generated tokens and slack inside each model's limit); every model and setting at
  every context that fits in GPU memory; a setting that does not fit is reported as such.
- **Measure.** Decode tokens/s only: 256 generated tokens after the prompt, prefill excluded; this
  repo: one warm-up and 2 timed runs, median; llama.cpp server: one warm-up and 2 requests, median of
  its decode timing. One prompt per context (WikiText-2 holdout text, as follow-up I). No exactness
  check (the settings' exactness is established at 512 and 4k already).
- **Qwen3-4B beyond 32k.** Its native window is 32,768; 64k and 128k use YaRN RoPE scaling (factor 4,
  original 32,768, as Qwen's model card recommends) in both engines: this repo implements
  Transformers' YaRN formula; llama.cpp gets `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768`.
  Gemma 4 12B and Qwen3.8 support 262,144 positions natively.
- Weights and KV cache stay matched as pre-registered (BF16 / exact INT4 codes; BF16 KV).

### Follow-up J results (2026-11-25): stable decode tokens/s, same weights in both engines

`scripts/tps_bench.py` (this repo) and `scripts/llamacpp_bench.py --no-check` (llama.cpp b12359,
BF16 KV, flash attention), the same WikiText-2 holdout prompt per context, 256 tokens, median of 2
timed runs after a warm-up, prefill excluded. Files: `*_tps*.json`, `*_llamacpp_ctx*_kvbf16.json`.

**Qwen3-4B, BF16 (identical tensors); self-draft: identical INT4 codes (Q4_1 in llama.cpp), k = 6.**
Above 32k both engines use YaRN (factor 4).

| Context | Ours L1 + L2 | llama.cpp BF16 | Ours / llama.cpp | Ours INT4 self-draft | llama.cpp + same draft | Best ours / best llama.cpp |
|---|---|---|---|---|---|---|
| 2k | 157.4 | 156.8 | 1.00 | **251.0** | 161.2 | **1.56×** |
| 4k | 152.5 | 150.9 | 1.01 | **235.7** | 145.0 | **1.56×** |
| 8k | 144.5 | 142.4 | 1.01 | **193.0** | 152.7 | **1.26×** |
| 16k | 131.0 | 124.3 | 1.05 | **163.9** | 124.3 | **1.32×** |
| 32k | 109.8 | 99.7 | 1.10 | **122.5** | 77.5 | **1.23×** |
| 64k | **82.8** | 69.7 | 1.19 | 73.0 | 51.8 | **1.19×** |
| 128k | **55.7** | 42.3 | **1.32** | does not fit | does not fit | **1.32×** |

**Gemma 4 12B, BF16 (identical tensors).**

| Context | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|---|
| Ours L1 + L2 | 50.8 | 50.7 | 50.5 | 50.2 | 49.5 | 48.0 | 46.1 |
| llama.cpp BF16 | 58.9 | 59.1 | 58.4 | 57.7 | 56.5 | 53.0 | 47.0 |
| Ours / llama.cpp | 0.86 | 0.86 | 0.86 | 0.87 | 0.88 | 0.91 | 0.98 |

**Qwen3.8-27B, the RedHat INT4 codes in both (Q4_0 GGUF written from them).**

| Context | 2k | 4k | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|---|
| Ours L1 + L2 | 57.4 | 56.5 | 54.9 | 52.0 | 47.2 | 39.8 | 30.2 |
| llama.cpp | 74.7 | 74.0 | 72.5 | 69.5 | 60.9 | 48.6 | 34.5 |
| Ours / llama.cpp | 0.77 | 0.76 | 0.76 | 0.75 | 0.78 | 0.82 | 0.88 |

Self-draft acceptance (ours) was 0.90 at every Qwen3-4B context; llama.cpp's own count with the same
draft was 0.75–0.90. Our 128k Qwen3-4B run first failed: this decoder held the attention and MLP
weights twice (Transformers' copies and its concatenated ones, ~7.5 GB); it now points Transformers'
projections at slices of the concatenated copies (values unchanged), after which 128k fits; the INT4
draft (3.3 GB) does not fit beside the 19 GB 128k cache. llama.cpp's speculative mode keeps a second
full-length KV cache for its draft, so at 128k it would need ~50 GB and was not run.

**What this shows.**
1. **Qwen3-4B: this repo wins at every context with the same weights.** Plain decoding is level up to
   8k and pulls ahead as the context grows (1.10× at 32k, 1.32× at 128k): its one-token attention
   (cuDNN SDPA over the cache) scales better than llama.cpp's at long context. The INT4 self-draft adds
   1.5× at short context, but its six draft steps each attend the full BF16 cache, so its gain shrinks
   with context and at 64k it is slower than plain decoding. llama.cpp's speculation with the same
   draft weights gains little at any context and loses from 32k up.
2. **Gemma 4 12B and Qwen3.8: llama.cpp is faster**, by 16% and 30% at 2k, and the gap narrows with
   context (2% and 14% at 128k). Both decoders here run their attention as batched
   matmuls (cuDNN takes heads ≤ 128 only), and Qwen3.8's DeltaNet step is ~20 small kernels per layer,
   so the fixed per-token cost is higher than llama.cpp's; the attention cost per extra token is lower.
3. **Next lever this points at:** the draft's attention at long context. A draft that attends a
   quantised or windowed copy of the cache (verify still exact) would keep the 1.5× of short contexts at
   long ones; this is the KV-cache work discussed earlier, now with a measured target.

## Follow-up K: quantised targets with 4-bit and 2-bit drafts, against llama.cpp (pre-registered 2026-11-25, before any run)

Written before the decoders run any of it. The user pointed out that most people run quantised
models, so the comparison that matters is a quantised target with a cheaper draft, not BF16.

**Target: Q8_0 in both engines.** llama.cpp's Q8_0 (8-bit codes in blocks of 32, one FP16 scale per
block) for every matrix including the tied embedding / LM head, made by `llama-quantize` from the BF16
GGUF. This repo quantises the same BF16 weights with the same formula (`ple.kernels.q8_0_quantize`, on
the CPU: codes and scales checked byte-identical to llama-quantize's on four Qwen3-4B tensors) and runs
them with a new Triton kernel (`ple.kernels.w8a16`: 8-bit weights × BF16 activations, fp32
accumulation; one-row GEMV and a ≤ 16-row tile kernel for verify passes; prefill dequantises and uses
cuBLAS). "Exact" now means the Q8_0 model's own greedy output. llama.cpp's quantised kernels convert
activations to 8 bits for integer dot products, so its arithmetic is not identical (as on Qwen3.8).

**Drafts.** (a) Q4: Qwen3-4B's saved GPTQ + clipping g32 draft (follow-up J, Q4_1 in llama.cpp);
Gemma 4 12B: round-to-nearest + clipping g32 (no GPTQ statistics are collected for Gemma blocks),
written into a Q4_1 GGUF the same way. (b) Q2: 2-bit drafts, pre-registered separately once a 2-bit
kernel exists. Qwen3.8 stays on its INT4 checkpoint (no higher-precision weights here); its drafts
are (b) or its MTP layer (follow-up H).

**Check before timing (2k context, 2 prompts).** This repo's Q8_0 one-token decoder against its own
teacher-forced pass on the same Q8_0 weights: violations as the usual rule, to validate the kernel.

**Measured, as follow-up J amended.** Stable decode tokens/s at 2k, 4k, 8k, 16k, 32k, 64k, 128k
(Qwen3-4B with YaRN above 32k in both), one holdout prompt, 256 tokens, median of 2 runs: ours Q8_0
with L1 + L2, ours Q8_0 + Q4 self-draft (k = 6); llama.cpp Q8_0, and Q8_0 + the same Q4_1 draft
(k = 6). BF16 KV in both. Settings that do not fit are reported as such.

### Follow-up K results: Qwen3-4B, Q8_0 target (2026-11-25)

**Check (2k, 2 prompts × 128 tokens).** The Q8_0 one-token decoder and the Q8_0 + Q4 self-draft
against the same Q8_0 model's teacher-forced pass: 0 violations for both (2–7 near-ties per prompt,
the one-row Triton kernel against the cuBLAS prefill path).

**Stable decode tokens/s, both engines on byte-identical Q8_0 weights, same Q4_1 draft (k = 6).**

| Context | Ours Q8_0 (L1 + L2) | Ours Q8_0 + Q4 draft | llama.cpp Q8_0 | llama.cpp Q8_0 + same draft | Best / best |
|---|---|---|---|---|---|
| 2k | 252.2 | **255.9** | 239.6 | 164.2 | 1.07× |
| 4k | 240.5 | **262.9** | 226.2 | 180.6 | 1.16× |
| 8k | 220.8 | **235.2** | 205.1 | 148.8 | 1.15× |
| 16k | **191.2** | 177.1 | 173.5 | 118.9 | 1.10× |
| 32k | **149.9** | 127.3 | 128.5 | 79.7 | 1.17× |
| 64k | **104.0** | 76.6 | 81.8 | 54.0 | 1.27× |
| 128k | **64.6** | 45.7 | 46.5 | does not fit | 1.39× |

Against BF16 (follow-up J) the Q8_0 target is 1.6× faster in this repo at 2k (157 → 252) and 1.16× at
128k (56 → 65). The Q4 self-draft that gave 1.6× over a BF16 target gives at most 1.09× over a Q8_0
target (4k) and loses from 16k: a 4-bit draft step costs about two thirds of an 8-bit target step
(draft fixed costs, ~0.9 ms of small kernels, do not shrink with the weights), and its acceptance
falls (0.8–0.9; it was quantised from BF16, not from the Q8_0 target). llama.cpp's speculation with
the same draft is slower than its plain decoding at every context.

**Q2 draft feasibility (exploratory, llama.cpp only, 2k, 3 prompts).** A pure Q2_K Qwen3-4B
(`llama-quantize --pure`, BF16 embedding) drafting for the Q8_0 target is accepted 53% (k = 2) down to
27% (k = 6) of the time, and llama.cpp runs at 76–90 tokens/s against 240 without it. Speculation at
this cost ratio needs acceptance near 0.75 or more, so a 2-bit self-draft of a 4B model is not worth a
2-bit kernel here; the 2-bit kernel is not built for Qwen3-4B. Gemma 4 12B, measured the same way
(pure Q2_K, BF16 embedding, Q8_0 target, 2k, 3 prompts): acceptance 0.01 (k = 2) and 0.00 (k = 4, 6);
llama.cpp runs at 15–26 tokens/s against 98 without the draft. A pure 2-bit Gemma 4 12B does not
predict the 8-bit model's tokens at all, so no 2-bit draft is pursued for either model.

### Amendment to follow-up F3, 2026-11-25 (before any F3 run)

`triton-windows` 3.9.0.post30 is installed (as approved) and works on the RTX 5090 (sm_120), with
`torch.compile`. The profile above shows the remaining per-token overhead is thousands of small
kernels (Gemma 4 12B 3,039 per token, 3.4 ms; Qwen3.8 4,940, 5.9 ms; Qwen3-4B 655, 0.9 ms). F3 is
extended from Qwen3-4B to all three models, on the targets of the latest comparisons (Qwen3-4B BF16 and
Q8_0, Gemma 4 12B BF16 and Q8_0, Qwen3.8 INT4):

- **F3a.** `torch.compile` (Inductor, `dynamic=False`, no Inductor CUDA graphs) of the decoder's
  one-token forward, captured in this repo's CUDA graph as before. Measured: stable tokens/s at 2k
  and 32k against the uncompiled L1 + L2 step on the same weights, and the greedy check at 2k
  (2 prompts × 128 tokens against the same model's uncompiled teacher-forced pass). It **counts** at a
  context if ≥ 5% faster, with 0 violations (on Gemma, where even the uncompiled step has
  self-violations, against that count, and the bar is not relaxed).
- **F3b.** If F3a leaves Qwen3.8's DeltaNet step as most of its non-matmul time, a hand-written Triton
  kernel for the recurrent update (one read and one write of the fp32 state per layer), measured the
  same way.
- Then the faster step replaces L1 + L2 in the llama.cpp comparison (same contexts, same weights).

### Follow-up K results: Gemma 4 12B, Q8_0 target (2026-11-25)

Check (2k, 2 prompts × 128 tokens): Q8_0 one-token decoder 0 violations; Q8_0 + Q4 draft 1 (Gemma's
known fragility). Draft: round to nearest + clipping, groups of 32 (Q4_1 in llama.cpp, the same codes).

| Context | Ours Q8_0 | Ours Q8_0 + Q4 draft | llama.cpp Q8_0 | llama.cpp Q8_0 + same draft |
|---|---|---|---|---|
| 2k | 81.7 | 91.3 | **97.1** | 64.3 |
| 4k | 81.6 | 43.0 (acceptance 0.3) | **96.7** | 80.7 |
| 8k | 81.1 | 88.1 | **95.2** | 83.9 |
| 16k | 80.2 | 89.7 | **93.3** | 80.6 |
| 32k | 77.3 | 51.9 (acceptance 0.5) | **89.9** | 57.9 |
| 64k | did not fit (prefill) | – | **81.4** | 49.3 |
| 128k | did not fit (prefill) | – | **68.5** | 52.6 |

llama.cpp leads Gemma by 15–19% at 8 bits, as at BF16. The round-to-nearest draft's acceptance swings
with the single prompt per context (0.3–0.9). Our 64k / 128k runs ran out of memory in the prefill
(the full-attention layers' prefill attention beside the 8-bit target and the 4-bit draft); they are
rerun with smaller prefill chunks below.

### Follow-up F3a results (2026-11-25): fused small kernels

**Compiling the whole forward failed:** 73 against 156 tokens/s on Qwen3-4B. No graph breaks, but
Inductor turned every in-place KV-cache write into a copy of the whole cache (`fused_copy_` kernels,
~82 µs each, ~70 per token). The working version compiles only pure helpers (each norm with its
residual add, q/k norm + RoPE, the MLP activation, Gemma's softcap, Qwen3.8's whole one-token DeltaNet
update with the depthwise conv written as a 4-term sum) and leaves matmuls, cache writes and attention
as they were. Profile of Gemma 4 12B's step: 3,039 → 1,205 kernels per token, non-matmul time 3.4 →
1.1 ms, step 19.1 → 17.2 ms (matmuls now 93% of it).

| 2k context | Unfused | Fused | Gain | Violations (fused) |
|---|---|---|---|---|
| Qwen3-4B BF16 | 156.7 | 164.6 | +5.0% | 0 / 128 |
| Qwen3-4B BF16 + Q4 draft | 250.5 | 258.6 | +3.2% | – |
| Qwen3-4B Q8_0 | 256.4 | 279.7 | +9.1% | 0 / 128 |
| Qwen3-4B Q8_0 + Q4 draft | 256.7 | **311.3** | **+21%** | – |
| Gemma 4 12B BF16 | 51.4 | 56.7 | +10% | 1 / 256 (unfused 0) |
| Qwen3.8 INT4 | 57.9 | **73.9** | **+28%** | 0 / 256 |

By the F3a bar (≥ 5%, 0 violations): it counts for Qwen3-4B BF16 and Q8_0 and for Qwen3.8; on Gemma it
is faster but has 1 violation where the unfused step had 0, so it does not count there. The largest
gains are where the small kernels were largest: Qwen3.8's DeltaNet update, and the 8-bit target's
self-draft, whose draft steps carried a fixed ~0.9 ms each. Against llama.cpp at 2k: Qwen3-4B Q8_0 311
against 240 tokens/s (1.30×), Qwen3.8 73.9 against 74.7 (level, was 23% behind), Gemma BF16 56.7
against 58.9 (4% behind, was 16%). All contexts follow.

### Follow-up F3a results at every context (2026-11-25)

Stable decode tokens/s, one holdout prompt per context, 256 tokens, median of 2 runs; same weights and BF16 KV
in both engines (llama.cpp b12359, the follow-up J / K runs). "Before" is the unfused exact step (L1 + L2).

| Context | Qwen3-4B Q8_0 before | after | after + Q4 draft | llama.cpp Q8_0 | Gemma 4 12B Q8_0 before | after | after + Q4 draft | llama.cpp Q8_0 | Qwen3.8 INT4 before | after | llama.cpp Q4_0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2k | 252.2 | 279.7 | **311.3** | 239.6 | 81.7 | 96.1 | 97.8 | **97.1** | 57.4 | 72.9 | **74.7** |
| 4k | 240.5 | 264.7 | **292.2** | 226.2 | 81.6 | 95.8 | 51.9 | **96.7** | 56.5 | 71.5 | **74.0** |
| 8k | 220.8 | 241.0 | **243.2** | 205.1 | 81.1 | 95.0 | 110.3 | 95.2 | 54.9 | 69.1 | **72.5** |
| 16k | 191.2 | **206.5** | 178.8 | 173.5 | 80.2 | **94.0** | 101.6 | 93.3 | 52.0 | 64.9 | **69.5** |
| 32k | 149.9 | **159.2** | 136.6 | 128.5 | 77.3 | **91.4** | 67.2 | 89.9 | 47.2 | 57.7 | **60.9** |
| 64k | 104.0 | **108.6** | 82.1 | 81.8 | – | **85.6** | 96.4 | 81.4 | 39.8 | 47.2 | **48.6** |
| 128k | 64.6 | **66.5** | 46.5 | 46.5 | – | **80.0** | – | 68.5 | 30.2 | **34.6** | 34.5 |

Checks at 2k (2 prompts × 128 tokens against the same model's teacher-forced pass): the fused step has
0 violations on all three models (Gemma 4 12B Q8_0 included, so F3a counts there on the 8-bit target; it
did not on BF16). Gemma's Q4 self-draft has 1 violation per prompt, as unfused (follow-up K): it is shown
but not counted as an exact setting, and its speed swings with the prompt (acceptance 0.3–0.9). Bold is
the faster exact setting of the two engines (Gemma's draft column excluded).

- **Qwen3-4B Q8_0:** ours is faster at every context: 1.30× at 2k (draft), 1.19× at 16k, 1.43× at 128k
  (fused step; the draft loses from 16k).
- **Gemma 4 12B Q8_0:** level with llama.cpp from 2k to 16k (0.99–1.01×), ahead at 32k (1.02×), 64k
  (1.05×) and 128k (1.17×). Before fusion it trailed by 15–19%.
- **Qwen3.8-27B INT4:** from 23% behind to 2–7% behind up to 64k, and level at 128k (34.6 against 34.5).
- Gemma at 128k did not fit beside its 4-bit draft; rerun without the draft (`quantize_q8_target(draft=False)`,
  prefill chunks of 256, `gemma-4-12B_tps_q8_fused_128k.json`): 80.0 tokens/s against llama.cpp's 68.5 (1.17×).

## Follow-up L: the KV cache at long context (pre-registered 2026-11-25, before any code or run)

Written before any of it exists, at the user's direction (Qwen3-4B dropped; Qwen3.8-27B, Gemma 4 12B, Inkling 2
Mini and Gemma 4 26B-A4B). Follow-up I promised this pre-registration and it was never written; the matched runs
(J, K, F3a, H) all used a BF16 KV cache in both engines. Bytes read per token from the cache at 128k context,
from the model shapes: Qwen3.8 16 full-attention layers × 4 KV heads × 256 × 2 (K, V) × 2 B = 64 KiB/token, 8.6
GB per step (~4.8 ms at 1.79 TB/s, against a 28.9 ms step); Gemma 4 12B 8 global layers × 1 KV head × 512 (the
40 sliding layers keep 1,024 slots), ~2 GB (~1.2 ms of 12.5 ms). **Predictions written now:** KV compression can
give Qwen3.8 at most ~15% at 128k and Gemma 4 12B at most ~10%; most of Qwen3.8's long-context slowdown (+15 ms
from 2k to 128k) is not bytes but our attention kernel, which reads the whole allocated cache through batched
matmuls with a dense mask.

**L1, a decode-attention kernel (exact).** A Triton split-KV ("flash-decoding") attention for 1–16 query rows
that reads only the filled part of the cache, BF16 K/V, fp32 softmax. It replaces `bmm_attention` in the
one-token step and the verify pass of Qwen35Decoder (and Gemma's global layers). Exact in the usual sense
(greedy check at 2k: 0 violations against the same model's teacher-forced pass). **Counts** at a context if
≥ 5% faster than the current step there.

**L2, an 8-bit KV cache (lossy).** K and V stored as int8 in blocks of 32 along the head dimension with one FP16
scale per block (llama.cpp's q8_0 layout), written at the same time as today's BF16 entries, read by the L1
kernel with dequantisation in registers. **L3, a 4-bit KV cache, TurboQuant-style (lossy).** K and V rotated by
a fixed random orthogonal (Hadamard × random signs) matrix along the head dimension, then 4-bit symmetric codes
in blocks of 32 with an FP16 scale; queries are rotated by the same matrix (q·k is unchanged) and the attention
output is rotated back. (TurboQuant, Zandieh et al. 2025, adds a 1-bit QJL residual for unbiased inner products;
that residual is not implemented here and the method is labelled "TurboQuant-style".)
- **Quality, both:** at 4k, 32k and 128k (Qwen3.8; Gemma 4 12B to the context that fits), one WikiText holdout
  prompt and its true 256-token continuation, teacher-forced through the decode step with the compressed cache
  against the BF16 cache: mean KL of the next-token distributions, top-1 agreement, and the greedy generation's
  violations against the BF16-KV model. **Passes quality** at a context if mean KL ≤ 0.01 and top-1 agreement
  ≥ 99% (the bar used for INT4 weights in L6), and **counts** if it also is ≥ 5% faster than L1 there.
- **Speed:** stable tokens/s at 2k–128k as in follow-up J; Qwen3.8 also with its MTP draft (H2, k = 4).

*Note 2026-11-25, before any llama.cpp run of L:* to keep the runs to a few hours, the llama.cpp KV-type runs use
2k, 8k, 32k and 128k (not all seven contexts); this repo's L1–L3 runs keep all seven.

**llama.cpp, same models (reported only).** `-ctk / -ctv` bf16, q8_0 and q4_0 at 2k–128k for Qwen3.8 (the
matched Q4_0 GGUF, plain and MTP), Gemma 4 12B (matched Q8_0), Gemma 4 26B-A4B (Google's QAT Q4_0 GGUF, plain and
with its official assistant draft) and Inkling 2 Mini (Q4_K_M). Quality with `llama-perplexity` on WikiText-2
test at 32k context, one chunk: perplexity and KL against the bf16-KV run (`--kl-divergence`).

## Follow-up M: looped models (pre-registered 2026-11-25, before any code or run)

Inkling 2 Mini runs its layers 6–18 three times per token (a fixed loop, `loop_config.num_passes = 3`); this
repo's engine does not run it, llama.cpp b12720 does (`inkling2_loop`, the pass count read from the GGUF's
metadata; a fixed loop needs ≥ 2 passes). Everything here is measured in llama.cpp, exactness reported as
llama.cpp's greedy output with and without the draft (identical tokens or not), 256 tokens, one prompt per
context, 2 runs.
- **M1, the model's own draft.** Inkling ships an MTP / DSpark draft (`mtp.safetensors`); llama.cpp exports it
  with `convert_hf_to_gguf.py --dspark` and runs it with `--spec-type draft-dspark`. Tokens/s and acceptance at
  2k–64k against plain decoding.
- **M2, a fewer-passes draft (new for looped models).** The draft is the same model with 2 loop passes instead
  of 3 (a GGUF copy with only the loop-count key changed), quantised further (Q3_K_S) so that both fit in 32 GB,
  run with `--spec-type draft-simple`. A looped model's draft needs no training: fewer passes of its own core.
  Tokens/s and acceptance at 2k–32k.
- *Note 2026-11-25, before any M run:* a Q3_K_S draft (12.6 GB) and the Q4_K_M target (17.5 GB) do not fit in
  32 GB together, so M2's draft is Q2_K (10.9 GB) with 2 passes (`Inkling-2-Mini-Q2_K-2pass.gguf`). Nothing else
  changes.
- *Note 2026-11-25, before any M run, after a literature search (`docs/looped_models_survey.md`):* M1 adds a
  draft-length sweep, `--spec-draft-n-max` 1, 2, 3, 4, 6 and 8, at 2k and 32k (bf16 KV); M2 adds a control
  draft, Q2_K with all 3 passes, to separate fewer passes from fewer bits. The KV-type runs keep k = 8.
- **M3, how much each pass matters (quality only).** Perplexity of the 2-pass model against the 3-pass model
  (`llama-perplexity`, WikiText-2 test, 4k context), to judge adaptive pass counts (early exit) for later work.
- **Counts** (M1, M2): ≥ 1.1× plain decoding with identical greedy output, at a context.
- Gemma 4 26B-A4B is not looped; it gets L's llama.cpp runs and its assistant draft only.

## Prior work (added 2026-11-24, after all runs above)

None of the levers here is new. Each reimplements a published or widely shipped technique. What this
document adds is measurement on this decoder under bars fixed in advance, with every emitted token
checked against the full model. All speedups are against this repository's own decoder
(`src/ple/decode.py`), not against a production engine. Production engines (vLLM;
FlashInfer-based servers) already ship most of these levers, so those gains should not be read as
gains over them; against llama.cpp (measured above) the exact levers reach parity and only the INT4
self-draft is faster. Sources were checked on their primary pages on
2026-11-24; BibTeX keys are in `docs/paper/references.bib`.

| Lever | Technique | Prior work |
|---|---|---|
| Baseline step | One decode step captured and replayed as a CUDA graph to cut launch overhead | Gray 2019 (NVIDIA, CUDA Graphs) |
| L1 fused q/k norm + RoPE | Kernel fusion of elementwise work around attention | FlashInfer fuses normalization and RoPE into its attention kernels (Ye et al. 2025, §3.2.3 and §4.3) |
| L2 fast attention | Attend over the filled cache only, with an IO-aware exact kernel; one-query decode attention split over the KV length | FlashAttention (Dao et al. 2022), FlashAttention-2 (Dao 2024), Flash-Decoding (Dao, Haziza, Massa and Sizov 2023); KV-cache serving in vLLM (Kwon et al. 2023) |
| L2d deterministic one-row step | Avoid kernels whose reduction order changes between runs | He 2025 (Thinking Machines) attributes most run-to-run differences in served models to batch-size-dependent kernels, not to atomics; PyTorch reproducibility notes. Here the batch is fixed at 1 and the cuDNN one-row kernel still varies run to run, so this is a different cause from the one that post addresses. |
| L3 GEMV alternatives | Faster 1-row weight GEMMs | cuBLAS is the baseline; Marlin (Frantar et al. 2025) targets INT4 × FP16 at batch sizes up to 16–32 |
| L4 sparse acceptance checks | Fewer host syncs in the speculative loop | Engineering detail; no specific source |
| L6 / L6b INT4 weights | Group-wise round to nearest; GPTQ; per-group clipping search | Small block size helps 4-bit (Dettmers and Zettlemoyer 2023); GPTQ (Frantar et al. 2023); weight clipping to minimise MSE (AWQ, Lin et al. 2024, §3.2) and learnable clipping (OmniQuant, Shao et al. 2024). Our clipping search is a grid search on squared rounding error, closest to AWQ's. |
| L7 / L7b INT4 self-draft | Speculative decoding with a quantized copy of the target as the draft, full model verifying | Blockwise parallel decoding (Stern et al. 2018); speculative decoding (Leviathan et al. 2023; Chen et al. 2023); self-speculation with skipped layers (Draft & Verify, Zhang et al. 2024; SWIFT, Xia et al. 2025; LayerSkip, Elhoushi et al. 2024); quantized self-drafts: QSpec (Zhao et al. 2025: W4A4 draft, W4A16 verifier, shared weights and KV cache), QuantSpec (Tiwari et al. 2025: 4-bit weights and a hierarchical 4-bit KV cache, aimed at long context), ML-SpecQD (Georganas et al. 2025: the BF16 target direct-cast to MXFP4 as draft) |

L7 is the plain case of that last line: weight-only INT4, a BF16 verifier, greedy acceptance of the
longest agreeing prefix (as in Stern et al. 2018), no training. Its 2.1–2.25× over the original
decoder is in the range those papers report, on a different decoder and GPU.

What is specific to this work:
1. **Measurement discipline.** Bars fixed before each run, dated amendments, exactness checked token
   by token against the original decoder, a fresh-process rerun and a second architecture.
2. **A determinism observation.** On this Windows PyTorch/cuDNN build the one-row cuDNN SDPA kernel
   is not bitwise repeatable at batch 1, while the batched-matmul path is (D1, L2d). This is one
   concrete case of a known class of problem, not a new class.
3. **When a better draft pays.** The GPTQ draft adds 7–9% where the plain INT4 draft's acceptance
   is the bottleneck (Qwen3-4B) and nothing where acceptance is already high (TinyLlama-1.1B).
   That is two models, not a general result.

### References

- Dao, Fu, Ermon, Rudra, Ré. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. https://arxiv.org/abs/2205.14135
- Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR 2024. https://arxiv.org/abs/2307.08691
- Dao, Haziza, Massa, Sizov. Flash-Decoding for long-context inference. Stanford CRFM blog, Oct 2023. https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023. https://arxiv.org/abs/2309.06180
- Ye et al. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. MLSys 2025. https://arxiv.org/abs/2501.01005
- He et al. Defeating Nondeterminism in LLM Inference. Thinking Machines Lab blog, Sep 2025. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- PyTorch. Reproducibility notes. https://docs.pytorch.org/docs/stable/notes/randomness.html
- Gray. Getting Started with CUDA Graphs. NVIDIA Technical Blog, Sep 2019. https://developer.nvidia.com/blog/cuda-graphs/
- Frantar, Castro, Chen, Hoefler, Alistarh. MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models. PPoPP 2025. https://arxiv.org/abs/2408.11743
- Dettmers, Zettlemoyer. The case for 4-bit precision: k-bit Inference Scaling Laws. ICML 2023. https://proceedings.mlr.press/v202/dettmers23a.html
- Frantar, Ashkboos, Hoefler, Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR 2023. https://arxiv.org/abs/2210.17323
- Lin et al. AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration. MLSys 2024. https://arxiv.org/abs/2306.00978
- Shao et al. OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models. ICLR 2024. https://arxiv.org/abs/2308.13137
- Stern, Shazeer, Uszkoreit. Blockwise Parallel Decoding for Deep Autoregressive Models. NeurIPS 2018. https://arxiv.org/abs/1811.03115
- Leviathan, Kalman, Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023. https://proceedings.mlr.press/v202/leviathan23a.html
- Chen et al. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv 2302.01318, 2023. https://arxiv.org/abs/2302.01318
- Zhang et al. Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding. ACL 2024. https://aclanthology.org/2024.acl-long.607/
- Xia et al. SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration. ICLR 2025. https://arxiv.org/abs/2410.06916
- Elhoushi et al. LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. ACL 2024. https://aclanthology.org/2024.acl-long.681/
- Zhao et al. QSpec: Speculative Decoding with Complementary Quantization Schemes. EMNLP 2025. https://aclanthology.org/2025.emnlp-main.240/
- Tiwari et al. QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache. ICML 2025. https://proceedings.mlr.press/v267/tiwari25b.html
- Georganas, Kalamkar, Kozlov, Heinecke. ML-SpecQD: Multi-Level Speculative Decoding with Quantized Drafts. arXiv 2503.13565, 2025. https://arxiv.org/abs/2503.13565

### Status of follow-ups L and M, 2026-11-25 (runs stopped at the user's direction; testing resumes Monday)

Done (results in `results/processed/decode_levers/`):
- **L1:** Qwen3.8 with the Triton split-KV kernel and a BF16 cache, 2k–64k with MTP, 128k plain
  (`Qwen3.8-27B-INT4_tps_kv_bf16.json`). The step is exact at 2k (fused step 0 violations, MTP 0 violations).
- **L2 / L3 speed:** Qwen3.8 int8 and int4 caches, 2k–128k (`..._tps_kv_int8.json`, `..._tps_kv_int4.json`).
- **L2 / L3 quality:** Qwen3.8 at 4k, 32k and 128k (`Qwen3.8-27B-INT4_kv_quality.json`):
  - int8 passes at 4k (KL 0.0006, top-1 99.2%) and 32k (0.0005, 99.6%) but misses at 128k (0.0006, 98.8%);
    0 violations everywhere.
  - int4 fails at 4k (0.0024, 96.5%) and 32k (0.0024, 98.8%). Its 128k run ran out of memory and the
    rerun is pending.
- **llama.cpp KV types:** Qwen3.8 (plain + MTP), Gemma 4 12B Q8_0, Gemma 4 26B-A4B QAT (plain + assistant
  draft) and Inkling (plain + DSpark k = 8). 2k with bf16 / q8_0 / q4_0 caches and 8k with bf16 are complete;
  8k with q8_0 is partial (`llamacpp_b12720/kv/`).

Not run yet:
- Our engine: Gemma 4 12B (KV store wired in, untested), and Qwen3.8 MTP at 128k with 512-row prefill chunks.
- llama.cpp: KV types at 8k (rest), 32k and 128k.
- M1's draft-length sweep, M2 (2-pass and 3-pass Q2_K drafts), M3 and the `llama-perplexity` KL checks.
