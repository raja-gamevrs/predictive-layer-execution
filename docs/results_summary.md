# Results summary (all runs to 2026-11-25)

Condensed from `docs/results_report.md`, which has every table, method note and raw-file path.
WikiText-2 held-out text, RTX 5090, BF16 unless stated. "proj err" = relative error at the next
block's attention input projection after its norm. Perplexity changes are relative to the exact model.

## 1. Predictability (H1) — median over boundaries

| Predictor | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| identity | 26% | 34% |
| extrapolation / velocity | 28–33% (worse than identity) | 40–46% (worse) |
| linear, full rank | 18% | 19% |
| linear rank 256 / 64 | 22% / 24% | 24% / 27% |

Consecutive updates are nearly uncorrelated (cos 0.08 / −0.08). Linear explains ~half the update
energy, only at full rank (~63% of a projection's cost).

## 2. Correction economics (H2/H3) — error left, linear predictor

| Correction | Cost (× one projection) | Qwen3-4B | Qwen3.5-4B |
|---|---|---|---|
| none | 0 | 17.8% | 18.7% |
| rank-512 basis | 0.26–0.33 | 10.1% | 12.3% |
| top 25% channels per token | 0.25 | 7.7% | 9.0% |
| worst 50% tokens | 0.50 | 10.6% | 10.9% |

Not low-rank, not token-concentrated; nothing cheap reaches exactness. **Gate failed.**

## 3. Output quality, uncorrected speculation

| Case | Qwen3-4B (11.40) | Qwen3.5-4B (11.82) | Ornith-1.5-9B (12.11) |
|---|---|---|---|
| one layer, identity | +0–1.2% (+0.25% at 26–30) | +1–4%; last block +21% | +0.9% to +26% |
| one layer, linear | +0–0.5% | +0.2–1.7% | −0.3% to +2.1% |
| every layer, identity | 54 | 462 | ~12,000 |
| every layer, linear | 13.25 | 15.52 | 15.93 |
| every layer, linear + top-25% | 11.50 | 12.14 | — |

Linear-attention blocks tolerate speculation 2–4× better than full-attention blocks at mid depth.

## 4. Start depth and lookahead

Early blocks and last 2 exact, linear: ≤ 1% ppl for 9/36 blocks (Qwen3-4B, +0.9%) and 3/32 (Qwen3.5-4B,
+0.8%); from layer 16: +4.1% / +5.5%; from 0: +16% / +29%. Lookahead k (single block, linear): proj
err 18% → 37% from k = 1 to 8 (Qwen3), median ppl cost +1.5% / +1.6% at k = 8.

## 5. BF16 reformulation floor

Exact model is bit-deterministic. Exact correction gives KL 0.0002–0.0015 and 1–2% top-1 flips,
unchanged by FP32 accumulation: BF16 rounding of the output along two equal paths. Report exactness
relative to this floor.

## 6. Precision — exact correction through each projection's own kernel (headline result)

| Model / variant | Exact ppl | Corrected KL (single boundaries) | Top-1 | Verdict |
|---|---|---|---|---|
| Qwen3.5-4B BF16 | 11.82 | 0.0002–0.0006 | 98.8–99.6% | floor |
| Qwen3.5-4B INT8 W8A16 (torchao) | 11.82 | 0.0002–0.0006 | 98.6–99.5% | exact |
| Qwen3.5-4B NVFP4 W4A16 (emulated) | 12.38 | 0.0002–0.0006 | 98.7–99.6% | exact |
| Qwen3.5-4B NVFP4 W4A4 (emulated) | 13.06 | 0.009–0.061 | 86.6–93.4% | broken, 15–100× |
| Ornith-1.5-9B BF16 | 12.11 | 0.0002–0.0011 | 98.5–99.5% | floor |
| Ornith-1.5-9B INT8 W8A16 | 12.09 | 0.0002–0.0009 | 98.5–99.5% | exact |
| Ornith-1.5-9B ModelOpt (FP8 W8A8 attention, real checkpoint) | 12.76 | 0.0008–0.0091 | 96.0–98.6% | broken, 3–10× |
| Qwen3.8-27B INT4 W4A16 (Red Hat, real checkpoint) | 9.33 | 0.0001–0.0006 (all 63 at once: 0.0006) | 98.7–99.8% | exact |

Qwen3.8-27B per layer is the most tolerant model tested (identity ≤ +1% except the last block, +10%;
linear ≤ +0.3%), but all-layer identity collapses hardest (ppl 234k); all-layer linear +13%.

Weight-only quantisation preserves exact reconciliation; activation quantisation breaks it, worst
at early layers. NVFP4 kernels are W4A4 for prefill, W4A16 for decode.

## 7. Overlap timing (Qwen3-4B, block 26 → 27, CUDA graphs)

Idle-GPU rerun, 2026-11-23, each sample = 50 back-to-back graph replays (CV < 1%). Saving per
speculated block, share of sequential time:

| Tokens | Max possible | Overlap, identity | Overlap, linear | Fused single GEMM, identity |
|---|---|---|---|---|
| 1 | 15% | +6.9% | +5.4% | +2.0% |
| 4–16 | 15–18% | +10.5% to +15.7% | +7.4% to +13.7% | +3.9% to +4.7% |
| 32–64 | 16% | +6.6% to +9.7% | +2.4% to +4.9% | +1.9% to +2.4% |
| 128 | 16% | +4.2% | −1.3% | +1.2% |
| 512 | 14% | +2.8% | −2.4% | +2.8% |
| 2048 | 12% | +0.7% | −4.3% | −0.4% |

(Fused column: share of block l's forward time, k = 1, mean of two runs.) Earlier single-replay runs
on a shared GPU missed the small-T gain. **Per block, overlap works at decode-sized inputs; above
512 tokens it does not.** End to end it is bounded by the quality budget (≈ 9 blocks with the linear
predictor, 1–2 with identity): estimated ≈ 2–4% at 1–16 tokens, not yet measured. Eager-mode
profile (shared GPU): 0–47% of side-stream kernel time concurrent, contention cancels it at ≥ 512.

## 8. Prior art

Mechanism published (Layer Parallelism incl. Qwen3-4B, CQIL, EasySpec, InfiniGen); ridge predictor
into later layers published (RidgeFill, Qwen3.5/3.8); predict-then-correct published (SNLP, Song et
al.). New: the precision result (§6), post-norm measurement, the correction-economics and
single-GPU-overlap negative results. "PLE" collides with Gemma's Per-Layer Embeddings.

## 9. Phase II — sensitivity atlas and static allocation (Qwen3-4B, Qwen3.5-4B)

Plan: `docs/phase2_plan.md`. Calibration on WikiText-2 train chunks disjoint from the fitting chunks;
test split untouched. Each block, attention and MLP removed or replaced by a ridge map:

| | Qwen3-4B | Qwen3.5-4B |
|---|---|---|
| Best single block (linear stand-in) | +1.5% ppl, 2.6% compute | +1.5%, 3.0% |
| Single ops ≤ 1% ppl | 25 attention stand-ins (0.6% each) | 13 attention (1.0% each), 2 MLP (1.8%) |
| Greedy set ≤ 1% ppl, calibration and holdout | 2 attention ops, 1.2% compute | 1 MLP op, 1.8% |
| First set removing ≥ 10% compute | 5 ops, +17.0% ppl | 5 ops, +12.6% |
| Rank corr. with KL: local state error / last-block error | 0.82 / 0.98 | 0.69 / 0.98 |

KL is near-additive for pairs (−6% to +15%), super-additive for larger sets (×1.15–1.47 at 5–24 ops,
Qwen3-4B). Middle layers are the most tolerant for blocks and MLPs; block 0 and the last block are
catastrophic. **Static allocation misses the proposed 10%-compute gate by an order of magnitude**;
the plan's stop rule for this route is met at 4B scale (larger models, healing, per-token policies
untested).

## 10. E1–E5 — can approximate PLE pay for itself? (2026-11-24)

Pre-registered in `docs/experiments_e1_e5.md` (full tables there). Bar: KL ≤ 0.01 and ≤ 1% ppl with
≥ 5% lower decode latency. WikiText-2 train calibration / holdout chunks; test split untouched.

| | Result | Verdict |
|---|---|---|
| E1 decode latency (Qwen3-4B, CUDA-graph decoder) | All 35 blocks: +1.2–3.0% (BF16 branch), +6.0–6.7% (FP8 branch); 17 blocks ≤ 3.4%; 9 blocks ≤ 2.0% | 5% only at all blocks + FP8 (+18–24% ppl) |
| E2 per-token gate | Within KL ≤ 0.01: 10–15% of (token, block) pairs speculated at every block (Qwen3-4B), 6–12% (Qwen3.5-4B), at most 27% with blocks 4–32; ~77% needed | Fail (both models) |
| E3 LoRA healing | All blocks +18.2% → +8.8% ppl; healed model drifts (KL 0.04) without speculation | Fail |
| E3b healing scaling (identity, all blocks, `docs/e3b_healing_scaling.md`) | LoRA r16: KL 1.65 → 0.21 at 1k steps, then flat (0.189 at 16k, +10.4% ppl; 0.174 / +9.4% after cooldown); r64 worse | Stop rule fires: no longer run justified |
| E4 trained predictor | Error 0.97× ridge (Qwen3-4B), 1.0× (Qwen3.5-4B); bar 0.8× | Fail |
| E5 FP8 speculative branch | Adds ≤ 0.004 KL; 2–5× the BF16 branch's saving | Own criterion met; overall fail |
| Self-speculative decoding (skipped sub-layers draft, full model verifies; `docs/self_speculative_decoding.md`) | Output exact (0 violations, 480 runs). Best on calibration: atlas ~26%, k = 2 → 0.94× holdout, 0.87× code; no-skip control 0.96×; a 26%-skipped draft step still costs 82% of a full step | Fail (bars 1.3× / 1.1×): slower than exact decoding at every setting |
| Decode levers (Qwen3-4B, `docs/decode_levers.md`): profile, kernels, one fix per part | Exact: fused q/k norm + RoPE 1.058×, fast attention (cuDNN / batched-matmul GQA) 1.269×, both 1.357× (8.71 → 6.42 ms/token, 0 violations). INT4 model 3.28× but +16–19% ppl. INT4 copy as self-spec draft, k = 4: 1.45× holdout, 1.56× code over the fast decoder (≈2× over the original), exact. Prefill: fast attention 1.07× (spread 6%), RoPE 1.03× | L1, L2 count; **L7 passes** (bars 1.3× / 1.1×); L4 (sparse checks) < 0.5%; L6 fails quality; prefill: none counts |
| Decode levers follow-up (`docs/decode_levers.md`): better INT4, replication, determinism | INT4 GPTQ + clip g32: holdout KL 0.034, ppl +2.8% (RTN 0.186, +19%). As draft, k = 6: 1.57× holdout, 1.66× code over L1 + L2 (2.13× / 2.25× over original), exact. Fresh-process rerun: all speedups within 0.5% on identical prompts. cuDNN one-row attention is non-deterministic; batched-matmul step (L2d) 1.29× vs 1.36×. TinyLlama-1.1B: L1 1.08×, L2 1.43×, both 1.60×, INT4 draft 2.14× / 2.18×, exact | L6b fails quality; **L7b improves on L7** (+8.6% / +6.9%) on Qwen3-4B, not on TinyLlama; D1 misses its rule on code (prompts changed) and one L2 flip; L2d counts; **TinyLlama replicates** L1, L2, L7 |
| Engine references and Gemma 4 12B (`docs/decode_levers.md`, references and follow-up E) | llama.cpp b12359 BF16: Qwen3-4B 162, TinyLlama 540 tokens/s (L1 + L2: 156–160, 525); its Q4_0-draft speculation 156 / 185 and 445 / 435, ours 244 / 258 and 698 / 709 (holdout / code). Transformers generate: 32 / 66 tokens/s. Gemma 4 12B decoder: gate fails as written (8 violations; Transformers vs itself 31), passes after the user's control-relative amendment; L1 1.07×, L2 1.11×, both 1.20× (42 → 51 tokens/s); llama.cpp BF16 59 tokens/s | Exact levers reach parity with llama.cpp on Qwen3-4B / TinyLlama, and the INT4 self-draft beats its speculation 1.3–1.6×. On Gemma the original decoder itself has 9 self-violations and the levers 13–35, so **exactness is not established and no Gemma lever counts**; llama.cpp is 17% faster there |
| Qwen3.8 and matched llama.cpp at 2k–128k (`docs/decode_levers.md`, follow-ups G and J) | Qwen3.8-27B INT4 decoder passes its gate (KL 0.0011, top-1 99.41%, 1 violation = Transformers' own); L1 1.02×, L2 1.04×, both 1.06×, exact. Matched weights (BF16 / identical INT4 codes, BF16 KV), stable tokens/s 2k → 128k: Qwen3-4B ours L1 + L2 157 → 56, self-draft 251 → 73 (64k), llama.cpp 157 → 42, its speculation 161 → 52 (64k); Gemma 4 12B ours 51 → 46, llama.cpp 59 → 47; Qwen3.8 ours 57 → 30, llama.cpp 75 → 35 | Qwen3-4B: **ours faster at every context** (best vs best 1.56× at 2k, 1.32× at 128k); Gemma and Qwen3.8: llama.cpp faster (16% / 30% at 2k, 2% / 14% at 128k); the self-draft's gain fades with context and is negative at 64k |
| Quantised targets and fused small kernels (`docs/decode_levers.md`, follow-ups K and F3a) | Q8_0 byte-identical to llama.cpp (Triton W8A16 kernel); fused pure helpers via torch.compile (whole-step compile was slower). Stable tokens/s 2k → 128k: Qwen3-4B Q8_0 ours 280 → 67 (311 with the Q4 draft at 2k), llama.cpp 240 → 47; Gemma 4 12B Q8_0 ours 96 → 86 (64k), llama.cpp 97 → 81 (64k); Qwen3.8 INT4 ours 73 → 35, llama.cpp 75 → 35. 2-bit drafts accepted 27–53% (Qwen3-4B), 0–2% (Gemma) | Fused step exact on all three (0 violations); Qwen3-4B ahead at every context (1.19–1.43×), Gemma level to slightly ahead, Qwen3.8 from 23% behind to level at 128k; 2-bit drafts not worth a kernel |
| Qwen3.8 MTP self-speculation (`docs/decode_levers.md`, follow-ups H and H2) | The checkpoint's MTP layer drafts k tokens, the INT4 model verifies, one CUDA graph per round (DeltaNet state rolled back per row). First run fed the pre-norm hidden (bug, found against llama.cpp's graph; kept as `*_mtp_prenorm.json`). Post-norm: 150 tokens/s at 2k (k = 4) vs fused step 73; H2 adds an INT4 draft-only LM head: 176 at 2k, 1.6–2.4× the fused step at 2k–64k. llama.cpp b12720's own MTP on the same GGUF: 158 at 2k | 0 violations (k = 2–4); H and H2 count; ours (best k) vs llama.cpp's MTP 0.94–1.11× (ahead at 2k, 4k, 16k, 32k, 128k) |
| Inkling 2 Mini (llama.cpp only) | Converted with llama.cpp b12720's converter (NVFP4 experts dequantised), Q4_K_M 17.5 GB: 97 → 66 tokens/s from 2k to 128k | Not run in our engine (MoE, looped layers) |

## Pending

NVIDIA Qwen3.8-27B-NVFP4 (FP8 W8A8 attention, NVFP4 W4A4 MLP) — download stopped at 1.4 / 20.4 GB
(2-hour job limit); loader (`ple.quant.packed`) is ready. Real NVFP4 kernel confirmation
(Linux). Decode test done (E1). New models on disk without configs or hooks: LFM3-1.7B-Base,
MiniCPM5-2B, Gemma-4-E4B (`G:\models`).
