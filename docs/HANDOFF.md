# Handoff — Predictive Layer Execution (state as of 2026-11-23 17:25 PT)

Written for the Claude session that takes over this project. Read this, then
`docs/results_report.md` (dated log of every run) and `docs/paper/draft.md` (paper draft).

## What the project is

Repo `C:\Users\gamev\predictive-layer-execution`, public at
https://github.com/raja-gamevrs/predictive-layer-execution (owner: raja-gamevrs, the user's personal
account; `main` is pushed and clean). Origin: two blueprint .docx files the user supplied
(research plan + "layer manipulation techniques"); their substance is in `README.md`,
`docs/techniques.md` and `docs/hypothesis_preregistration.md`. The blueprints themselves are **not**
in the public repo (deliberately).

PLE = predict block l+1's input from h_l, launch block l+1's attention input projections early,
reconcile against the exact state (`y = W·N(ĥ) + W·e`, cheap approximate corrections, or none).

## Where it stands

**As a speedup method it failed the gates** (correction economics and single-GPU overlap), and the
mechanism is already published (Layer Parallelism, CQIL, EasySpec, InfiniGen, RidgeFill, SNLP — see
`docs/research_review.md`, verified 2026-11-23). The user agreed to reframe as an **empirical /
negative-results paper built around the precision finding**:

> Exact reconciliation survives weight-only quantisation (INT8 W8A16, NVFP4 W4A16) but breaks under
> activation quantisation (NVFP4 W4A4 emulated: 15–100× above the BF16 floor on Qwen3.5-4B; real
> NVIDIA ModelOpt FP8 W8A8 attention on Ornith-1.5-9B: 3–10×). NVFP4 kernels are W4A4 at prefill and
> W4A16 at decode, so exactness fails in prefill and holds in decode.

Full results summary (all runs) is `docs/results_summary.md`; detail in `docs/results_report.md`
and sections 4.1–4.7 of the draft. Key supporting results: errors must be measured after the next block's norm; BF16
"reformulation floor" (KL ≈ 0.001, 1–2% top-1 flips) is the reference for any exactness claim;
per-layer sensitivity map (late-mid layers tolerant, first/last sensitive, linear-attention blocks
more tolerant than softmax blocks); cumulative budget ≤ 1% ppl ≈ 9/36 blocks (Qwen3-4B), 3/32
(Qwen3.5-4B); lookahead up to k = 8 costs only ~1.5% per block; overlap: 0–47% real kernel
concurrency but contention cancels it.

## Open decisions (ask the user; do not decide for them)

1. **Rename.** User wants more tests before renaming. Options offered: "Speculate, Then Reconcile?
   Exactness and Economics of Intra-Forward Speculation in Quantized LLMs", "Exact Reconciliation
   Breaks Under Activation Quantization", "When Can a Transformer Start Its Next Layer Early?".
   "PLE" collides with Gemma's Per-Layer Embeddings; at minimum always spell it out. Repo rename
   also undecided.
2. **Direction** beyond the paper: optional pivot toward RidgeFill-style *skipping* (where published
   wins are) was offered, not decided.

## Next planned work

1. **Qwen3.8-27B** — last model in the plan. Downloads into `G:\models` (see below). Main = 
   `RedHatAI/Qwen3.8-27B-INT4` (compressed-tensors W4A16, `71dface`); ablation =
   `nvidia/Qwen3.8-27B-NVFP4` (ModelOpt, W4A4 per config, `482ca0f`). Needs configs (copy the
   pattern of `configs/ornith_15_9b_modelopt.yaml`; INT4 loads via Transformers + `compressed-tensors`,
   already installed). Check the NVIDIA checkpoint's `hf_quant_config.json`: `ple.quant.modelopt`
   currently supports only `FP8` and `W4A16_NVFP4` layer algos; an NVFP4 W4A4 algo will need a new
   branch (dequantise weights + `nvfp4_fake_quantize` activations with the stored scales).
   27B BF16 does not fit the 5090; dequantised INT4 → BF16 would be ~55 GB, so the INT4 model must
   run through its compressed-tensors kernels, and correction must use `full_module` only.
2. Optional: confirm W4A4 on a **real NVFP4 kernel** (no Windows build; Linux/WSL2 Ubuntu 24.04 is on
   this PC but has no Rust/Node/PyTorch set up there).
3. Optional: decode-with-KV-cache test (all quality numbers so far are prefill/teacher-forced).
4. Paper: finalise pre-registration tolerances (currently exploratory), rewrite abstract around
   the precision result.

## How to run things

- Python env: `C:\Users\gamev\predictive-layer-execution\.venv` (PyTorch 2.14.1+cu130,
  Transformers 5.24.0, torchao 0.18, kernels 0.18.2, compressed-tensors 0.20.1; lock in
  `requirements-lock.txt`). Tests: `.\.venv\Scripts\python -m pytest -q` (28 pass).
- Env vars used for every run: `HF_HOME=G:\hf`, `HF_HUB_OFFLINE=1` (models are local),
  `TQDM_DISABLE=1`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for 9B+ models.
- Models on disk: HF cache `G:\hf` (Qwen3-4B-Base, TinyLlama, WikiText-2) and `G:\models\<org>\<name>`
  (Qwen3.5-4B-Base, Ornith-1.5-9B, Ornith-1.5-9B-NVFP4; Qwen3.8 in progress). Every config pins the
  HF commit; `path:` configs were downloaded at that commit with sha256 checks.
- Scripts (all write CSV + JSON metadata to `results/processed/` named via `run_name(cfg)`):
  `verify_model.py`, `predictability_study.py`, `error_structure_study.py`,
  `approx_quality_study.py` (`--corrections none full_module` for quantised models),
  `depth_sweep_study.py`, `bench_overlap.py` (`--graphs`, `--profile`). Exact commands are in
  `docs/paper/draft.md` §8.
- Downloads: `scripts/download_models.ps1 -Repos "org/name@commit,..."` (curl, resumable, sha256).
  Background jobs in Claude Code stop at 2 h; the script resumes `.part` files on rerun.
- GitHub: portable `gh` at `%LOCALAPPDATA%\Programs\gh\bin\gh.exe`, signed in as raja-gamevrs;
  git pushes use it. Commit messages end with the Co-Authored-By line used in `git log`.

## Gotchas learned the hard way

- Windows shows stale file sizes for files still being written: read true size by opening the file
  (`[IO.File]::Open(...).Length`), not `Get-ChildItem`.
- The 5090 also drives the desktop (Parsec, ComfyUI may be running): timing CV is above the 5% gate;
  quality metrics are unaffected. Do not restart ComfyUI.
- Qwen3.5/Ornith linear-attention layers run on Transformers' slow reference kernels
  (`flash-linear-attention` not installed) — fine for quality, invalid for timing.
- Transformers' own NVFP4 needs a Hub kernel with no Windows build; use the emulation
  (`quantization: nvfp4_w4a16_emulated | nvfp4_w4a4_emulated`).
- Qwen3.5 full-attention blocks are 3, 7, …, 31: boundaries 2, 6, …, 30 hit only full-attention
  blocks (use `--single-layers 1 5 9 …` for linear-attention blocks).
- Report numbers only after checking them against the CSVs; two table cells were wrong on first
  write and were corrected.

## Update, 2026-11-23 evening (session that took over)

- Overlap rerun on an idle GPU with batched timing: overlap saves +6.9% to +15.7% of a block at
  1–64 tokens, ≤ 2.8% from 512 up (`results_report.md`, "H4 rerun"); fused speculative QKV
  (`scripts/bench_fused_qkv.py`) +1.9% to +5.5%. End-to-end decode test still needed.
- Phase II plan written (`docs/phase2_plan.md`); atlas and static allocation run on both 4B models
  (`scripts/sensitivity_atlas.py`, `scripts/allocation_study.py`): ≤ 1.8% compute within 1% ppl, so
  the plan's stop rule is met at 4B scale. Next direction is the user's call.
- Results dashboard (private artifact): https://claude.ai/artifact/Hp6ECKsrz7RgeR5kA2H87e

## Update, 2026-11-24 (E1–E5)

- Five pre-registered experiments (`docs/experiments_e1_e5.md`): end-to-end decode with a minimal
  CUDA-graph Qwen3 decoder (`src/ple/decode.py`, `scripts/e1_decode_bench.py`), per-token gate,
  LoRA healing, trained predictor, FP8 speculative branch (`scripts/e2_*` to `e5_*`, shared
  `scripts/e_common.py`). **None passes**: the best decode saving is ~6.5% (FP8 branch, all blocks)
  at +18–24% ppl, and no quality fix gets near 1% / KL 0.01 at that coverage.
- Lossless PLE (exact correction) saves no time by construction: the correction is a full projection
  on the critical path. The earlier "+6.9–15.7% of a block" overlap numbers are uncorrected.
- Ridge and E4 predictors are cached in `results/raw/e1_e5/` (gitignored, ~1 GB each).
- Downloaded and hash-checked (no configs yet): `G:\models\nvidia\Qwen3.8-27B-NVFP4` (482ca0f),
  `G:\models\LiquidAI\LFM3-1.7B-Base`, `G:\models\openbmb\MiniCPM5-2B`, `G:\models\google\gemma-4-E4B`.
- `.env` (gitignored) holds `HF_TOKEN`; `hf download` with `HF_XET_HIGH_PERFORMANCE=1` runs at ~44 MB/s.
- E3b (`docs/e3b_healing_scaling.md`, `scripts/e3b_healing_scaling.py`): training Qwen3-4B for up to
  16k steps (8.2M tokens) to expect identity speculation at every block flattens at holdout KL ≈ 0.19 /
  +10% ppl; rank 64 is no better. Pre-registered stop rule 2 fired. A full-rank q/k/v variant
  (`--mode qkv_delta --delta-opt factored`) fits in 14.4 GB but was never evaluated.
- Self-speculative decoding (`docs/self_speculative_decoding.md`, `src/ple/self_spec.py`,
  `scripts/self_spec_bench.py`): atlas-chosen skipped sub-layers draft k tokens, the full model
  verifies in one pass. Output is exact, but every setting is slower than exact decoding (best 0.94×
  holdout; bar 1.3×). A draft step costs 48–82% of a full step, verify ~1 step; only drafts too
  inaccurate to be accepted are cheap enough. Time with `--time-only` (interleaved exact/self-spec
  pairs); the first, contended timing pass is kept as `*_timed_contended.json` and not used.
- Decode levers (`docs/decode_levers.md`, `scripts/lever_*.py`): attention was 22% of the decode step
  at 3% of bandwidth; GEMMs are already 78–87% of peak. `Qwen3Decoder(..., attn="fast", rope="fused")`
  is exact and 1.36× faster (6.42 ms/token). `quantize_int4()` + `SelfSpecDecoder(..., draft_w4=True)`
  (INT4 copy drafts, BF16 verifies) is exact and 1.45–1.56× faster again at k = 4. One verify-pass
  rounding flip (gap 0.375) shows up at k = 2–3 on one prompt. INT4 alone is +16–19% ppl. Flash
  attention and Triton are unavailable on this Windows build; prefill timing is eager and noisy.
- Decode levers follow-up (same doc, `scripts/lever_int4.py`): `quantize_int4(method="gptq", clip=True,
  group=32, hessians=...)` cuts INT4 KL ~6× (holdout 0.186 → 0.034) but still fails the 0.01 / 1% bar.
  As a draft on Qwen3-4B it is exact and 2.13× / 2.25× over the original at k = 6 (L7b, +7–9% over
  L7); on TinyLlama it is not better than RTN g128. cuDNN's one-row SDPA is not bitwise deterministic:
  `dec.step_attn = "bmm"` (L2d) makes the fast path repeatable at −5% (Qwen3-4B). Code prompts are now
  read from commit a118bda (`--code-rev`). The decoder runs Llama blocks; TinyLlama replicates L1, L2
  and L7 (1.60× exact, 2.14× / 2.18× with the INT4 draft).
- Decode levers are known techniques, not contributions: the "Prior work" section at the end of
  `docs/decode_levers.md` maps each lever to its sources (keys in `docs/paper/references.bib`), and
  "Raw decode speed" gives ms/token and tokens/s (Qwen3-4B 115 → 156 exact → 244–258 tokens/s with
  the L7b draft). The baseline is this repo's decoder. llama.cpp b12359 (G:/llamacpp, GGUFs converted
  locally; `scripts/llamacpp_bench.py`): BF16 162 / 540 tokens/s (Qwen3-4B / TinyLlama), so L1 + L2 is at
  parity; its own Q4_0-draft speculation 156–185 / 435–445, below our INT4 self-draft (244–258 / 698–709).
  vLLM and SGLang not measured.
- Gemma 4 12B (follow-up E, `src/ple/gemma4_decode.py`, `scripts/decoder_gate.py`): decoder passes the gate only
  after the user's amendment (our violations ≤ Transformers' own self-inconsistency: 8 ≤ 31). Must use separate
  q/k/v and gate/up GEMMs (a concatenated GEMM's rounding alone changes logits). Levers 1.07 / 1.11 / 1.20× but
  the original decoder has 9 self-violations and levers 13–35: no lever counts. INT4 copy does not fit (24 GB
  BF16 + ComfyUI). llama.cpp BF16 59 tokens/s vs our 51.
- Follow-up F (pre-registered, `scripts/lever_draft2.py`): layer-skipped and reduced-vocabulary INT4 drafts on
  Qwen3-4B, then a compiled step after installing triton-windows 3.9.0.post30 (approved). Not run yet.
- Follow-up G (pre-registered): Qwen3.8-27B INT4 decoder `src/ple/qwen35_decode.py`, three-pass gate
  `scripts/decoder_gate_split.py` (logits kept in G:/ple_cache/gate). GGUF via `G:/llamacpp/tools/convert_skip_kv_scales.py`
  (the converter cannot map the k_scale / v_scale tensors).
- Follow-up J (matched against llama.cpp, user-amended to 2k–128k, tokens/s only): `scripts/tps_bench.py`,
  `scripts/llamacpp_bench.py --context C --kv bf16 --no-check`, matched GGUFs from `scripts/make_matched_gguf.py`
  (Qwen3-4B Q4_1 draft from our saved codes `G:/ple_cache/draft_qwen3_4b_gptq_clip_g32.pt`; Qwen3.8 Q4_0 from the
  RedHat codes). Qwen3-4B: ours faster at every context; Gemma / Qwen3.8: llama.cpp faster, gap closing at 128k.
  The self-draft loses at 64k (draft attends the full BF16 cache): a windowed or quantized-KV draft is the next lever.
  Qwen3Decoder now points Transformers' q/k/v and gate/up at slices of its concatenated copies (no duplicate weights).
- Not done: MTP self-spec on Qwen3.8 (`src/ple/mtp_spec.py`, `scripts/mtp_bench.py` written, never run); follow-up F
  (layer-skipped / reduced-vocabulary drafts, `scripts/lever_draft2.py`, never run); triton-windows not installed.
  Stock Transformers `generate` (eager; no Triton on Windows) on the same prompts
  (`scripts/hf_baseline_bench.py`): 32 tokens/s Qwen3-4B, 66 TinyLlama, same greedy text (0 violations);
  L1 + L2 is 5.0× / 7.9× faster in the same process.

## In flight at handoff

- Download job in the *previous* session (gvpro-chat) for `RedHatAI/Qwen3.8-27B-INT4` (6.4 of
  18.1 GB at 17:20) then `nvidia/Qwen3.8-27B-NVFP4` (20 GB). It will stop at its 2-hour limit
  (~18:25). Rerun to finish:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\download_models.ps1 -Repos "RedHatAI/Qwen3.8-27B-INT4@71dface38d4cbfa34121f1d446d4a0fa1025f13c,nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80"`
  (check no other `curl.exe` is writing the same `.part` first).
- Nothing else is running on the GPU.

## Update, 2026-11-25 (overnight: quantized targets, fused kernels, MTP, Inkling, story)

- **Plain-language story** of the whole project: `docs/STORY.md`. Article drafts (not posted):
  `docs/articles/medium.md`, `docs/articles/linkedin.md`; charts in `docs/articles/figures/` from
  `scripts/make_charts.py --png` (SVG, PNG rendered with headless Chrome using its own profile dir).
- **Dashboard rebuilt** as one chronological story (files/index.html version 17; new dataset `story` =
  `results/processed/story/story.json` from `scripts/build_story_data.py`, uploaded as an asset). Source
  in `dashboard/` (see its README). Write it only through ArtifactData, never a file publish. An older
  `story.json` asset (c68fa6c2…) is unreferenced; deleting it needs the user's OK.
- **Q8_0 targets (follow-up K):** `ple.kernels` (byte-identical Q8_0 quantiser on the CPU, Triton W8A16
  GEMV / small-GEMM). `tps_bench.py --target q8`. Gemma's `quantize_q8_target(..., draft=False)` skips the
  INT4 draft copy (used for the 128k rerun).
- **F3a fused helpers** (`compile_step(True)`, settings `l12c` / `l7bc`): exact on all three models; see
  the all-context table in `docs/decode_levers.md`. Qwen3.8 also has a fused multi-row verify pass
  (`_f_delta_verify`) for MTP.
- **Qwen3.8 MTP (follow-up H, amended 2026-11-25):** `tps_bench.py --settings l12c --mtp-ks 2 3 4`
  (`ple.mtp_spec.MTPSelfSpec.prefill_long / reset / capture(reset_fn)`). Output
  `Qwen3.8-27B-INT4_tps_mtp.json`; the k = 1–3 smoke run is `..._tps_mtp_smoke.json`. k = 2 shows 1
  violation in 256 tokens (both runs); k = 3 and 4 show 0.
- **llama.cpp b12720** at `G:/llamacpp/b12720/bin` (MTP: `--spec-type draft-mtp`); matched runs
  before it used b12359 (`G:/llamacpp/bin`). b12720 results go to
  `results/processed/decode_levers/llamacpp_b12720/` so they never overwrite the b12359 files.
- **Inkling 2 Mini:** converted with b12720's own converter (`convert_hf_to_gguf.py`, NVFP4 experts
  dequantised; BF16 GGUF 52 GB) and `llama-quantize ... Q4_K_M` (17.5 GB), both in
  `G:/llamacpp/gguf/inkling/`. Config `configs/inkling2_mini.yaml`. Benchmarked in llama.cpp only; our
  decoders do not run its architecture (MoE 128×8, looped layers, d_rel attention).
- 2-bit draft results (llama.cpp, exploratory) moved into the repo: `results/processed/decode_levers/q2_draft/`.

**Results of the overnight runs (2026-11-25), all exact unless noted** (full tables: `docs/decode_levers.md`,
follow-ups F3a, H, H2; plain story: `docs/STORY.md`):
- Qwen3-4B Q8_0: ours 1.19–1.43× llama.cpp at 2k–128k. Gemma 4 12B Q8_0: level to 16k, 1.17× at 128k
  (128k rerun without the draft: `gemma-4-12B_tps_q8_fused_128k.json`).
- Qwen3.8 fused step: 2–7% behind llama.cpp plain, level at 128k. MTP: the first run fed the pre-norm hidden
  (bug; `*_mtp_prenorm*.json`); post-norm `*_tps_mtp.json` (2k, with check), `*_tps_mtp_rest.json`
  (4k–64k), `*_tps_mtp_128k.json` (512-row prefill chunks). H2 (INT4 draft-only LM head, `--mtp-head int4`):
  `*_tps_mtp_h2.json`, `*_tps_mtp_h2_128k.json` (the 128k H2 run paged ~0.2 GB to system RAM). Best:
  176 tokens/s at 2k vs llama.cpp's own MTP 158; ahead at 5 of 7 lengths, 2–6% behind at 8k and 64k.
- llama.cpp b12720 MTP runs: `results/processed/decode_levers/llamacpp_b12720/` (128k in `mtp128k/`).
  `llamacpp_bench.py` now estimates MTP memory as one extra layer, not a second KV cache.
- Inkling 2 Mini (llama.cpp Q4_K_M): 97 → 66 tokens/s at 2k → 128k (128k needs `--extra "-ub 128 -b 128"`).
- Not done: Inkling in our engine; a browser check of the published dashboard (the page was rendered
  locally with a stand-in runtime, no errors); nothing committed.

## Update, 2026-11-25 (later): the Radan engine

The engine modules (`decode`, `gemma4_decode`, `qwen35_decode`, `kernels`, `mtp_spec`, `self_spec`, `quant`) moved
to the installable package `radan-engine/` (`radan_engine`, v0.1.0, Apache-2.0), named after Radan, the user's
open-source local-agents project. `src/ple/<module>.py` are aliases (`sys.modules` points at the
`radan_engine` module), so every script and doc reference to `ple.*` still works; install the engine first
(`pip install -e radan-engine`). New: `radan_engine.Engine.load(path)` / `.generate(...)` and
`python -m radan_engine`. Checked on the GPU through the API: Qwen3-4B Q8_0 274 tokens/s, Gemma 4 12B Q8_0 96,
Qwen3.8 74 plain / 112 with MTP on a short prompt; 0 violations for every path (64 tokens, teacher-forced).
Defaults: Qwen3 / Gemma no draft (the Qwen3 INT4 self-draft is opt-in with GPTQ codes: on a short generic prompt
it was slower, 254 vs 273), Qwen3.8 MTP with the INT4 draft head, k = 4.
