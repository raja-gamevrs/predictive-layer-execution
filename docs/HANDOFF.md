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

## In flight at handoff

- Download job in the *previous* session (gvpro-chat) for `RedHatAI/Qwen3.8-27B-INT4` (6.4 of
  18.1 GB at 17:20) then `nvidia/Qwen3.8-27B-NVFP4` (20 GB). It will stop at its 2-hour limit
  (~18:25). Rerun to finish:
  `powershell -NoProfile -ExecutionPolicy Bypass -File scripts\download_models.ps1 -Repos "RedHatAI/Qwen3.8-27B-INT4@71dface38d4cbfa34121f1d446d4a0fa1025f13c,nvidia/Qwen3.8-27B-NVFP4@482ca0f3832238542f8f5295dde86b5f22711d80"`
  (check no other `curl.exe` is writing the same `.part` first).
- Nothing else is running on the GPU.
