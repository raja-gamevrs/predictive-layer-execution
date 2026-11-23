# Predictive Layer Execution (PLE)

A falsifiable investigation into speculating downstream transformer projections from
*predicted* intermediate activations, then reconciling that speculative work against the
exact computation.

> PLE is an unproven hypothesis. The goal is reproducible evidence — including negative
> results. A valid outcome of this project is GO, PIVOT, or STOP.

## Idea

```
Exact:      h_(l+1) = F_l(h_l)
Predict:    ĥ_(l+1) = h_l + P_l(h_l)
Speculate:  ŷ = W ĥ_(l+1) + b
Error:      e = h_(l+1) − ĥ_(l+1)
Correct:    y = ŷ + W e          (exact, but W e can cost as much as recomputing)
```

The research problem is finding corrections that are cheaper than recomputation
(low-rank, block, sparse), or enough real overlap on the GPU to cut critical-path latency
at matched output quality.

## Starting stack

| Component | Choice |
|---|---|
| Primary model | [Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) |
| Secondary model | [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) (base) |
| Debug model | TinyLlama-1.1B class |
| Model execution | PyTorch + Hugging Face Transformers |
| Initial precision | BF16 (metric accumulators in FP32) |
| Quantization experiments | INT4/NF4 later, as an ablation only |
| Initial GPU | RTX 3090 24 GB |
| Comparative GPU | RTX 5090 32 GB |
| First experiment | Hidden-state prediction + projection correction economics |
| First workload | Prefill |
| Paper decision | After novelty review + experimental gate |

## Stage plan

| Stage | Work | Gate |
|---|---|---|
| A | Literature, prior-art ledger, novelty boundary | Written novelty boundary before coding |
| B | Pinned baseline, exact reference, activation capture | Baseline CV < 5% |
| C | Predictors: identity, linear, low-rank, tiny MLP | Beats identity / zero-delta on held-out data |
| D | Correction economics microbenchmarks | A structured correction materially cheaper than exact projection |
| E | Runtime overlap with CUDA streams/events | Trace shows real concurrency, no severe contention |
| F | Prefill sweeps, then decode | ≥ 5% stable latency gain at matched quality |

Thresholds are pre-registered in [`docs/hypothesis_preregistration.md`](docs/hypothesis_preregistration.md)
and [`configs/quality_thresholds.yaml`](configs/quality_thresholds.yaml). Changing them after
results are visible requires a dated rationale and a rerun, and is labelled exploratory.

## Layout

```
configs/          model, sweep and threshold configs
docs/             research review, novelty matrix, preregistration, protocol, results
src/ple/          library code (loader, exact reference, capture, predictors, correction, ...)
tests/            correctness tests (exact path, correction identity, commit safety, ...)
scripts/          entry points (verify_model, capture_traces, fit_predictors, bench_*)
data/             prompt manifest; traces/ is git-ignored
results/          raw/, processed/, plots/
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
pytest
python scripts/verify_model.py --config configs/debug_tinyllama.yaml
```

## Rules of the road

- The exact forward path is never modified; speculative tensors are never committed unverified.
- Never report theoretical FLOP savings as measured latency.
- Keep dtype, batch, sequence and software versions matched across GPUs; 3090 vs 5090 is a
  hardware comparison, not a PLE ablation.
- Predictor training prompts and evaluation prompts are disjoint.
- Publish everything, including regressions and no-speedup cases.

## License

Code is Apache-2.0. Model weights keep their own licenses — check each model card.
