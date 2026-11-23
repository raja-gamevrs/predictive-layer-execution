# Reference environment — RTX 5090 (baseline GPU)

Recorded 2026-11-23 by `scripts/verify_model.py`. Exact packages: [`requirements-lock.txt`](../requirements-lock.txt).

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB, compute capability 12.0 |
| Driver | 617.42 (CUDA 13.4 capable) |
| CPU / RAM | Intel Core i7-13700KF / 64 GB |
| OS | Windows 11 Pro 10.0.26300 |
| Python | 3.11.14 |
| PyTorch | 2.14.1+cu130 |
| Transformers | 5.24.0 |

## Model smoke tests (BF16, batch 1, single short prompt)

| Model | Revision | Layers (hidden states) | Peak VRAM |
|---|---|---|---|
| TinyLlama-1.1B-intermediate-step-1431k-3T | `59f6f37` | 22 (23) | 2.24 GB |
| Qwen3-4B-Base | `906bfd4` | 36 (37) | 8.14 GB |

Not yet measured: timing baselines, variance, longer sequences.
