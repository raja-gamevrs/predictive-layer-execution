"""Custom GPU kernels (Triton) for the decode harness (docs/decode_levers.md, follow-up K).

q8_0_quantize   llama.cpp's Q8_0 bit for bit (quantize_row_q8_0_ref, computed on the CPU: GPU float
                division / FMA round differently): blocks of 32 along the input dim, d = max|x| / 127,
                q = roundf(x * (1 / d)) in -127..127, d stored as FP16. The weight is q * fp16(d).
w8a16           y = x W^T for x (M, K) BF16 and Q8_0 weights (q int8 (N, K), d fp16 (N, K // 32)).
                M = 1: Triton GEMV, fp32 accumulation. 2 <= M <= 16: Triton tile kernel (weights
                dequantised to BF16 per tile, tensor-core dot, fp32 accumulation). M > 16 (prefill):
                dequantise to BF16 and cuBLAS.
"""

import torch
import triton
import triton.language as tl


def q8_0_quantize(w: torch.Tensor):
    """w (N, K) float -> (q int8 (N, K), d fp16 (N, K // 32)) on w's device, bit-for-bit llama.cpp's Q8_0."""
    N, K = w.shape
    assert K % 32 == 0
    x = w.detach().to("cpu", torch.float32).view(N, K // 32, 32)
    d = x.abs().amax(-1) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    y = x * inv[..., None]
    q = torch.sign(y) * torch.floor(y.abs() + 0.5)              # roundf: half away from zero
    return q.to(torch.int8).view(N, K).contiguous().to(w.device), d.to(torch.float16).contiguous().to(w.device)


def q8_0_dequantize(q, d, dtype=torch.bfloat16):
    N, K = q.shape
    return (q.view(N, K // 32, 32).float() * d.float()[..., None]).view(N, K).to(dtype)


@triton.autotune(
    configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w) for bn in (2, 4, 8) for bk in (512, 1024, 2048) for w in (4, 8)],
    key=["N", "K"])
@triton.jit
def _gemv_q8(x_ptr, q_ptr, d_ptr, y_ptr, N, K, BN: tl.constexpr, BK: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    nmask = rn < N
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        kmask = rk < K
        w = tl.load(q_ptr + rn[:, None] * K + rk[None, :], mask=nmask[:, None] & kmask[None, :], other=0)
        rg = k0 // 32 + tl.arange(0, BK // 32)
        s = tl.load(d_ptr + rn[:, None] * (K // 32) + rg[None, :], mask=nmask[:, None] & (rg[None, :] < K // 32), other=0)
        xv = tl.load(x_ptr + rk, mask=kmask, other=0.0).to(tl.float32)
        prod = tl.reshape(w.to(tl.float32) * xv[None, :], (BN, BK // 32, 32))
        acc += tl.sum(tl.sum(prod, axis=2) * s.to(tl.float32), axis=1)
    tl.store(y_ptr + rn, acc.to(tl.bfloat16), mask=nmask)


@triton.autotune(
    configs=[triton.Config({"BN": bn, "BK": bk}, num_warps=w) for bn in (16, 32, 64) for bk in (128, 256) for w in (4,)],
    key=["N", "K"])
@triton.jit
def _gemm_q8_small(x_ptr, q_ptr, d_ptr, y_ptr, M, N, K, BN: tl.constexpr, BK: tl.constexpr):
    rn = tl.program_id(0) * BN + tl.arange(0, BN)
    rm = tl.arange(0, 16)
    nmask, mmask = rn < N, rm < M
    acc = tl.zeros((16, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        kmask = rk < K
        xt = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=mmask[:, None] & kmask[None, :], other=0.0)
        w = tl.load(q_ptr + rn[:, None] * K + rk[None, :], mask=nmask[:, None] & kmask[None, :], other=0)
        rg = k0 // 32 + tl.arange(0, BK // 32)
        s = tl.load(d_ptr + rn[:, None] * (K // 32) + rg[None, :], mask=nmask[:, None] & (rg[None, :] < K // 32), other=0)
        wd = tl.reshape(tl.reshape(w.to(tl.float32), (BN, BK // 32, 32)) * s.to(tl.float32)[:, :, None], (BN, BK))
        acc = tl.dot(xt.to(tl.bfloat16), tl.trans(wd.to(tl.bfloat16)), acc)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=mmask[:, None] & nmask[None, :])


def w8a16(x: torch.Tensor, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """x (M, K) BF16 @ dequant(q, d)^T -> (M, N) BF16."""
    M, K = x.shape
    N = q.shape[0]
    if M > 16:
        return torch.nn.functional.linear(x, q8_0_dequantize(q, d, x.dtype))
    x = x.contiguous()
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if M == 1:
        _gemv_q8[lambda meta: (triton.cdiv(N, meta["BN"]),)](x, q, d, y, N, K)
    else:
        _gemm_q8_small[lambda meta: (triton.cdiv(N, meta["BN"]),)](x, q, d, y, M, N, K)
    return y
