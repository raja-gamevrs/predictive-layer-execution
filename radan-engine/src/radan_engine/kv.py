"""KV cache storage and a split-KV decode-attention kernel (follow-up L of docs/decode_levers.md).

    KVStore(n_layers, n_kv, max_len, head_dim, mode)
        mode "bf16"  BF16 K / V (L1: the kernel alone, exact)
        mode "int8"  int8 codes in blocks of 32 along the head dimension, one FP16 scale per block (llama.cpp's
                     q8_0 layout) (L2, lossy)
        mode "int4"  K / V rotated by a fixed random orthogonal matrix (Hadamard x random signs), then 4-bit
                     codes in blocks of 32 with an FP16 scale, two codes per byte (L3, "TurboQuant-style": no
                     QJL residual) (lossy)

    store.write(j, pos_idx, k, v)        k, v (T, KV, D) after RoPE; quantised and written at pos_idx
    store.attend(j, q, pos0, scale)      q (H, T, D) for rows at positions pos0 .. pos0 + T - 1 (causal),
                                         1 <= T <= 16 -> (H, T, D); reads only the filled part of the cache
    store.dense(j, end)                  BF16 (KV, end, D) K and V for positions < end (prefill chunks)

The kernel splits the sequence across programs (flash-decoding), keeps an fp32 online softmax per split, and a
second kernel merges the splits. Packed 4-bit bytes hold dimension d (low nibble) and d + D/2 (high nibble), so
the kernel works on the two halves of the head dimension without interleaving.
"""

import math

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
MODES = {"bf16": 0, "int8": 1, "int4": 2}


def _configs():
    return [triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=st)
            for bn in (32, 64, 128) for w in (4, 8) for st in (1, 2, 3)]


def _prune(configs, named_args, **kw):
    half = kw.get("HALF", named_args.get("HALF", 128))
    return [c for c in configs if c.kwargs["BLOCK_N"] * half <= 8192 * (2 if c.num_stages == 1 else 1)] or configs[:1]


@triton.autotune(configs=_configs(), key=["HALF", "MODE", "ROWS_TOTAL", "ROWS"], prune_configs_by={"early_config_prune": _prune})
@triton.jit
def _attn_split(Q, K, V, KS, VS, POS0, O, MX, LS,
                T, G, ROWS_TOTAL, SPLIT, qk_scale,
                sq_kv, sq_r, sk_kv, sk_n, ss_kv, ss_n, so_kv, so_s, so_r, sm_kv, sm_s,
                HALF: tl.constexpr, ROWS: tl.constexpr, BLOCK_N: tl.constexpr, MODE: tl.constexpr, GS: tl.constexpr):
    kv = tl.program_id(0)
    sp = tl.program_id(1)
    rb = tl.program_id(2)
    pos0 = tl.load(POS0)
    rows = rb * ROWS + tl.arange(0, ROWS)
    rmask = rows < ROWS_TOTAL
    t = rows % T
    limit = pos0 + t                                   # last key position each row may see
    hi_lim = pos0 + T                                  # keys < hi_lim exist
    start = sp * SPLIT
    end = tl.minimum(start + SPLIT, hi_lim)
    d = tl.arange(0, HALF)
    q_lo = tl.load(Q + kv * sq_kv + rows[:, None] * sq_r + d[None, :], mask=rmask[:, None], other=0.0)
    q_hi = tl.load(Q + kv * sq_kv + rows[:, None] * sq_r + HALF + d[None, :], mask=rmask[:, None], other=0.0)
    m_i = tl.full([ROWS], float("-inf"), tl.float32)
    l_i = tl.zeros([ROWS], tl.float32)
    acc_lo = tl.zeros([ROWS, HALF], tl.float32)
    acc_hi = tl.zeros([ROWS, HALF], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        nmask = n < end
        if MODE == 0:
            k_lo = tl.load(K + kv * sk_kv + n[:, None] * sk_n + d[None, :], mask=nmask[:, None], other=0.0)
            k_hi = tl.load(K + kv * sk_kv + n[:, None] * sk_n + HALF + d[None, :], mask=nmask[:, None], other=0.0)
            v_lo = tl.load(V + kv * sk_kv + n[:, None] * sk_n + d[None, :], mask=nmask[:, None], other=0.0)
            v_hi = tl.load(V + kv * sk_kv + n[:, None] * sk_n + HALF + d[None, :], mask=nmask[:, None], other=0.0)
        elif MODE == 1:
            gg = tl.arange(0, HALF // GS)
            sk_lo = tl.reshape(tl.broadcast_to(tl.load(KS + kv * ss_kv + n[:, None] * ss_n + gg[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)[:, :, None], (BLOCK_N, HALF // GS, GS)), (BLOCK_N, HALF))
            sk_hi = tl.reshape(tl.broadcast_to(tl.load(KS + kv * ss_kv + n[:, None] * ss_n + HALF // GS + gg[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)[:, :, None], (BLOCK_N, HALF // GS, GS)), (BLOCK_N, HALF))
            sv_lo = tl.reshape(tl.broadcast_to(tl.load(VS + kv * ss_kv + n[:, None] * ss_n + gg[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)[:, :, None], (BLOCK_N, HALF // GS, GS)), (BLOCK_N, HALF))
            sv_hi = tl.reshape(tl.broadcast_to(tl.load(VS + kv * ss_kv + n[:, None] * ss_n + HALF // GS + gg[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)[:, :, None], (BLOCK_N, HALF // GS, GS)), (BLOCK_N, HALF))
            k_lo = (tl.load(K + kv * sk_kv + n[:, None] * sk_n + d[None, :], mask=nmask[:, None], other=0).to(tl.float32) * sk_lo).to(tl.bfloat16)
            k_hi = (tl.load(K + kv * sk_kv + n[:, None] * sk_n + HALF + d[None, :], mask=nmask[:, None], other=0).to(tl.float32) * sk_hi).to(tl.bfloat16)
            v_lo = (tl.load(V + kv * sk_kv + n[:, None] * sk_n + d[None, :], mask=nmask[:, None], other=0).to(tl.float32) * sv_lo).to(tl.bfloat16)
            v_hi = (tl.load(V + kv * sk_kv + n[:, None] * sk_n + HALF + d[None, :], mask=nmask[:, None], other=0).to(tl.float32) * sv_hi).to(tl.bfloat16)
        else:
            gg = tl.arange(0, HALF // GS)
            sk_lo = tl.reshape(tl.broadcast_to(tl.load(KS + kv * ss_kv + n[:, None] * ss_n + gg[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)[:, :, None], (BLOCK_N, HALF // GS, GS)), (BLOCK_N, HALF))
            sk_hi = tl.reshape(tl.broadcast_to(tl.load(KS + kv * ss_kv + n[:, None] * ss_n + HALF // GS + gg[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)[:, :, None], (BLOCK_N, HALF // GS, GS)), (BLOCK_N, HALF))
            sv_lo = tl.reshape(tl.broadcast_to(tl.load(VS + kv * ss_kv + n[:, None] * ss_n + gg[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)[:, :, None], (BLOCK_N, HALF // GS, GS)), (BLOCK_N, HALF))
            sv_hi = tl.reshape(tl.broadcast_to(tl.load(VS + kv * ss_kv + n[:, None] * ss_n + HALF // GS + gg[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)[:, :, None], (BLOCK_N, HALF // GS, GS)), (BLOCK_N, HALF))
            kb = tl.load(K + kv * sk_kv + n[:, None] * sk_n + d[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            vb = tl.load(V + kv * sk_kv + n[:, None] * sk_n + d[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            k_lo = (((kb & 15) - 8).to(tl.float32) * sk_lo).to(tl.bfloat16)
            k_hi = (((kb >> 4) - 8).to(tl.float32) * sk_hi).to(tl.bfloat16)
            v_lo = (((vb & 15) - 8).to(tl.float32) * sv_lo).to(tl.bfloat16)
            v_hi = (((vb >> 4) - 8).to(tl.float32) * sv_hi).to(tl.bfloat16)
        s = tl.dot(q_lo, tl.trans(k_lo)) + tl.dot(q_hi, tl.trans(k_hi))          # (ROWS, BLOCK_N) fp32
        s = s * qk_scale
        ok = (n[None, :] <= limit[:, None]) & nmask[None, :]
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        pb = p.to(tl.bfloat16)
        acc_lo = acc_lo * alpha[:, None] + tl.dot(pb, v_lo)
        acc_hi = acc_hi * alpha[:, None] + tl.dot(pb, v_hi)
        m_i = m_new
    tl.store(O + kv * so_kv + sp * so_s + rows[:, None] * so_r + d[None, :], acc_lo, mask=rmask[:, None])
    tl.store(O + kv * so_kv + sp * so_s + rows[:, None] * so_r + HALF + d[None, :], acc_hi, mask=rmask[:, None])
    tl.store(MX + kv * sm_kv + sp * sm_s + rows, m_i, mask=rmask)
    tl.store(LS + kv * sm_kv + sp * sm_s + rows, l_i, mask=rmask)


@triton.jit
def _attn_merge(O, MX, LS, OUT, NS, so_kv, so_s, so_r, sm_kv, sm_s, sout_kv, sout_r,
                D: tl.constexpr, BS: tl.constexpr, CH: tl.constexpr):
    kv = tl.program_id(0)
    r = tl.program_id(1)
    s = tl.arange(0, BS)
    smask = s < NS
    m = tl.load(MX + kv * sm_kv + s * sm_s + r, mask=smask, other=float("-inf"))
    l = tl.load(LS + kv * sm_kv + s * sm_s + r, mask=smask, other=0.0)
    mx = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - mx))
    denom = tl.sum(w * l, 0)
    d = tl.arange(0, D)
    acc = tl.zeros([D], tl.float32)
    for c0 in range(0, NS, CH):
        sc = c0 + tl.arange(0, CH)
        cmask = sc < NS
        mc = tl.load(MX + kv * sm_kv + sc * sm_s + r, mask=cmask, other=float("-inf"))
        wc = tl.where(mc == float("-inf"), 0.0, tl.exp2(mc - mx))
        o = tl.load(O + kv * so_kv + sc[:, None] * so_s + r * so_r + d[None, :], mask=cmask[:, None], other=0.0)
        acc += tl.sum(o * wc[:, None], 0)
    tl.store(OUT + kv * sout_kv + r * sout_r + d, (acc / denom).to(tl.bfloat16))


def _rotation(D: int, device, seed: int = 0) -> torch.Tensor:
    """Orthonormal Hadamard (Sylvester) times a fixed random sign per input dimension: (D, D) fp32."""
    assert D & (D - 1) == 0, "head_dim must be a power of two for the Hadamard rotation"
    h = torch.ones(1, 1)
    while h.shape[0] < D:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (D,), generator=g) * 2 - 1).float()
    return (signs[:, None] * h / math.sqrt(D)).to(device)


class KVStore:
    GS = 32
    BLOCK_N = 128   # split sizes are multiples of the largest tuned block
    ROWS = 16

    def __init__(self, n_layers: int, n_kv: int, max_len: int, head_dim: int, mode: str, device, splits_target: int = 680):
        self.n, self.kv, self.L, self.D, self.mode = n_layers, n_kv, max_len, head_dim, mode
        self.code = MODES[mode]
        self.dev = device
        D, G = head_dim, head_dim // self.GS
        if mode == "bf16":
            self.k = torch.zeros(n_layers, n_kv, max_len, D, device=device, dtype=torch.bfloat16)
            self.v = torch.zeros_like(self.k)
            self.ks = self.vs = self.k                   # unused
        else:
            width = D if mode == "int8" else D // 2
            dt = torch.int8 if mode == "int8" else torch.uint8
            self.k = torch.zeros(n_layers, n_kv, max_len, width, device=device, dtype=dt)
            self.v = torch.zeros_like(self.k)
            self.ks = torch.zeros(n_layers, n_kv, max_len, G, device=device, dtype=torch.float16)
            self.vs = torch.zeros_like(self.ks)
        self.R = _rotation(D, device) if mode == "int4" else None
        self.Rb = self.R.to(torch.bfloat16) if self.R is not None else None
        per = max(1, splits_target // n_kv)
        self.split = max(self.BLOCK_N, -(-max_len // per) // self.BLOCK_N * self.BLOCK_N)
        self.ns = -(-max_len // self.split)
        self.bs = triton.next_power_of_2(self.ns)
        self._bufs = {}

    def bytes_per_token(self) -> float:
        per = {"bf16": 2.0 * self.D, "int8": self.D + 2.0 * self.D / self.GS, "int4": self.D / 2 + 2.0 * self.D / self.GS}[self.mode]
        return 2 * self.n * self.kv * per

    # ---- writes -------------------------------------------------------------------------------------
    def _quant(self, x):
        """x (T, KV, D) -> codes, scales in this store's layout."""
        T = x.shape[0]
        xf = x.float()
        if self.mode == "int4":
            xf = xf @ self.R
        b = xf.view(T, self.kv, self.D // self.GS, self.GS)
        amax = b.abs().amax(-1, keepdim=True)
        qmax = 127.0 if self.mode == "int8" else 7.0
        s = (amax / qmax).clamp_min(1e-12)
        c = torch.round(b / s).clamp(-qmax - (0 if self.mode == "int8" else 1), qmax).view(T, self.kv, self.D)
        s = s.squeeze(-1).to(torch.float16)
        if self.mode == "int8":
            return c.to(torch.int8), s
        c = (c + 8).to(torch.uint8)
        h = self.D // 2
        return c[..., :h] | (c[..., h:] << 4), s

    def write(self, j: int, pos_idx: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """k, v (T, KV, D), written at positions pos_idx (T,)."""
        if self.mode == "bf16":
            self.k[j].index_copy_(1, pos_idx, k.transpose(0, 1).to(torch.bfloat16))
            self.v[j].index_copy_(1, pos_idx, v.transpose(0, 1).to(torch.bfloat16))
            return
        kc, kss = self._quant(k)
        vc, vss = self._quant(v)
        self.k[j].index_copy_(1, pos_idx, kc.transpose(0, 1))
        self.v[j].index_copy_(1, pos_idx, vc.transpose(0, 1))
        self.ks[j].index_copy_(1, pos_idx, kss.transpose(0, 1))
        self.vs[j].index_copy_(1, pos_idx, vss.transpose(0, 1))

    # ---- reads --------------------------------------------------------------------------------------
    def _deq(self, c, s):
        """codes (KV, N, W), scales (KV, N, G) -> (KV, N, D) fp32 in the stored (possibly rotated) basis."""
        if self.mode == "int8":
            x = c.float()
        else:
            x = torch.cat([(c & 15).float() - 8, (c >> 4).float() - 8], -1)
        KV, N, D = x.shape
        return (x.view(KV, N, D // self.GS, self.GS) * s.float()[..., None]).view(KV, N, D)

    def dense(self, j: int, end: int):
        """BF16 K, V (KV, end, D) for positions < end (prefill / chunk attention)."""
        if self.mode == "bf16":
            return self.k[j, :, :end], self.v[j, :, :end]
        out = []
        for c, sc in ((self.k, self.ks), (self.v, self.vs)):
            o = torch.empty(self.kv, end, self.D, device=self.dev, dtype=torch.bfloat16)
            for a in range(0, end, 8192):                  # slices: fp32 temporaries stay small at 128k
                b = min(end, a + 8192)
                x = self._deq(c[j, :, a:b], sc[j, :, a:b])
                if self.mode == "int4":
                    x = x @ self.R.T
                o[:, a:b] = x.to(torch.bfloat16)
            out.append(o)
        return out[0], out[1]

    def attend(self, j: int, q: torch.Tensor, pos0: torch.Tensor, scale: float) -> torch.Tensor:
        """q (H, T, D) bf16 for rows at positions pos0 .. pos0 + T - 1 -> (H, T, D) bf16."""
        H, T, D = q.shape
        g = H // self.kv
        rows = g * T
        if self.mode == "int4":
            q = (q.float() @ self.R).to(torch.bfloat16)
        qr = q.reshape(self.kv, rows, D).contiguous()
        key = (rows,)
        if key not in self._bufs:
            self._bufs[key] = (torch.zeros(self.kv, self.ns, rows, D, device=self.dev, dtype=torch.float32),
                               torch.zeros(self.kv, self.ns, rows, device=self.dev, dtype=torch.float32),
                               torch.zeros(self.kv, self.ns, rows, device=self.dev, dtype=torch.float32),
                               torch.empty(self.kv, rows, D, device=self.dev, dtype=torch.bfloat16))
        O, MX, LS, OUT = self._bufs[key]
        K, V, KS, VS = self.k[j], self.v[j], self.ks[j], self.vs[j]
        R = 16 if rows <= 16 or D > 256 else 32           # all verify rows of a KV head in one program when they fit
        rb = -(-rows // R)
        _attn_split[(self.kv, self.ns, rb)](
            qr, K, V, KS, VS, pos0, O, MX, LS,
            T, g, rows, self.split, scale * LOG2E,
            qr.stride(0), qr.stride(1), K.stride(0), K.stride(1), KS.stride(0), KS.stride(1),
            O.stride(0), O.stride(1), O.stride(2), MX.stride(0), MX.stride(1),
            HALF=D // 2, ROWS=R, MODE=self.code, GS=self.GS)
        _attn_merge[(self.kv, rows)](O, MX, LS, OUT, self.ns, O.stride(0), O.stride(1), O.stride(2), MX.stride(0), MX.stride(1),
                                     OUT.stride(0), OUT.stride(1), D=D, BS=self.bs, CH=8, num_warps=4)
        out = OUT
        if self.mode == "int4":
            out = (out.float() @ self.R.T).to(torch.bfloat16)
        return out.view(H, T, D)
