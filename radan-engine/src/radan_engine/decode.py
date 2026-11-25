"""Minimal single-stream Qwen3 decoder for end-to-end decode timing of PLE (experiments E1, E5).

Runs on the weights of a loaded Transformers Qwen3 model, with production-style packed q/k/v and
gate/up matrices, a static KV cache and one decode step captured in a CUDA graph (position, mask
and next token are updated on the GPU, so replays run back to back with no host involvement).

Speculation of block i's attention inputs from block i-1 (source boundary i-1 in `spec`):
  overlap   side stream, started when block i-1 starts: ĥ = f(h_(i-1)), n̂ = RMSNorm_i(ĥ),
            qkv_i = n̂ · W_qkv,iᵀ (BF16, or an FP8 copy of W_qkv,i). Block i then skips its own
            norm + QKV GEMM on the critical path and uses qkv_i uncorrected.
  fused     identity predictor folded into block i-1's own QKV GEMM: block i-1 computes
            h / rms(h) · [W_qkv,i-1 diag(g_i-1) | W_qkv,i diag(g_i)]ᵀ in one GEMM. Source boundaries
            must not be consecutive (block i-1 must compute its own qkv).
"""

from dataclasses import dataclass, field

import torch
from radan_engine.kernels import q8_0_quantize, w8a16
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass
class SpecPlan:
    mode: str = "none"                    # none | overlap | fused
    sources: list[int] = field(default_factory=list)  # boundary l: block l+1 speculated from h_l
    predictor: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None  # l -> (A, b): ĥ = h + h A + b; None = identity
    fp8: bool = False                     # overlap: speculative QKV GEMM with FP8 weights; none: every block's
                                          # own QKV GEMM in FP8 (control for the FP8 kernel's own speed)


def _fp8_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 copy of w (out, in) and its FP32 scale. (Row-wise scaled FP8 GEMM is not
    supported on sm_120 in this PyTorch build, so weights and activations use one scale each.)"""
    scale = w.float().abs().amax().clamp(min=1e-12) / torch.finfo(torch.float8_e4m3fn).max
    return (w.float() / scale).to(torch.float8_e4m3fn), scale


def fp8_linear(x: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x (rows, in) BF16 · dequant(wq)ᵀ. Uses torch._scaled_mm with a per-tensor activation scale when
    available, else dequantises (still FP8 storage, for numerics)."""
    rows = x.shape[0]
    pad = (-rows) % 16
    xs = x.float().abs().amax().clamp(min=1e-12) / torch.finfo(torch.float8_e4m3fn).max
    xq = (x.float() / xs).to(torch.float8_e4m3fn)
    if pad:
        xq = torch.cat([xq, xq.new_zeros(pad, xq.shape[1])])
    out = torch._scaled_mm(xq, wq.t(), scale_a=xs.reshape(()), scale_b=scale.reshape(()), out_dtype=torch.bfloat16)
    return out[:rows]


CLIP_RATIOS = tuple(1 - 0.025 * i for i in range(13))  # 1.0 .. 0.7


def int4_group_params(wg: torch.Tensor, clip: bool = False):
    """wg (..., group) float -> per-group (scale, mn) for asymmetric 4-bit codes 0..15. clip: shrink
    the [min, max] range by the ratio in CLIP_RATIOS with the lowest squared rounding error."""
    mx, mn = wg.amax(-1), wg.amin(-1)
    if not clip:
        return (mx - mn).clamp(min=1e-8) / 15, mn
    best = torch.full_like(mx, float("inf"))
    bs, bm = (mx - mn).clamp(min=1e-8) / 15, mn
    for r in CLIP_RATIOS:
        hi, lo = mx * r, mn * r
        sc = (hi - lo).clamp(min=1e-8) / 15
        q = ((wg - lo[..., None]) / sc[..., None]).round().clamp(0, 15)
        err = (q * sc[..., None] + lo[..., None] - wg).pow(2).sum(-1)
        better = err < best
        best, bs, bm = torch.where(better, err, best), torch.where(better, sc, bs), torch.where(better, lo, bm)
    return bs, bm


# ---- fused helpers (follow-up F3a): pure functions compiled once; the caches are written outside them
def _f_rms(h, w, eps: float):
    return F.rms_norm(h, (h.shape[-1],), w, eps)


def _f_add_rms(h, a, w_next, eps: float):
    h2 = h + a
    return h2, F.rms_norm(h2, (h2.shape[-1],), w_next, eps)


def _f_silu_mul(gu):
    g, u = gu.chunk(2, dim=-1)
    return F.silu(g) * u


def _f_qk(qkv, wc, ws, nh: int, nkv: int, hd: int, eps: float, qk_norm: bool):
    """L1's fused q/k norm + RoPE (norm weight folded into the tables), and v, from the qkv projection."""
    T = qkv.shape[0]
    nq, nk = nh * hd, nkv * hd
    qk = qkv[:, :nq + nk].reshape(T, nh + nkv, hd)
    n = F.rms_norm(qk, (hd,), None, eps) if qk_norm else qk
    qk = torch.addcmul(n * wc, torch.roll(n, hd // 2, -1), ws)
    return qk[:, :nh].transpose(0, 1), qk[:, nh:].transpose(0, 1), qkv[:, nq + nk:].view(T, nkv, hd).transpose(0, 1)


_FUSED = {}


def fused(name):
    if name not in _FUSED:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _FUSED[name] = torch.compile(globals()[name], dynamic=False, options={"triton.cudagraphs": False})
    return _FUSED[name]


def compile_forward(dec):
    """torch.compile of dec.forward for fixed one-token shapes, without Inductor's own CUDA graphs (the
    caller captures the step in its own graph). Shared by every decoder class."""
    return torch.compile(dec.forward, dynamic=False, fullgraph=False, options={"triton.cudagraphs": False})


CODE_SINK = None  # a list: int4_pack_codes also appends (codes uint8, scale bf16, zero bf16, group) to it (follow-up J)


def int4_pack_codes(q: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, group: int, inner_k_tiles: int = 8):
    """Codes q (out, in) in 0..15 with per-group scale / min (out, in // group) -> tinygemm operands
    (packed, scales_and_zeros, group); the kernel dequantises as (q - 8) * scale + zero."""
    packed = torch.ops.aten._convert_weight_to_int4pack((q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8), inner_k_tiles)
    sz = torch.stack([scale, mn + scale * 8], -1).transpose(0, 1).contiguous().to(torch.bfloat16)
    if CODE_SINK is not None:
        CODE_SINK.append((q.to(torch.uint8).cpu(), sz[..., 0].t().contiguous().cpu(), sz[..., 1].t().contiguous().cpu(), group))
    return packed, sz, group


def int4_from_codes(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group: int, inner_k_tiles: int = 8):
    """tinygemm operands from saved codes (out, in) 0..15 and the bf16 scale / zero (out, in // group) it
    stored: the same dequantisation, (q - 8) * scale + zero."""
    q = codes.to(torch.int32).cuda()
    packed = torch.ops.aten._convert_weight_to_int4pack((q[:, ::2] << 4 | q[:, 1::2]).to(torch.uint8), inner_k_tiles)
    sz = torch.stack([scale.cuda(), zero.cuda()], -1).transpose(0, 1).contiguous().to(torch.bfloat16)
    return packed, sz, group


def int4_pack(w: torch.Tensor, group: int = 128, inner_k_tiles: int = 8, clip: bool = False):
    """Round-to-nearest INT4, one scale and zero per `group` inputs (optionally with a clipping search),
    packed for torch's tinygemm kernel (_weight_int4pack_mm)."""
    out_f, in_f = w.shape
    wg = w.float().view(out_f, in_f // group, group)
    scale, mn = int4_group_params(wg, clip)
    q = ((wg - mn[..., None]) / scale[..., None]).round().clamp(0, 15).to(torch.int32).view(out_f, in_f)
    return int4_pack_codes(q, scale, mn, group, inner_k_tiles)


def int4_pack_chunked(w: torch.Tensor, group: int = 128, clip: bool = False, rows: int = 16384):
    """int4_pack over row chunks (a 262k-row LM head with the clipping search needs several full-size
    fp32 temporaries otherwise); the same codes and operands as int4_pack, and one CODE_SINK entry."""
    global CODE_SINK
    sink, CODE_SINK = CODE_SINK, ([] if CODE_SINK is not None else None)
    parts = [int4_pack(w[r:r + rows], group, clip=clip) for r in range(0, w.shape[0], rows)]
    packed = torch.cat([p[0] for p in parts], 0)
    sz = torch.cat([p[1] for p in parts], 1)
    if sink is not None:
        c = CODE_SINK
        sink.append((torch.cat([x[0] for x in c]), torch.cat([x[1] for x in c]), torch.cat([x[2] for x in c]), c[0][3]))
    CODE_SINK = sink
    return packed, sz, group


def int4_gptq(w: torch.Tensor, H: torch.Tensor, group: int = 128, clip: bool = False, block: int = 128,
              damp: float = 0.01, inner_k_tiles: int = 8):
    """GPTQ (Frantar et al. 2022) to asymmetric INT4 with per-group scales: columns are rounded in
    order and each rounding error is spread over the not-yet-rounded columns through the inverse of
    H = sum of xᵀx over calibration inputs (in, in). Group parameters come from the updated weights."""
    W = w.float().clone()
    out_f, in_f = W.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.diag(H).mean() * torch.eye(in_f, device=H.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros(out_f, in_f, dtype=torch.int32, device=W.device)
    scales = torch.zeros(out_f, in_f // group, device=W.device)
    mins = torch.zeros_like(scales)
    for i1 in range(0, in_f, block):
        i2 = min(i1 + block, in_f)
        W1, Hi = W[:, i1:i2].clone(), Hinv[i1:i2, i1:i2]
        E = torch.zeros_like(W1)
        for i in range(i2 - i1):
            c = i1 + i
            if c % group == 0:
                sc, mn = int4_group_params(W1[:, i:i + group], clip)
                scales[:, c // group], mins[:, c // group] = sc, mn
            sc, mn = scales[:, c // group], mins[:, c // group]
            w_ = W1[:, i]
            q = ((w_ - mn) / sc).round().clamp(0, 15)
            Q[:, c] = q.to(torch.int32)
            err = (w_ - (q * sc + mn)) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None]
            E[:, i] = err
        W[:, i2:] -= E @ Hinv[i1:i2, i2:]
    return int4_pack_codes(Q, scales, mins, group, inner_k_tiles)


def int4_linear(x: torch.Tensor, packed) -> torch.Tensor:
    return torch.ops.aten._weight_int4pack_mm(x, packed[0], packed[2], packed[1])


def gqa_bmm_attention(q, k, v, mask):
    """q (1, H, T, D), k/v (1, KV, L, D), mask (1, 1, T, L) additive -> (1, H, T, D). Fixed shapes
    (graph-safe); faster than every SDPA backend here for 2-8 query rows (verify passes)."""
    _, H, T, D = q.shape
    KV = k.shape[1]
    g = H // KV
    s = torch.baddbmm(mask[0].expand(KV, T, -1).repeat(1, g, 1), q.reshape(KV, g * T, D), k[0].transpose(1, 2), alpha=D ** -0.5)
    return torch.bmm(torch.softmax(s.float(), -1).to(q.dtype), v[0]).view(1, H, T, D)


class Qwen3Decoder:
    def __init__(self, hf_model, max_len: int, attn: str = "sdpa", rope: str = "ref"):
        """attn: "sdpa" (default backend, masked, whole cache: the E1 decoder) or "fast" (cuDNN SDPA
        for 1-row steps, batched-matmul GQA for 2-16 rows, causal SDPA over the prompt only for
        prefill). rope: "ref" or "fused" (q and k normed and rotated together, norm weight folded
        into the rotation tables: 4 kernels per layer instead of ~14). "fast" and "fused" change only
        BF16 rounding order (docs/decode_levers.md, L1 and L2). Llama blocks (no q/k norm) are also
        supported: q and k are then only rotated."""
        self.attn_impl, self.rope_impl = attn, rope
        # one-row "fast" steps: "cudnn" (fastest; its split-KV reduction is not bitwise
        # deterministic run to run) or "bmm" (gqa_bmm_attention, deterministic)
        self.step_attn = "cudnn"
        cfg = hf_model.config
        self.d, self.nh, self.nkv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.qk_norm = hasattr(hf_model.model.layers[0].self_attn, "q_norm")
        self.eps = cfg.rms_norm_eps
        self.dev, self.dtype = hf_model.device, hf_model.dtype
        self.max_len = max_len
        m = hf_model.model
        self.embed = m.embed_tokens.weight
        self.final_norm = m.norm.weight
        self.lm_head = hf_model.lm_head.weight
        self.layers, self._hf_linears = [], [hf_model.lm_head, m.embed_tokens]
        with torch.no_grad():
            for layer in m.layers:
                a, mlp = layer.self_attn, layer.mlp
                self.layers.append({
                    "ln1": layer.input_layernorm.weight, "ln2": layer.post_attention_layernorm.weight,
                    "qkv": torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]).contiguous(),
                    "qn": a.q_norm.weight if self.qk_norm else torch.ones(self.hd, device=self.dev, dtype=self.dtype),
                    "kn": a.k_norm.weight if self.qk_norm else torch.ones(self.hd, device=self.dev, dtype=self.dtype),
                    "o": a.o_proj.weight,
                    "gu": torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight]).contiguous(), "down": mlp.down_proj.weight,
                })
                # point Transformers' projections at slices of the concatenated copies, so the weights are held
                # once (follow-up J: a 128k cache needs the ~7.5 GB the duplicates took); values are unchanged
                L, o = self.layers[-1], 0
                for mod in (a.q_proj, a.k_proj, a.v_proj):
                    n = mod.weight.shape[0]
                    mod.weight.data = L["qkv"][o:o + n]
                    o += n
                ni = mlp.gate_proj.weight.shape[0]
                mlp.gate_proj.weight.data, mlp.up_proj.weight.data = L["gu"][:ni], L["gu"][ni:]
                self._hf_linears += [a.q_proj, a.k_proj, a.v_proj, a.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj]
        rp = getattr(cfg, "rope_parameters", None) or {}
        self.theta = rp.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        self.set_max_len(max_len)
        qk_w = torch.stack([torch.cat([l["qn"].expand(self.nh, -1), l["kn"].expand(self.nkv, -1)]) for l in self.layers])
        self.qk_w = qk_w.contiguous()                                    # (layers, nh + nkv, hd)
        self.qk_w_roll = torch.roll(qk_w, self.hd // 2, -1).contiguous()
        self.w4 = None                                                   # INT4 copies (quantize_int4)
        self.draft_vocab = None      # (V,) token ids the INT4 head scores, or None for the full vocabulary (F2)
        self.q8 = None               # Q8_0 target weights (quantize_q8_target, follow-up K)
        self.fuse = False            # F3a fused helpers (compile_step)
        self.side = torch.cuda.Stream()
        self.plan = SpecPlan()
        self._fused, self._fp8 = {}, {}

    def set_max_len(self, max_len: int, yarn: tuple[float, int] | None = None):
        """(Re)allocate the KV cache and RoPE tables for max_len positions (follow-up I: one size per context,
        so the original decoder, which attends over its whole cache, is not slowed by unused slots).
        yarn = (factor, original_max_positions): YaRN RoPE scaling exactly as Transformers' _compute_yarn_parameters
        (beta_fast 32, beta_slow 1, truncate, attention factor 0.1 ln(factor) + 1 on cos / sin)."""
        import math
        self.max_len = max_len
        self.k_cache = self.v_cache = None
        torch.cuda.empty_cache()
        dim = self.hd
        pos_freqs = self.theta ** (torch.arange(0, dim, 2, device=self.dev, dtype=torch.float32) / dim)
        inv, af = 1.0 / pos_freqs, 1.0
        if yarn is not None:
            factor, orig = yarn
            corr = lambda rot: (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(self.theta))  # noqa: E731
            low, high = max(math.floor(corr(32)), 0), min(math.ceil(corr(1)), dim - 1)
            high = high + 0.001 if low == high else high
            ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32, device=self.dev) - low) / (high - low), 0, 1)
            extra = 1 - ramp
            inv = (1.0 / (factor * pos_freqs)) * (1 - extra) + (1.0 / pos_freqs) * extra
            af = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
        self.yarn = yarn
        ang = torch.arange(max_len, device=self.dev, dtype=torch.float32)[:, None] * inv[None]
        ang = torch.cat([ang, ang], dim=-1)
        self.cos, self.sin = (ang.cos() * af).to(self.dtype), (ang.sin() * af).to(self.dtype)
        self.k_cache = torch.zeros(len(self.layers), 1, self.nkv, max_len, self.hd, device=self.dev, dtype=self.dtype)
        self.v_cache = torch.zeros_like(self.k_cache)
        sign = torch.cat([-torch.ones(self.hd // 2), torch.ones(self.hd // 2)]).to(self.dev, self.dtype)
        self.sin_signed = self.sin * sign

    def prefill_chunked(self, ids: torch.Tensor, start: int = 0, chunk: int = 2048, logits_rows: int = 1, w4: bool = False):
        """Run ids at positions start .. start + T - 1 through the model in chunks (explicit causal mask per
        chunk, never a T x T mask), continuing the cache; start = 0 begins a new sequence. Returns the logits
        of the last `logits_rows` rows (follow-up I: long prompts, and teacher-forced checks as extensions)."""
        T = ids.shape[0]
        if start == 0:
            self.k_cache.zero_()
            self.v_cache.zero_()
        keep, ar = [], torch.arange(self.max_len, device=self.dev)
        for s0 in range(0, T, chunk):
            pos = torch.arange(start + s0, start + min(T, s0 + chunk), device=self.dev)
            mask = torch.where(ar[None, :] <= pos[:, None], 0.0, float("-inf")).to(self.dtype)[None, None]
            lg = self.forward(ids[s0:s0 + chunk], pos, mask, w4=w4)
            first_kept = T - logits_rows
            if s0 + lg.shape[0] > first_kept:
                keep.append(lg[max(0, first_kept - s0):])
            del lg, mask
        return torch.cat(keep)

    def compile_step(self, on: bool = True):
        """Follow-up F3a: forwards of up to 16 rows (decode steps, drafts, verify passes) use the fused
        (torch.compile'd) pure helpers; matmuls, cache writes and attention are unchanged. (Compiling the
        whole forward was slower: Inductor turned each in-place KV write into a copy of the whole cache.)"""
        self.fuse = on

    def snapshot(self):
        """State a decode step changes apart from cache slots past the prompt: none for this decoder."""
        return None

    def restore(self, snap):
        pass

    # ---- building blocks -------------------------------------------------------------------------
    def rms(self, x, w):
        return F.rms_norm(x, (x.shape[-1],), w, self.eps)

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def _rope_tables(self, pos_idx):
        """Fused RoPE tables: for small T, each layer's q/k norm weight folded into cos / signed sin
        for all layers at once (2 kernels per step); for large T just cos / signed sin."""
        cos, sin = self.cos[pos_idx], self.sin_signed[pos_idx]               # (T, hd)
        if pos_idx.shape[0] <= 16:
            return self.qk_w[:, None] * cos[None, :, None], self.qk_w_roll[:, None] * sin[None, :, None]
        return cos, sin

    def _attend(self, i, qkv, pos_idx, mask, tables=None, prefill=False, w4=False):
        """qkv (T, qkv_width) for positions pos_idx (T,), mask (1, 1, T, max_len) additive."""
        T = qkv.shape[0]
        L = self.layers[i]
        nq, nk = self.nh * self.hd, self.nkv * self.hd
        if self.fuse and self.rope_impl == "fused" and tables[0].dim() == 4:
            q, k, v = fused("_f_qk")(qkv, tables[0][i], tables[1][i], self.nh, self.nkv, self.hd, self.eps, self.qk_norm)
        elif self.rope_impl == "fused":
            qk = qkv[:, :nq + nk].reshape(T, self.nh + self.nkv, self.hd)
            n = F.rms_norm(qk, (self.hd,), None, self.eps) if self.qk_norm else qk
            wc, ws = tables
            if wc.dim() == 4:
                wc, ws = wc[i], ws[i]
            else:
                wc, ws = self.qk_w[i] * wc[:, None], self.qk_w_roll[i] * ws[:, None]
            qk = torch.addcmul(n * wc, torch.roll(n, self.hd // 2, -1), ws)
            q, k = qk[:, :self.nh].transpose(0, 1), qk[:, self.nh:].transpose(0, 1)
            v = qkv[:, nq + nk:].view(T, self.nkv, self.hd).transpose(0, 1)
        else:
            q, k, v = qkv.split([nq, nk, nk], dim=-1)
            q, k = q.view(T, self.nh, self.hd), k.view(T, self.nkv, self.hd)
            if self.qk_norm:
                q, k = self.rms(q, L["qn"]), self.rms(k, L["kn"])
            q, k = q.transpose(0, 1), k.transpose(0, 1)
            v = v.view(T, self.nkv, self.hd).transpose(0, 1)
            cos, sin = self.cos[pos_idx][None], self.sin[pos_idx][None]
            q = q * cos + self._rot(q) * sin
            k = k * cos + self._rot(k) * sin
        self.k_cache[i, 0].index_copy_(1, pos_idx, k)
        self.v_cache[i, 0].index_copy_(1, pos_idx, v)
        kc, vc = self.k_cache[i], self.v_cache[i]
        if self.attn_impl == "fast" and prefill:      # prompt fills positions 0..T-1: causal, no mask
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(q[None], kc[:, :, :T], vc[:, :, :T], is_causal=True, enable_gqa=True)
        elif self.attn_impl == "fast" and T == 1 and self.step_attn == "cudnn":
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(q[None], kc, vc, attn_mask=mask, enable_gqa=True)
        elif self.attn_impl == "fast" and T <= 16:
            out = gqa_bmm_attention(q[None], kc, vc, mask)
        else:
            out = F.scaled_dot_product_attention(q[None], kc, vc, attn_mask=mask, enable_gqa=True)
        out = out[0].transpose(0, 1).reshape(T, nq)
        return int4_linear(out, self.w4[i]["o"]) if w4 else self._lin(out, i, "o")

    def _mlp(self, i, x, w4=False):
        L = self.layers[i]
        if w4:
            if self.fuse:
                return int4_linear(fused("_f_silu_mul")(int4_linear(x, self.w4[i]["gu"])), self.w4[i]["down"])
            g, u = int4_linear(x, self.w4[i]["gu"]).chunk(2, dim=-1)
            return int4_linear(F.silu(g) * u, self.w4[i]["down"])
        if self.fuse:
            return self._lin(fused("_f_silu_mul")(self._lin(x, i, "gu")), i, "down")
        g, u = self._lin(x, i, "gu").chunk(2, dim=-1)
        return self._lin(F.silu(g) * u, i, "down")

    def _qkv(self, i, h, w4=False):
        x = self.rms(h, self.layers[i]["ln1"])
        return int4_linear(x, self.w4[i]["qkv"]) if w4 else self._lin(x, i, "qkv")

    def quantize_int4(self, group: int = 128, method: str = "rtn", clip: bool = False, head: bool = True,
                      hessians=None):
        """INT4 weight-only copies of every linear layer and (head=True) the LM head, used by
        forward(..., w4=True), e.g. for a quantized self-speculative draft. BF16 weights are kept.
        method "rtn" (round to nearest) or "gptq"; gptq needs hessians(i) -> {"qkv", "o", "gu",
        "down": H} for layer i and hessians("head") -> H. head=False keeps the LM head in BF16."""
        def q(w, H):
            return int4_gptq(w, H, group, clip) if method == "gptq" else int4_pack(w, group, clip=clip)
        global CODE_SINK
        CODE_SINK = []
        try:
            with torch.no_grad():
                self.w4 = []
                for i, l in enumerate(self.layers):
                    Hs = hessians(i) if method == "gptq" else {}
                    self.w4.append({k: q(l[k], Hs.get(k)) for k in ("qkv", "o", "gu", "down")})
                    del Hs
                self.w4_head = q(self.lm_head, hessians("head") if method == "gptq" else None) if head else None
            keys = [(i, k) for i in range(len(self.layers)) for k in ("qkv", "o", "gu", "down")] + ([("head", None)] if head else [])
            self.w4_codes = dict(zip(keys, CODE_SINK))   # codes, bf16 scale, bf16 zero, group: exactly what the kernel uses
        finally:
            CODE_SINK = None
        self._w4_head_args = (group, method, clip)

    # ---- Q8_0 target (follow-up K) -------------------------------------------------------------------
    def _lin(self, x, i, key):
        return x @ self.layers[i][key].t() if self.q8 is None else w8a16(x, *self.q8[i][key])

    def _q8_rows(self, ids):
        q, d = self.q8_head
        return (q[ids].view(ids.shape[0], -1, 32).float() * d[ids].float()[..., None]).view(ids.shape[0], -1).to(self.dtype)

    def quantize_q8_target(self, free_bf16: bool = True):
        """Make the target the Q8_0 model: every linear layer and the (tied) embedding / LM head quantised
        exactly as llama.cpp's Q8_0 (radan_engine.kernels.q8_0_quantize) and run with the W8A16 kernel. With
        free_bf16 the BF16 weights are released (Transformers' copies included), so a long cache fits."""
        with torch.no_grad():
            self.q8 = [{k: q8_0_quantize(l[k]) for k in ("qkv", "o", "gu", "down")} for l in self.layers]
            self.q8_head = q8_0_quantize(self.lm_head)
            if free_bf16:
                for l in self.layers:
                    for k in ("qkv", "o", "gu", "down"):
                        l[k] = None
                for mod in self._hf_linears:
                    mod.weight.data = torch.empty(0, device=self.dev, dtype=self.dtype)
                self.embed = self.lm_head = None
        torch.cuda.empty_cache()

    def save_int4_codes(self, path):
        torch.save({"codes": self.w4_codes, "args": self._w4_head_args}, path)

    def load_int4_codes(self, path):
        """INT4 copies rebuilt from saved codes (follow-up J: the same draft in this repo and in llama.cpp)."""
        d = torch.load(path)
        self.w4_codes, self._w4_head_args = d["codes"], d["args"]
        with torch.no_grad():
            self.w4 = [{k: int4_from_codes(*self.w4_codes[(i, k)]) for k in ("qkv", "o", "gu", "down")} for i in range(len(self.layers))]
            self.w4_head = int4_from_codes(*self.w4_codes[("head", None)]) if ("head", None) in self.w4_codes else None
        self.draft_vocab = None

    def quantize_int4_head(self, vocab=None, hessians=None):
        """Re-quantize only the INT4 LM head (same method as quantize_int4), over the token ids `vocab`
        (follow-up F2: a reduced-vocabulary draft head) or the full vocabulary (vocab=None)."""
        group, method, clip = self._w4_head_args
        w = self.lm_head if vocab is None else self.lm_head[vocab]
        with torch.no_grad():
            self.w4_head = int4_gptq(w, hessians("head"), group, clip) if method == "gptq" else int4_pack(w, group, clip=clip)
        self.draft_vocab = None if vocab is None else vocab.to(self.dev)
        torch.cuda.empty_cache()

    def greedy(self, logits, w4=False):
        """Greedy token ids from forward() logits; an INT4 head over draft_vocab is mapped back to full ids."""
        idx = logits.argmax(-1)
        return self.draft_vocab[idx] if w4 and self.draft_vocab is not None and self.w4_head is not None else idx

    def _spec_qkv(self, l, h):
        """Speculative qkv for block l+1 from h_l (overlap mode)."""
        p = self.plan
        h_hat = h if p.predictor is None else h + h @ p.predictor[l][0] + p.predictor[l][1]
        n_hat = self.rms(h_hat, self.layers[l + 1]["ln1"])
        if p.fp8:
            wq, s = self._fp8[l + 1]
            return fp8_linear(n_hat, wq, s)
        return n_hat @ self.layers[l + 1]["qkv"].t()

    # ---- configuration ---------------------------------------------------------------------------
    def set_plan(self, plan: SpecPlan):
        n = len(self.layers)
        assert all(0 <= l < n - 1 for l in plan.sources)
        if plan.mode == "fused":
            s = set(plan.sources)
            assert not any(l - 1 in s for l in s), "fused sources must not be consecutive"
            with torch.no_grad():
                for l in s:
                    a, b = self.layers[l], self.layers[l + 1]
                    self._fused[l] = torch.cat([a["qkv"].float() * a["ln1"].float()[None],
                                                b["qkv"].float() * b["ln1"].float()[None]]).to(self.dtype).contiguous()
        if plan.fp8:
            need = [l + 1 for l in plan.sources] if plan.mode == "overlap" else range(n)
            for i in need:
                if i not in self._fp8:
                    self._fp8[i] = _fp8_weight(self.layers[i]["qkv"])
        self.plan = plan

    # ---- forward -----------------------------------------------------------------------------------
    def forward(self, ids, pos_idx, mask, exact_first: bool = False, skip_attn=(), skip_mlp=(), w4: bool = False,
                prefill: bool = False):
        """ids (T,), returns logits (T, vocab). Applies the current speculation plan. Blocks in
        `skip_attn` / `skip_mlp` skip that sub-layer entirely (self-speculative drafting); a skipped
        attention writes no KV, which is fine because the full-model verify pass rewrites it."""
        p = self.plan
        src = set(p.sources) if p.mode != "none" else set()
        h = self.embed[ids] if self.q8 is None else self._q8_rows(ids)
        tables = self._rope_tables(pos_idx) if self.rope_impl == "fused" else None
        if self.fuse and p.mode == "none" and not p.fp8 and not skip_attn and not skip_mlp and ids.shape[0] <= 16:
            # F3a: residual add + next norm in one fused kernel; L1's q/k norm + RoPE and SiLU-mul fused
            e = self.eps
            x = fused("_f_rms")(h, self.layers[0]["ln1"], e)
            for i, L in enumerate(self.layers):
                qkv = int4_linear(x, self.w4[i]["qkv"]) if w4 else self._lin(x, i, "qkv")
                h, xm = fused("_f_add_rms")(h, self._attend(i, qkv, pos_idx, mask, tables, prefill, w4), L["ln2"], e)
                nxt = self.layers[i + 1]["ln1"] if i + 1 < len(self.layers) else self.final_norm
                h, x = fused("_f_add_rms")(h, self._mlp(i, xm, w4), nxt, e)
            if w4 and self.w4_head is not None:
                return int4_linear(x, self.w4_head)
            return x @ self.lm_head.t() if self.q8 is None else w8a16(x, *self.q8_head)
        pending = None  # speculative qkv for the current block
        for i in range(len(self.layers)):
            L = self.layers[i]
            main = torch.cuda.current_stream()
            spec_next, keep = None, None
            if p.mode == "overlap" and i in src:
                self.side.wait_stream(main)
                keep = h  # read on the side stream: hold it until the join so its memory is not reused
                with torch.cuda.stream(self.side):
                    spec_next = self._spec_qkv(i, h)
            if pending is not None:
                qkv = pending
                if exact_first:  # prefill: position 0 (attention sink) stays exact, as in the hook simulation
                    qkv = torch.cat([self._qkv(i, h[:1]), pending[1:]])
            elif p.mode == "fused" and i in src:
                x = F.rms_norm(h, (self.d,), None, self.eps)
                both = x @ self._fused[i].t()
                qkv, spec_next = both.chunk(2, dim=-1)
            elif i in skip_attn:
                qkv = None
            elif p.fp8 and p.mode == "none":
                qkv = fp8_linear(self.rms(h, L["ln1"]), *self._fp8[i])
            else:
                qkv = self._qkv(i, h, w4)
            if i not in skip_attn:
                h = h + self._attend(i, qkv, pos_idx, mask, tables, prefill, w4)
            if i not in skip_mlp:
                h = h + self._mlp(i, self.rms(h, L["ln2"]), w4)
            if p.mode == "overlap" and spec_next is not None:
                main.wait_stream(self.side)
                spec_next.record_stream(main)
            del keep
            pending = spec_next
        x = self.rms(h, self.final_norm)
        if w4 and self.w4_head is not None:
            return int4_linear(x, self.w4_head)
        return x @ self.lm_head.t() if self.q8 is None else w8a16(x, *self.q8_head)

    def prefill(self, ids: torch.Tensor, w4: bool = False) -> torch.Tensor:
        T = ids.shape[0]
        self.k_cache.zero_()
        self.v_cache.zero_()
        pos = torch.arange(T, device=self.dev)
        mask = torch.full((T, self.max_len), float("-inf"), device=self.dev, dtype=self.dtype)
        mask[:, :T] = torch.triu(torch.full((T, T), float("-inf"), device=self.dev, dtype=self.dtype), 1)
        return self.forward(ids, pos, mask[None, None], exact_first=True, prefill=True, w4=w4)

    # ---- graph-captured decode -------------------------------------------------------------------
    def make_step(self, prompt_len: int, first_token: int, w4: bool = False):
        """Static buffers + a CUDA-graph decode step: reads token/pos, writes logits and the greedy next
        token, advances pos and unmasks it. Returns (replay, buffers)."""
        buf = {"tok": torch.tensor([first_token], device=self.dev),
               "pos": torch.tensor([prompt_len], device=self.dev),
               "mask": torch.full((1, 1, 1, self.max_len), float("-inf"), device=self.dev, dtype=self.dtype)}
        buf["mask"][..., :prompt_len] = 0
        start = {k: v.clone() for k, v in buf.items()}

        fwd = getattr(self, "_step_forward", None) or self.forward

        def step():
            buf["mask"].index_fill_(-1, buf["pos"], 0.0)
            logits = fwd(buf["tok"], buf["pos"], buf["mask"], w4=w4)
            buf["logits"] = logits
            buf["tok"].copy_(self.greedy(logits, w4))
            buf["pos"].add_(1)

        def reset():
            for k in ("tok", "pos", "mask"):
                buf[k].copy_(start[k])

        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                reset()
                step()
        torch.cuda.current_stream().wait_stream(warm)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        reset()
        return graph.replay, reset, buf
