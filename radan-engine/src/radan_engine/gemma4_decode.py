"""Gemma 4 (dense, text) decoder for the decode-lever harness (docs/decode_levers.md, follow-up E).

Same interface as radan_engine.decode.Qwen3Decoder (prefill, forward, make_step, quantize_int4, attn_impl /
rope_impl / step_attn), so lever_bench.py, self_spec.py and the reference benchmarks run unchanged.
It reimplements Transformers' Gemma4Unified text model:

  h = embed[ids] * sqrt(d)                                (in BF16, as Transformers)
  per layer:  x = norm(h, ln1);  q, k, v = x W_q, x W_k, x W_v   (full layers: v = k before its norm)
              q = rope(norm(q, q_norm)), k = rope(norm(k, k_norm)), v = norm(v)  (v norm has no weight)
              h = h + norm(attn(q, k, v; scale 1) W_o, post_attn)
              h = h + norm(down(gelu_tanh(x W_g) * x W_u), post_ff),  x = norm(h, pre_ff)
              h = h * layer_scalar
  logits = 30 * tanh((norm(h) E^T) / 30)

Sliding layers (head 256, 8 KV heads) use RoPE base 1e4 over the whole head; full layers (head 512,
one KV head with K = V) use "proportional" RoPE base 1e6 on a quarter of the head. The harness never
exceeds the 1,024-token sliding window, so every layer attends to the whole context (asserted).

attn "sdpa" / rope "ref" follow Transformers' operation order (the original decoder); "fast" and
"fused" are the L2 and L1 levers. cuDNN SDPA on this build takes heads of at most 128, so with
Gemma 4 12B's heads of 256 and 512 every one-row "fast" step uses batched matmuls (L2 = L2d here).
"""

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from radan_engine.decode import SpecPlan, int4_linear, int4_pack, int4_pack_chunked
from radan_engine.kernels import q8_0_quantize, w8a16


# ---- fused helpers (follow-up F3a): pure functions only, so torch.compile fuses them without touching the
# caches (compiling the whole step turned every in-place KV write into a copy of the whole cache)
def _norm_ref(x, w, eps):
    xf = x.float()
    y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if w is not None:
        y = y * w.float()
    return y.type_as(x)


def _rot(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _f_norm(x, w, eps):
    return _norm_ref(x, w, eps)


def _f_add_norm(h, a, w_post, w_next, eps):
    """h + norm(a, w_post), and norm of that with w_next (the next sub-layer's input)."""
    h2 = h + _norm_ref(a, w_post, eps)
    return h2, _norm_ref(h2, w_next, eps)


def _f_add_norm_scale(h, a, w_post, scalar, eps):
    return (h + _norm_ref(a, w_post, eps)) * scalar


def _f_qkv(q, k, v, qn, kn, cos, sin, eps):
    q = _norm_ref(q, qn, eps)
    k = _norm_ref(k, kn, eps)
    q = (q * cos + _rot(q) * sin).transpose(0, 1)
    k = (k * cos + _rot(k) * sin).transpose(0, 1)
    return q, k, _norm_ref(v, None, eps).transpose(0, 1)


def _f_gelu_mul(g, u):
    return F.gelu(g, approximate="tanh") * u


def _f_softcap(logits, cap):
    return torch.tanh(logits / cap) * cap


_FUSED = {}


def fused(name):
    """The compiled version of helper `name` (compiled once per process; torch.compile specialises shapes)."""
    if name not in _FUSED:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _FUSED[name] = torch.compile(globals()[name], dynamic=False, options={"triton.cudagraphs": False})
    return _FUSED[name]


def bmm_attention(q, k, v, mask, scale):
    """q (1, H, T, D), k/v (1, KV, L, D), mask (1, 1, T, L) additive -> (1, H, T, D)."""
    _, H, T, D = q.shape
    KV = k.shape[1]
    g = H // KV
    s = torch.baddbmm(mask[0].expand(KV, T, -1).repeat(1, g, 1), q.reshape(KV, g * T, D), k[0].transpose(1, 2), alpha=scale)
    return torch.bmm(torch.softmax(s.float(), -1).to(q.dtype), v[0]).view(1, H, T, D)


def is_gemma4(hf_model) -> bool:
    return getattr(hf_model.config, "model_type", "").startswith("gemma4")


class Gemma4Decoder:
    def __init__(self, hf_model, max_len: int, attn: str = "sdpa", rope: str = "ref"):
        self.attn_impl, self.rope_impl, self.step_attn = attn, rope, "cudnn"
        top = hf_model.config
        cfg = getattr(top, "text_config", top)
        lm = hf_model.model.language_model if hasattr(hf_model.model, "language_model") else hf_model.model
        assert not getattr(cfg, "enable_moe_block", False), "mixture-of-experts Gemma 4 is not implemented"
        assert getattr(cfg, "num_kv_shared_layers", 0) == 0, "KV-shared layers are not implemented"
        assert getattr(cfg, "hidden_size_per_layer_input", 0) in (0, None), "per-layer inputs are not implemented"
        self.cfg = cfg
        self.d, self.nh, self.eps = cfg.hidden_size, cfg.num_attention_heads, cfg.rms_norm_eps
        self.softcap = cfg.final_logit_softcapping
        self.dev, self.dtype = hf_model.device, hf_model.dtype
        self.window = cfg.sliding_window
        self.embed = lm.embed_tokens.weight
        self.embed_scale = torch.tensor(cfg.hidden_size ** 0.5).to(self.dtype).to(self.dev)
        self.final_norm = lm.norm.weight
        self.lm_head = hf_model.lm_head.weight
        self.types = [("full" if t == "full_attention" else "sliding") for t in cfg.layer_types]
        # head size and KV heads differ by layer type (the config marks them per layer): read them from the weights
        self.geom = {}
        for t, layer in zip(self.types, lm.layers):
            if t not in self.geom:
                hd = layer.self_attn.q_norm.weight.shape[0]
                self.geom[t] = (hd, layer.self_attn.k_proj.weight.shape[0] // hd)
        self.layers, self.slot = [], []
        count = {"sliding": 0, "full": 0}
        with torch.no_grad():
            for i, layer in enumerate(lm.layers):
                t = self.types[i]
                hd, nkv = self.geom[t]
                a, mlp = layer.self_attn, layer.mlp
                kv_eq = a.v_proj is None
                # separate projections, as Transformers runs them: this model is sensitive enough at some
                # positions that one concatenated q/k/v GEMM's different rounding changes the logits
                # (follow-up E, gate diagnosis); the INT4 copies are concatenated, as for Qwen3
                self.layers.append({"type": t, "hd": hd, "nkv": nkv, "kv_eq": kv_eq,
                                    "ln1": layer.input_layernorm.weight, "post_attn": layer.post_attention_layernorm.weight,
                                    "pre_ff": layer.pre_feedforward_layernorm.weight, "post_ff": layer.post_feedforward_layernorm.weight,
                                    "qn": a.q_norm.weight, "kn": a.k_norm.weight,
                                    "q": a.q_proj.weight, "k": a.k_proj.weight, "v": None if kv_eq else a.v_proj.weight,
                                    "o": a.o_proj.weight, "g": mlp.gate_proj.weight, "u": mlp.up_proj.weight,
                                    "down": mlp.down_proj.weight, "scalar": layer.layer_scalar})
                self.slot.append(count[t])
                count[t] += 1
        self.count = count
        # fused-RoPE weights: per type, (layers of that type, nh + nkv, hd), q/k norm weight folded in
        self.qk_w, self.qk_w_roll = {}, {}
        for t in ("sliding", "full"):
            hd, nkv = self.geom[t]
            ws = [torch.cat([l["qn"].expand(self.nh, -1), l["kn"].expand(nkv, -1)]) for l in self.layers if l["type"] == t]
            w = torch.stack(ws).contiguous()
            self.qk_w[t], self.qk_w_roll[t] = w, torch.roll(w, hd // 2, -1).contiguous()
        self.k_cache, self.v_cache = {}, {}
        self.set_max_len(max_len)
        self.w4, self.w4_head = None, None
        self.q8 = None
        self.fuse = False            # F3a: fused helpers (set by compile_step)
        self.plan = SpecPlan()

    def set_max_len(self, max_len: int):
        """(Re)allocate caches and RoPE tables. Up to the 1,024-token window every layer keeps max_len slots
        (the gate-checked path); beyond it the sliding layers keep a ring of `window` slots, slot = pos % window,
        with ring_pos holding the position in each slot (follow-up I)."""
        cfg = self.cfg
        self.max_len = max_len
        self.ring = max_len > self.window
        self.k_cache, self.v_cache = {}, {}
        torch.cuda.empty_cache()
        # RoPE tables per layer type (Transformers: fp32 angles, cos/sin cast to BF16)
        self.cos, self.sin, self.sin_signed = {}, {}, {}
        pos = torch.arange(max_len, device=self.dev, dtype=torch.float32)
        for t, lt in (("sliding", "sliding_attention"), ("full", "full_attention")):
            hd = self.geom[t][0]
            rp = cfg.rope_parameters[lt]
            base = rp["rope_theta"]
            if rp.get("rope_type", "default") == "proportional":
                ang = int(rp.get("partial_rotary_factor", 1.0) * hd // 2)
                inv = 1.0 / (base ** (torch.arange(0, 2 * ang, 2, dtype=torch.int64, device=self.dev).float() / hd))
                inv = torch.cat([inv, torch.zeros(hd // 2 - ang, device=self.dev)])
            else:
                inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.int64, device=self.dev).float() / hd))
            emb = torch.cat([pos[:, None] * inv[None]] * 2, dim=-1)
            self.cos[t], self.sin[t] = emb.cos().to(self.dtype), emb.sin().to(self.dtype)
            sign = torch.cat([-torch.ones(hd // 2), torch.ones(hd // 2)]).to(self.dev, self.dtype)
            self.sin_signed[t] = self.sin[t] * sign
        for t in ("sliding", "full"):
            hd, nkv = self.geom[t]
            slots = self.window if (t == "sliding" and self.ring) else max_len
            self.k_cache[t] = torch.zeros(self.count[t], 1, nkv, slots, hd, device=self.dev, dtype=self.dtype)
            self.v_cache[t] = torch.zeros_like(self.k_cache[t])
        self.ring_pos = torch.full((self.window,), -1, dtype=torch.long, device=self.dev)
        self._ring_mask = None

    def _ring_prepare(self, pos_idx):
        """Once per forward with a ring: the sliding layers' additive mask. One row: the slot is written
        first (its old position is outside the window), then the ring is attended. Several rows: they attend
        the old ring and the new rows together (mask over W + T keys); the ring is written after the layers."""
        W = self.window
        p = pos_idx[:, None]
        if pos_idx.shape[0] == 1:
            self.ring_pos.index_copy_(0, pos_idx % W, pos_idx)
            keys = self.ring_pos[None]
        else:
            keys = torch.cat([self.ring_pos, pos_idx])[None]
        ok = (keys >= 0) & (keys <= p) & (keys > p - W)
        self._ring_mask = torch.where(ok, 0.0, float("-inf")).to(self.dtype)[None, None]

    def snapshot(self):
        """With a ring, decode steps overwrite ring slots: keep them; otherwise nothing past the prompt changes."""
        if not self.ring:
            return None
        return self.k_cache["sliding"].clone(), self.v_cache["sliding"].clone(), self.ring_pos.clone()

    def restore(self, snap):
        if snap is not None:
            self.k_cache["sliding"].copy_(snap[0])
            self.v_cache["sliding"].copy_(snap[1])
            self.ring_pos.copy_(snap[2])

    def prefill_chunked(self, ids: torch.Tensor, start: int = 0, chunk: int = 1024, logits_rows: int = 1, w4: bool = False):
        """As Qwen3Decoder.prefill_chunked; chunks of at most `window` rows so a chunk never overwrites ring
        slots it still needs."""
        chunk = min(chunk, self.window)
        T = ids.shape[0]
        if start == 0:
            for t in ("sliding", "full"):
                self.k_cache[t].zero_()
                self.v_cache[t].zero_()
            self.ring_pos.fill_(-1)
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

    # ---- building blocks -------------------------------------------------------------------------
    def norm_ref(self, x, w):
        """Transformers' Gemma4UnifiedRMSNorm: fp32, torch.pow(mean + eps, -0.5), weight in fp32."""
        xf = x.float()
        y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if w is not None:
            y = y * w.float()
        return y.type_as(x)

    def norm(self, x, w):
        """Hidden-state norms stay as in Transformers for every lever (L1 changes only q/k/v norm + RoPE)."""
        return self.norm_ref(x, w)

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def _rope_tables(self, pos_idx):
        out = {}
        for t in ("sliding", "full"):
            cos, sin = self.cos[t][pos_idx], self.sin_signed[t][pos_idx]          # (T, hd)
            out[t] = (self.qk_w[t][:, None] * cos[None, :, None], self.qk_w_roll[t][:, None] * sin[None, :, None])
        return out

    def _proj_qkv(self, i, x, w4):
        """Raw q, k, v (T, width) for layer i; full layers have v = k (before its norm)."""
        L = self.layers[i]
        hd, nkv = L["hd"], L["nkv"]
        nq, nk = self.nh * hd, nkv * hd
        if w4:
            qkv = int4_linear(x, self.w4[i]["qkv"])
            q, k = qkv[:, :nq], qkv[:, nq:nq + nk]
            return q, k, (k if L["kv_eq"] else qkv[:, nq + nk:])
        q, k = self._lin(x, i, "q"), self._lin(x, i, "k")
        return q, k, (k if L["kv_eq"] else self._lin(x, i, "v"))

    def _attend(self, i, qkv, pos_idx, mask, tables, prefill, w4):
        q_raw, k_raw, v_raw = qkv
        T = q_raw.shape[0]
        L = self.layers[i]
        t, hd, nkv, j = L["type"], L["hd"], L["nkv"], self.slot[i]
        nq = self.nh * hd
        v_raw = v_raw.reshape(T, nkv, hd)
        if self.fuse:
            cos, sin = self.cos[t][pos_idx][:, None], self.sin[t][pos_idx][:, None]
            q, k, v = fused("_f_qkv")(q_raw.reshape(T, self.nh, hd), k_raw.reshape(T, nkv, hd), v_raw, L["qn"], L["kn"], cos, sin, self.eps)
        elif self.rope_impl == "fused":
            qk = torch.cat([q_raw, k_raw], -1).reshape(T, self.nh + nkv, hd)
            n = F.rms_norm(qk, (hd,), None, self.eps)
            wc, ws = tables[t]
            qk = torch.addcmul(n * wc[j], torch.roll(n, hd // 2, -1), ws[j])
            q, k = qk[:, :self.nh].transpose(0, 1), qk[:, self.nh:].transpose(0, 1)
            v = F.rms_norm(v_raw, (hd,), None, self.eps).transpose(0, 1)
        else:
            q = self.norm_ref(q_raw.reshape(T, self.nh, hd), L["qn"])
            k = self.norm_ref(k_raw.reshape(T, nkv, hd), L["kn"])
            cos, sin = self.cos[t][pos_idx][:, None], self.sin[t][pos_idx][:, None]
            q = (q * cos + self._rot(q) * sin).transpose(0, 1)
            k = (k * cos + self._rot(k) * sin).transpose(0, 1)
            v = self.norm_ref(v_raw, None).transpose(0, 1)
        kc, vc = self.k_cache[t][j], self.v_cache[t][j]
        if t == "sliding" and self.ring:
            W, m = self.window, self._ring_mask
            if T == 1:
                kc[0].index_copy_(1, pos_idx % W, k)
                vc[0].index_copy_(1, pos_idx % W, v)
                keys, vals = kc, vc
            else:  # attend the old ring and these rows together, then write the rows (chunk <= window)
                keys, vals = torch.cat([kc, k[None]], 2), torch.cat([vc, v[None]], 2)
            if self.attn_impl == "fast" and T <= 16:
                out = bmm_attention(q[None], keys, vals, m, 1.0)
            else:
                out = F.scaled_dot_product_attention(q[None], keys, vals, attn_mask=m, enable_gqa=True, scale=1.0)
            if T > 1:
                kc[0].index_copy_(1, pos_idx % W, k)
                vc[0].index_copy_(1, pos_idx % W, v)
            out = out[0].transpose(0, 1).reshape(T, nq)
            return int4_linear(out, self.w4[i]["o"]) if w4 else self._lin(out, i, "o")
        kc[0].index_copy_(1, pos_idx, k)
        vc[0].index_copy_(1, pos_idx, v)
        if self.attn_impl == "fast" and prefill:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q[None], kc[:, :, :T], vc[:, :, :T], is_causal=True, enable_gqa=True, scale=1.0)
        elif self.attn_impl == "fast" and T == 1 and self.step_attn == "cudnn" and hd <= 128:  # cuDNN SDPA here: head <= 128 only
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                out = F.scaled_dot_product_attention(q[None], kc, vc, attn_mask=mask, enable_gqa=True, scale=1.0)
        elif self.attn_impl == "fast" and T <= 16:
            out = bmm_attention(q[None], kc, vc, mask, 1.0)
        else:
            out = F.scaled_dot_product_attention(q[None], kc, vc, attn_mask=mask, enable_gqa=True, scale=1.0)
        out = out[0].transpose(0, 1).reshape(T, nq)
        return int4_linear(out, self.w4[i]["o"]) if w4 else self._lin(out, i, "o")

    def _mlp(self, i, x, w4):
        L = self.layers[i]
        if w4:
            g, u = int4_linear(x, self.w4[i]["gu"]).chunk(2, dim=-1)
            return int4_linear(F.gelu(g, approximate="tanh") * u, self.w4[i]["down"])
        if self.fuse:
            return self._lin(fused("_f_gelu_mul")(self._lin(x, i, "g"), self._lin(x, i, "u")), i, "down")
        return self._lin(F.gelu(self._lin(x, i, "g"), approximate="tanh") * self._lin(x, i, "u"), i, "down")

    # ---- Q8_0 target (follow-up K) -------------------------------------------------------------------
    def _lin(self, x, i, key):
        return F.linear(x, self.layers[i][key]) if self.q8 is None else w8a16(x, *self.q8[i][key])

    def _q8_rows(self, ids):
        q, d = self.q8_head
        return (q[ids].view(ids.shape[0], -1, 32).float() * d[ids].float()[..., None]).view(ids.shape[0], -1).to(self.dtype)

    def quantize_q8_target(self, hf_model, draft_group: int = 32, draft_clip: bool = True, codes_path=None, draft: bool = True):
        """Layer by layer (the BF16 weights, an 8-bit copy and a 4-bit copy do not fit together): the Q8_0
        target (llama.cpp's formula, radan_engine.kernels.q8_0_quantize), an INT4 draft copy (round to nearest + the
        clipping search, groups of draft_group, LM head included; its exact codes saved to codes_path for
        llama.cpp's Q4_1), then the layer's BF16 weights are freed (Transformers' copies included).
        draft=False: the Q8_0 target only (no INT4 copy: about 6.5 GB more room for the KV cache)."""
        import radan_engine.decode as pd
        lm = hf_model.model.language_model if hasattr(hf_model.model, "language_model") else hf_model.model
        pd.CODE_SINK = []
        self.q8, self.w4 = [], []
        try:
            with torch.no_grad():
                for i, (l, layer) in enumerate(zip(self.layers, lm.layers)):
                    keys = ["q", "k"] + ([] if l["kv_eq"] else ["v"]) + ["o", "g", "u", "down"]
                    self.q8.append({k: q8_0_quantize(l[k]) for k in keys})
                    if draft:
                        qkv = torch.cat([l["q"], l["k"]] + ([] if l["kv_eq"] else [l["v"]]))
                        self.w4.append({"qkv": int4_pack(qkv, draft_group, clip=draft_clip), "o": int4_pack(l["o"], draft_group, clip=draft_clip),
                                        "gu": int4_pack(torch.cat([l["g"], l["u"]]), draft_group, clip=draft_clip),
                                        "down": int4_pack(l["down"], draft_group, clip=draft_clip)})
                        del qkv
                    a, mlp = layer.self_attn, layer.mlp
                    for k in keys:
                        l[k] = None
                    for mod in [a.q_proj, a.k_proj, a.o_proj, mlp.gate_proj, mlp.up_proj, mlp.down_proj] + ([] if a.v_proj is None else [a.v_proj]):
                        mod.weight.data = torch.empty(0, device=self.dev, dtype=self.dtype)
                    torch.cuda.empty_cache()
                if draft:
                    self.w4_head = int4_pack_chunked(self.lm_head, draft_group, clip=draft_clip)
                else:
                    self.w4 = None
                self.q8_head = q8_0_quantize(self.lm_head)
                keys = [(i, k) for i in range(len(self.layers)) for k in ("qkv", "o", "gu", "down")] + [("head", None)]
                codes = dict(zip(keys, pd.CODE_SINK))
                if codes_path and draft:
                    torch.save({"codes": codes, "args": (draft_group, "rtn", draft_clip)}, codes_path)
                lm.embed_tokens.weight.data = torch.empty(0, device=self.dev, dtype=self.dtype)
                hf_model.lm_head.weight.data = torch.empty(0, device=self.dev, dtype=self.dtype)
                self.embed = self.lm_head = None
        finally:
            pd.CODE_SINK = None
        torch.cuda.empty_cache()

    def quantize_int4(self, group: int = 128, method: str = "rtn", clip: bool = False, head: bool = True, hessians=None):
        """INT4 round-to-nearest copies of every linear layer and (head=True) the tied LM head."""
        assert method == "rtn", "GPTQ statistics are only collected for Qwen3 / Llama blocks"
        with torch.no_grad():
            self.w4 = []
            for l in self.layers:
                qkv = torch.cat([l["q"], l["k"]] + ([] if l["kv_eq"] else [l["v"]]))
                self.w4.append({"qkv": int4_pack(qkv, group, clip=clip), "o": int4_pack(l["o"], group, clip=clip),
                                "gu": int4_pack(torch.cat([l["g"], l["u"]]), group, clip=clip),
                                "down": int4_pack(l["down"], group, clip=clip)})
                del qkv
            self.w4_head = int4_pack(self.lm_head, group, clip=clip) if head else None
        torch.cuda.empty_cache()

    def _head(self, x, w4):
        if w4 and self.w4_head is not None:
            logits = int4_linear(x, self.w4_head)
        else:
            logits = F.linear(x, self.lm_head) if self.q8 is None else w8a16(x, *self.q8_head)
        if self.softcap is not None:
            logits = fused("_f_softcap")(logits, self.softcap) if self.fuse else torch.tanh(logits / self.softcap) * self.softcap
        return logits

    # ---- forward -----------------------------------------------------------------------------------
    def forward(self, ids, pos_idx, mask, exact_first: bool = False, skip_attn=(), skip_mlp=(), w4: bool = False,
                prefill: bool = False):
        assert self.plan.mode == "none", "layer speculation plans are only implemented for Qwen3 / Llama blocks"
        h = (self.embed[ids] if self.q8 is None else self._q8_rows(ids)) * self.embed_scale
        tables = self._rope_tables(pos_idx) if self.rope_impl == "fused" else None
        if self.ring:
            self._ring_prepare(pos_idx)
        if self.fuse and not skip_attn and not skip_mlp:
            # F3a: the same arithmetic with each norm (and its residual add / layer scalar) one fused kernel
            e = self.eps
            x = fused("_f_norm")(h, self.layers[0]["ln1"], e)
            for i, L in enumerate(self.layers):
                a = self._attend(i, self._proj_qkv(i, x, w4), pos_idx, mask, tables, prefill, w4)
                h, xm = fused("_f_add_norm")(h, a, L["post_attn"], L["pre_ff"], e)
                h = fused("_f_add_norm_scale")(h, self._mlp(i, xm, w4), L["post_ff"], L["scalar"], e)
                x = fused("_f_norm")(h, self.layers[i + 1]["ln1"] if i + 1 < len(self.layers) else self.final_norm, e)
            if self.ring and ids.shape[0] > 1:
                self.ring_pos.index_copy_(0, pos_idx % self.window, pos_idx)
            return self._head(x, w4)
        for i, L in enumerate(self.layers):
            if i not in skip_attn:
                x = self.norm(h, L["ln1"])
                qkv = self._proj_qkv(i, x, w4)
                h = h + self.norm(self._attend(i, qkv, pos_idx, mask, tables, prefill, w4), L["post_attn"])
            if i not in skip_mlp:
                h = h + self.norm(self._mlp(i, self.norm(h, L["pre_ff"]), w4), L["post_ff"])
            h = h * L["scalar"]
        if self.ring and ids.shape[0] > 1:
            self.ring_pos.index_copy_(0, pos_idx % self.window, pos_idx)
        return self._head(self.norm(h, self.final_norm), w4)

    def prefill(self, ids: torch.Tensor, w4: bool = False) -> torch.Tensor:
        T = ids.shape[0]
        if self.ring:
            return self.prefill_chunked(ids, logits_rows=T, w4=w4)
        for t in ("sliding", "full"):
            self.k_cache[t].zero_()
            self.v_cache[t].zero_()
        pos = torch.arange(T, device=self.dev)
        mask = torch.full((T, self.max_len), float("-inf"), device=self.dev, dtype=self.dtype)
        mask[:, :T] = torch.triu(torch.full((T, T), float("-inf"), device=self.dev, dtype=self.dtype), 1)
        return self.forward(ids, pos, mask[None, None], prefill=True, w4=w4)

    def greedy(self, logits, w4=False):
        return logits.argmax(-1)

    def compile_step(self, on: bool = True):
        """Follow-up F3a: every forward uses the fused (torch.compile'd) pure helpers for norms, residual adds,
        q/k/v norm + RoPE, the MLP activation and the softcap; matmuls, cache writes and attention unchanged."""
        self.fuse = on

    def make_step(self, prompt_len: int, first_token: int, w4: bool = False):
        from radan_engine.decode import Qwen3Decoder
        return Qwen3Decoder.make_step(self, prompt_len, first_token, w4)


def make_decoder(hf_model, max_len: int, **kw):
    """Qwen3Decoder for Qwen3 / Llama blocks, Gemma4Decoder for dense Gemma 4, Qwen35Decoder for Qwen3.5 / 3.8."""
    if is_gemma4(hf_model):
        return Gemma4Decoder(hf_model, max_len, **kw)
    from radan_engine.qwen35_decode import Qwen35Decoder, is_qwen35
    if is_qwen35(hf_model):
        return Qwen35Decoder(hf_model, max_len, **kw)
    from radan_engine.decode import Qwen3Decoder
    return Qwen3Decoder(hf_model, max_len, **kw)
