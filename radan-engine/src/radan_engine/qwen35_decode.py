"""Qwen3.5-architecture decoder (Qwen3.8-27B, INT4 W4A16) for the decode-lever harness
(docs/decode_levers.md, follow-up G).

Same interface as radan_engine.decode.Qwen3Decoder (prefill, forward, make_step, greedy, attn_impl / rope_impl /
step_attn), so lever_bench.py and the reference benchmarks run unchanged. It reimplements Transformers'
Qwen3_5 text model, with the checkpoint's symmetric INT4 group-128 weights repacked for torch's
tinygemm kernel (codes + 8, zero 0, so each weight dequantises to the checkpoint's q * scale):

  Gated DeltaNet layers (48 of 64):
      mixed = x W_qkv, z = x W_z, b = x W_b, a = x W_a        (W_b, W_a in BF16)
      mixed = silu(depthwise causal conv1d(mixed), kernel 4)  -> q, k (16 heads x 128), v (48 x 128)
      beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
      per value head (q, k shared by groups of 3): S = S * exp(g); S += k (beta (v - S^T k))^T;
      o = S^T (l2norm(q) / sqrt(128))  with l2norm(k)        (fp32 state)
      out = (norm(o) * w_norm) * silu(z), then W_out
  Gated attention layers (16 of 64): q and a sigmoid output gate from W_q, zero-centred q/k norms,
      RoPE on the first 64 of 256 dims (text positions make the multi-axis RoPE plain RoPE), GQA 24 / 4
  MLP: SwiGLU. Norms: zero-centred RMSNorm, x * (1 + w). LM head and embeddings in BF16.

attn "sdpa" / rope "ref" follow Transformers' operation order (the original decoder). Prefill uses
Transformers' own chunked delta rule and causal conv, so the recurrent and conv states after the
prompt are the ones Transformers would hold. Steps of more than one row after prefill (a self-spec
verify pass) are not implemented yet.
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from radan_engine.decode import SpecPlan, int4_linear, int4_pack_codes
from radan_engine.gemma4_decode import bmm_attention


# ---- fused helpers (follow-up F3a): pure functions, compiled once; caches and states are written outside
def _zc_norm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).type_as(x)


def _f_norm(x, w, eps):
    return _zc_norm(x, w, eps)


def _f_add_norm(h, a, w_next, eps):
    h2 = h + a
    return h2, _zc_norm(h2, w_next, eps)


def _f_silu_mul(g, u):
    return F.silu(g) * u


def _f_delta_step(conv_state, mixed, conv_w, S, b, a, A_log, dt_bias, z, norm_w, eps: float, lk: int, dk: int, lv: int, dv: int):
    """One-token Gated DeltaNet step (as Qwen35Decoder._delta for T = 1), conv as a 4-term sum:
    returns the new conv state, the new recurrent state and the gated-norm output (1, lv * dv)."""
    full = torch.cat([conv_state[0], mixed.t()], -1).to(conv_w.dtype)                 # (C, 4)
    conv = F.silu((full.float() * conv_w.float()).sum(-1)).to(mixed.dtype)              # (C,)
    q, k, v = conv.split([lk * dk, lk * dk, lv * dv])
    r = lv // lk
    qa = q.view(lk, dk).repeat_interleave(r, 0).float()
    ka = k.view(lk, dk).repeat_interleave(r, 0).float()
    qa = qa * torch.rsqrt((qa * qa).sum(-1, keepdim=True) + 1e-6) / (dk ** 0.5)
    ka = ka * torch.rsqrt((ka * ka).sum(-1, keepdim=True) + 1e-6)
    beta = b.sigmoid()[0].float()
    g = (-A_log.float().exp() * F.softplus(a.float() + dt_bias))[0]
    S1 = S * g.exp()[:, None, None]
    delta = (v.view(lv, dv).float() - (S1 * ka[..., None]).sum(-2)) * beta[:, None]
    S2 = S1 + ka[..., None] * delta[:, None, :]
    o = (S2 * qa[..., None]).sum(-2).to(mixed.dtype)                                   # (lv, dv)
    of = o.float()
    on = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)).to(mixed.dtype)
    on = (norm_w * on) * F.silu(z.view(lv, dv).float())
    return full[None, :, 1:], S2, on.to(mixed.dtype).reshape(1, lv * dv)


def _f_delta_verify(conv_state, mixed, conv_w, S, b, a, A_log, dt_bias, z, norm_w, eps: float, lk: int, dk: int, lv: int, dv: int):
    """T-row verify pass of the Gated DeltaNet (as Qwen35Decoder._delta's verify branch, follow-up H), fused:
    the conv as a K-term sum per row, the recurrent rule row by row from S (unrolled, T is static).
    Returns the conv inputs (1, C, K - 1 + T), the state after every row (T, lv, dk, dv), the output (T, lv * dv)."""
    T, K = mixed.shape[0], conv_w.shape[-1]
    full = torch.cat([conv_state[0], mixed.t()], -1).to(conv_w.dtype)                 # (C, K - 1 + T)
    cw = conv_w.float()
    conv = full[:, 0:T].float() * cw[:, 0:1]
    for i in range(1, K):
        conv = conv + full[:, i:i + T].float() * cw[:, i:i + 1]
    conv = F.silu(conv).to(mixed.dtype).t()                                            # (T, C)
    q, k, v = conv.split([lk * dk, lk * dk, lv * dv], -1)
    r = lv // lk
    qa = q.reshape(T, lk, dk).repeat_interleave(r, 1).float()
    ka = k.reshape(T, lk, dk).repeat_interleave(r, 1).float()
    qa = qa * torch.rsqrt((qa * qa).sum(-1, keepdim=True) + 1e-6) / (dk ** 0.5)
    ka = ka * torch.rsqrt((ka * ka).sum(-1, keepdim=True) + 1e-6)
    beta = b.sigmoid().float()                                                         # (T, lv)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)                         # (T, lv)
    vf = v.reshape(T, lv, dv).float()
    hist, outs = [], []
    for t in range(T):
        S = S * g[t].exp()[:, None, None]
        delta = (vf[t] - (S * ka[t][..., None]).sum(-2)) * beta[t][:, None]
        S = S + ka[t][..., None] * delta[:, None, :]
        hist.append(S)
        outs.append((S * qa[t][..., None]).sum(-2))
    o = torch.stack(outs).to(mixed.dtype)                                              # (T, lv, dv)
    of = o.reshape(-1, dv).float()
    on = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)).to(mixed.dtype)
    on = (norm_w * on) * F.silu(z.reshape(-1, dv).float())
    return full[None], torch.stack(hist), on.to(mixed.dtype).reshape(T, lv * dv)


def _f_qk_rope(q, k, qn, kn, cos, sin, rot: int, eps: float):
    def rope(x):
        r, pz = x[..., :rot], x[..., rot:]
        h = rot // 2
        return torch.cat([r * cos + torch.cat([-r[..., h:], r[..., :h]], -1) * sin, pz], -1)
    return rope(_zc_norm(q, qn, eps)).transpose(0, 1), rope(_zc_norm(k, kn, eps)).transpose(0, 1)


def _f_gate(out, gate):
    return out * torch.sigmoid(gate)


_FUSED = {}


def fused(name):
    if name not in _FUSED:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _FUSED[name] = torch.compile(globals()[name], dynamic=False, options={"triton.cudagraphs": False})
    return _FUSED[name]


def is_qwen35(hf_model) -> bool:
    mt = getattr(hf_model.config, "model_type", "")
    return mt.startswith("qwen3_5")


def _tinygemm_from_packed(mod):
    """PackedLinear (compressed-tensors symmetric INT4, group g) -> tinygemm operands, exactly q * scale."""
    from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
    q = unpack_from_int32(mod.weight_packed, 4, torch.Size((mod.out_features, mod.in_features))).to(torch.int32)
    scale = mod.weight_scale.float()
    return int4_pack_codes(q + 8, scale, -8 * scale, mod.group_size)


class Qwen35Decoder:
    def __init__(self, hf_model, max_len: int, attn: str = "sdpa", rope: str = "ref"):
        self.attn_impl, self.rope_impl, self.step_attn = attn, rope, "cudnn"
        cfg = hf_model.config.get_text_config()
        lm = hf_model.model.language_model if hasattr(hf_model.model, "language_model") else hf_model.model
        self.cfg = cfg
        self.eps = cfg.rms_norm_eps
        self.dev, self.dtype, self.max_len = hf_model.device, torch.bfloat16, max_len
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.lk, self.lv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.dk, self.dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        self.conv_k = cfg.linear_conv_kernel_dim
        self.embed = lm.embed_tokens.weight
        self.final_norm = lm.norm.weight
        self.lm_head = hf_model.lm_head.weight
        self.types = ["delta" if t == "linear_attention" else "attn" for t in cfg.layer_types]
        self.layers, self.slot = [], []
        count = {"delta": 0, "attn": 0}

        def take(mod):
            """tinygemm operands for a packed layer, freeing the checkpoint's packed buffers."""
            w = _tinygemm_from_packed(mod)
            for b in ("weight_packed", "weight_scale"):
                delattr(mod, b)
            return w

        with torch.no_grad():
            for i, layer in enumerate(lm.layers):
                t = self.types[i]
                L = {"type": t, "ln1": layer.input_layernorm.weight, "ln2": layer.post_attention_layernorm.weight,
                     "g": take(layer.mlp.gate_proj), "u": take(layer.mlp.up_proj), "down": take(layer.mlp.down_proj)}
                if t == "delta":
                    la = layer.linear_attn
                    L.update({"qkv": take(la.in_proj_qkv), "z": take(la.in_proj_z), "out": take(la.out_proj),
                              "b": la.in_proj_b.weight, "a": la.in_proj_a.weight, "conv": la.conv1d.weight.squeeze(1),
                              "A_log": la.A_log, "dt_bias": la.dt_bias, "norm": la.norm.weight})
                else:
                    a = layer.self_attn
                    L.update({"q": take(a.q_proj), "k": take(a.k_proj), "v": take(a.v_proj), "o": take(a.o_proj),
                              "qn": a.q_norm.weight, "kn": a.k_norm.weight})
                self.layers.append(L)
                self.slot.append(count[t])
                count[t] += 1
                torch.cuda.empty_cache()
        rp = cfg.rope_parameters
        self.rot = int(self.hd * rp.get("partial_rotary_factor", 1.0))
        self.theta = rp["rope_theta"]
        self.n_attn = count["attn"]
        # fused q/k norm weights (L1): per attention layer, (nh + nkv, hd) of (1 + norm weight)
        ws = [torch.cat([(1 + l["qn"].float()).expand(self.nh, -1), (1 + l["kn"].float()).expand(self.nkv, -1)])
              for l in self.layers if l["type"] == "attn"]
        self.qk_w = torch.stack(ws).to(self.dtype).contiguous()                       # (n_attn, nh + nkv, hd)
        self.set_max_len(max_len)
        conv_dim = 2 * self.lk * self.dk + self.lv * self.dv
        self.conv_state = torch.zeros(count["delta"], 1, conv_dim, self.conv_k - 1, device=self.dev, dtype=self.dtype)
        self.rec_state = torch.zeros(count["delta"], 1, self.lv, self.dk, self.dv, device=self.dev, dtype=torch.float32)
        self.n_delta = count["delta"]
        self.verify_rows = 0          # > 0 after enable_verify(rows): multi-row steps keep per-row states (follow-up H)
        self.extending = False        # prefill_chunked: multi-row chunks continue the recurrent / conv state (follow-up I)
        self.fuse = False             # F3a: fused helpers for one-token steps (compile_step)
        self.last_hidden = None       # pre-final-norm hidden states of the last forward (T, d)
        self.w4, self.w4_head, self.draft_vocab = None, None, None
        self.plan = SpecPlan()

    def set_max_len(self, max_len: int):
        """(Re)allocate the attention layers' KV cache and the RoPE tables for max_len positions."""
        self.max_len = max_len
        self.k_cache = self.v_cache = None
        torch.cuda.empty_cache()
        inv = 1.0 / (self.theta ** (torch.arange(0, self.rot, 2, dtype=torch.float, device=self.dev) / self.rot))
        ang = torch.arange(max_len, device=self.dev, dtype=torch.float32)[:, None] * inv[None]
        emb = torch.cat([ang, ang], -1)
        self.cos, self.sin = emb.cos().to(self.dtype), emb.sin().to(self.dtype)
        sign = torch.cat([-torch.ones(self.rot // 2), torch.ones(self.rot // 2)]).to(self.dev, self.dtype)
        self.sin_signed = self.sin * sign
        self.kv = None
        if getattr(self, "kv_mode", None):        # follow-up L: KVStore (Triton decode attention, optional int8 / int4)
            from radan_engine.kv import KVStore
            self.kv = KVStore(self.n_attn, self.nkv, max_len, self.hd, self.kv_mode, self.dev)
            self.k_cache = self.v_cache = torch.zeros(1, 1, 1, 1, 1, device=self.dev, dtype=self.dtype)
            return
        self.k_cache = torch.zeros(self.n_attn, 1, self.nkv, max_len, self.hd, device=self.dev, dtype=self.dtype)
        self.v_cache = torch.zeros_like(self.k_cache)

    def set_kv(self, mode):
        """Follow-up L: None (BF16 cache, batched-matmul attention), "bf16" (Triton split-KV attention), "int8", "int4"."""
        self.kv_mode = mode
        self.set_max_len(self.max_len)

    def prefill_chunked(self, ids: torch.Tensor, start: int = 0, chunk: int = 2048, logits_rows: int = 1, w4: bool = False):
        """As Qwen3Decoder.prefill_chunked; the DeltaNet layers run each chunk with Transformers' chunked delta
        rule from the current recurrent state, and the conv from the current conv state."""
        T = ids.shape[0]
        if start == 0:
            self.k_cache.zero_()
            self.v_cache.zero_()
            self.conv_state.zero_()
            self.rec_state.zero_()
        keep, ar = [], torch.arange(self.max_len, device=self.dev)
        self.extending = True
        try:
            for s0 in range(0, T, chunk):
                pos = torch.arange(start + s0, start + min(T, s0 + chunk), device=self.dev)
                mask = torch.where(ar[None, :] <= pos[:, None], 0.0, float("-inf")).to(self.dtype)[None, None]
                lg = self.forward(ids[s0:s0 + chunk], pos, mask)
                first_kept = T - logits_rows
                if s0 + lg.shape[0] > first_kept:
                    keep.append(lg[max(0, first_kept - s0):])
                del lg, mask
        finally:
            self.extending = False
        return torch.cat(keep)

    def snapshot(self):
        """The recurrent and conv states change on every step; the attention KV cache only past the prompt."""
        return self.rec_state.clone(), self.conv_state.clone()

    def restore(self, snap):
        self.rec_state.copy_(snap[0])
        self.conv_state.copy_(snap[1])

    # ---- building blocks -------------------------------------------------------------------------
    def norm(self, x, w):
        """Qwen3_5RMSNorm: fp32, rsqrt(mean + eps), times (1 + w) in fp32."""
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + w.float())).type_as(x)

    @staticmethod
    def _rot_half(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def _rope(self, x, pos_idx):
        cos, sin = self.cos[pos_idx][:, None], self.sin[pos_idx][:, None]           # (T, 1, rot)
        r, p = x[..., :self.rot], x[..., self.rot:]
        return torch.cat([r * cos + self._rot_half(r) * sin, p], -1)

    def _mlp(self, i, x):
        L = self.layers[i]
        if self.fuse and x.shape[0] <= 16:
            return int4_linear(fused("_f_silu_mul")(int4_linear(x, L["g"]), int4_linear(x, L["u"])), L["down"])
        return int4_linear(F.silu(int4_linear(x, L["g"])) * int4_linear(x, L["u"]), L["down"])

    def _attention(self, i, x, pos_idx, mask, prefill):
        L, j = self.layers[i], self.slot[i]
        T = x.shape[0]
        qg = int4_linear(x, L["q"]).view(T, self.nh, 2 * self.hd)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(T, -1)
        k = int4_linear(x, L["k"]).view(T, self.nkv, self.hd)
        v = int4_linear(x, L["v"]).view(T, self.nkv, self.hd)
        if self.kv is not None:
            return self._attention_kv(L, j, T, q, k, v, gate, pos_idx, mask, prefill)
        if self.fuse and T <= 16 and not prefill and not self.extending:   # one-token steps and verify passes
            cos, sin = self.cos[pos_idx][:, None], self.sin[pos_idx][:, None]
            q, k = fused("_f_qk_rope")(q, k, L["qn"], L["kn"], cos, sin, self.rot, self.eps)
            v = v.transpose(0, 1)
            kc, vc = self.k_cache[j], self.v_cache[j]
            kc[0].index_copy_(1, pos_idx, k)
            vc[0].index_copy_(1, pos_idx, v)
            out = bmm_attention(q[None], kc, vc, mask, self.hd ** -0.5)
            return int4_linear(fused("_f_gate")(out[0].transpose(0, 1).reshape(T, -1), gate), L["o"])
        if self.rope_impl == "fused":
            # zero-centred q/k norm with (1 + w) folded into the RoPE tables on the rotated dims
            qk = F.rms_norm(torch.cat([q, k], 1), (self.hd,), None, self.eps) * self.qk_w[j]   # weight before RoPE
            cos, sin = self.cos[pos_idx][:, None], self.sin_signed[pos_idx][:, None]
            r = qk[..., :self.rot]
            qk = torch.cat([torch.addcmul(r * cos, torch.roll(r, self.rot // 2, -1), sin), qk[..., self.rot:]], -1)
            q, k = qk[:, :self.nh], qk[:, self.nh:]
        else:
            q, k = self.norm(q, L["qn"]), self.norm(k, L["kn"])
            q, k = self._rope(q, pos_idx), self._rope(k, pos_idx)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        kc, vc = self.k_cache[j], self.v_cache[j]
        kc[0].index_copy_(1, pos_idx, k)
        vc[0].index_copy_(1, pos_idx, v)
        scale = self.hd ** -0.5
        if self.attn_impl == "fast" and prefill:
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                out = F.scaled_dot_product_attention(q[None], kc[:, :, :T], vc[:, :, :T], is_causal=True, enable_gqa=True, scale=scale)
        elif self.attn_impl == "fast" and T <= 16:
            out = bmm_attention(q[None], kc, vc, mask, scale)
        else:
            out = F.scaled_dot_product_attention(q[None], kc, vc, attn_mask=mask, enable_gqa=True, scale=scale)
        out = out[0].transpose(0, 1).reshape(T, -1) * torch.sigmoid(gate)
        return int4_linear(out, L["o"])

    def _delta(self, i, x, prefill):
        from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn, torch_chunk_gated_delta_rule
        L, j = self.layers[i], self.slot[i]
        T = x.shape[0]
        mixed = int4_linear(x, L["qkv"])                                              # (T, conv_dim)
        z = int4_linear(x, L["z"]).view(T, self.lv, self.dv)
        b, a = F.linear(x, L["b"]), F.linear(x, L["a"])
        if self.fuse and T == 1 and not prefill and not self.extending:
            nc, S2, on = fused("_f_delta_step")(self.conv_state[j], mixed, L["conv"], self.rec_state[j][0], b, a, L["A_log"],
                                                L["dt_bias"], z, L["norm"], self.eps, self.lk, self.dk, self.lv, self.dv)
            self.conv_state[j].copy_(nc)
            self.rec_state[j][0].copy_(S2)
            return int4_linear(on, L["out"])
        if self.fuse and 1 < T <= self.verify_rows and not prefill and not self.extending:   # fused verify pass (H + F3a)
            full, hist, on = fused("_f_delta_verify")(self.conv_state[j], mixed, L["conv"], self.rec_state[j][0], b, a, L["A_log"],
                                                      L["dt_bias"], z, L["norm"], self.eps, self.lk, self.dk, self.lv, self.dv)
            self.conv_full[j, ..., :full.shape[-1]].copy_(full)
            self.rec_hist[j, :T].copy_(hist)
            return int4_linear(on, L["out"])
        mt = mixed.t()[None]                                                          # (1, C, T)
        if self.extending and T > 1:
            full = torch.cat([self.conv_state[j], mt], -1).to(L["conv"].dtype)
            self.conv_state[j].copy_(full[..., -(self.conv_k - 1):])
            conv = F.silu(F.conv1d(full, L["conv"].unsqueeze(1), None, padding=0, groups=full.shape[1])[..., -T:]).to(mt.dtype)
        elif prefill:
            self.conv_state[j].copy_(F.pad(mt, (self.conv_k - 1 - T, 0))[..., -(self.conv_k - 1):] if T < self.conv_k - 1
                                     else mt[..., -(self.conv_k - 1):])
            conv = causal_conv1d_fn(mt, L["conv"], None, activation="silu")
        else:
            assert T == 1 or T <= self.verify_rows, "multi-row steps need enable_verify(rows) first"
            full = torch.cat([self.conv_state[j], mt], -1).to(L["conv"].dtype)
            if T == 1:
                self.conv_state[j].copy_(full[..., 1:])
            else:
                self.conv_full[j, ..., :full.shape[-1]].copy_(full)      # commit() re-cuts the conv state from it
            conv = F.silu(F.conv1d(full, L["conv"].unsqueeze(1), None, padding=0, groups=full.shape[1])[..., -T:]).to(mt.dtype)
        q, k, v = conv[0].t().split([self.lk * self.dk, self.lk * self.dk, self.lv * self.dv], -1)
        q, k, v = q.reshape(T, self.lk, self.dk), k.reshape(T, self.lk, self.dk), v.reshape(T, self.lv, self.dv)
        beta = b.sigmoid()
        g = -L["A_log"].float().exp() * F.softplus(a.float() + L["dt_bias"])
        if prefill or (self.extending and T > 1):
            o, s = torch_chunk_gated_delta_rule(q[None], k[None], v[None], g=g[None], beta=beta[None],
                                                initial_state=None if prefill else self.rec_state[j],
                                                output_final_state=True, use_qk_l2norm_in_kernel=True)
            self.rec_state[j].copy_(s)
            o = o[0]                                                                  # (T, lv, dv)
        else:
            r = self.lv // self.lk
            qa = q.repeat_interleave(r, 1).float()                                    # (T, lv, dk)
            ka = k.repeat_interleave(r, 1).float()
            qa = qa * torch.rsqrt((qa * qa).sum(-1, keepdim=True) + 1e-6)
            ka = ka * torch.rsqrt((ka * ka).sum(-1, keepdim=True) + 1e-6)
            qa = qa / (self.dk ** 0.5)
            outs = []
            for t in range(T):  # the recurrent rule, one row at a time (Transformers' torch_recurrent_gated_delta_rule)
                S = self.rec_state[j][0] if T == 1 else (self.rec_state[j][0] if t == 0 else self.rec_hist[j, t - 1]).clone()
                qf, kf = qa[t], ka[t]
                S.mul_(g[t].exp()[:, None, None])
                kv_mem = (S * kf[..., None]).sum(-2)                                  # (lv, dv)
                delta = (v[t].float() - kv_mem) * beta[t].float()[:, None]
                S.add_(kf[..., None] * delta[:, None, :])
                if T > 1:
                    self.rec_hist[j, t].copy_(S)                                       # commit() picks the kept row's state
                outs.append((S * qf[..., None]).sum(-2))
            o = torch.stack(outs).to(x.dtype)                                         # (T, lv, dv)
        of = o.reshape(-1, self.dv).float()
        on = (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        on = (L["norm"] * on) * F.silu(z.reshape(-1, self.dv).float())
        return int4_linear(on.to(x.dtype).reshape(T, -1), L["out"])

    # ---- forward -----------------------------------------------------------------------------------
    def _attention_kv(self, L, j, T, q, k, v, gate, pos_idx, mask, prefill):
        """Follow-up L: K / V go to the KVStore (quantised there for int8 / int4); one-token steps and verify passes
        use its split-KV kernel, prefill chunks dense attention over its (dequantised) prefix."""
        if T <= 16:
            cos, sin = self.cos[pos_idx][:, None], self.sin[pos_idx][:, None]
            q, k = fused("_f_qk_rope")(q, k, L["qn"], L["kn"], cos, sin, self.rot, self.eps)   # (nh, T, hd), (nkv, T, hd)
        else:                                                   # prefill chunks: the reference norm + RoPE
            q, k = self.norm(q, L["qn"]), self.norm(k, L["kn"])
            q, k = self._rope(q, pos_idx).transpose(0, 1), self._rope(k, pos_idx).transpose(0, 1)
        self.kv.write(j, pos_idx, k.transpose(0, 1), v)
        scale = self.hd ** -0.5
        if T <= 16 and not prefill and not self.extending:
            out = self.kv.attend(j, q, pos_idx[:1], scale)                      # (nh, T, hd)
            return int4_linear(fused("_f_gate")(out.transpose(0, 1).reshape(T, -1), gate), L["o"])
        end = int(pos_idx[-1]) + 1
        kd, vd = self.kv.dense(j, end)
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            if prefill:
                out = F.scaled_dot_product_attention(q[None], kd[None], vd[None], is_causal=True, enable_gqa=True, scale=scale)
            else:
                out = F.scaled_dot_product_attention(q[None], kd[None], vd[None], attn_mask=mask[..., :end], enable_gqa=True, scale=scale)
        out = out[0].transpose(0, 1).reshape(T, -1) * torch.sigmoid(gate)
        return int4_linear(out, L["o"])

    def forward(self, ids, pos_idx, mask, exact_first: bool = False, skip_attn=(), skip_mlp=(), w4: bool = False,
                prefill: bool = False):
        assert not w4 and self.plan.mode == "none" and not skip_attn and not skip_mlp, "not implemented for Qwen3.5 blocks"
        h = self.embed[ids]
        if self.fuse and (ids.shape[0] == 1 or ids.shape[0] <= self.verify_rows) and not prefill and not self.extending:
            e = self.eps
            x = fused("_f_norm")(h, self.layers[0]["ln1"], e)
            for i, L in enumerate(self.layers):
                a = self._delta(i, x, prefill) if L["type"] == "delta" else self._attention(i, x, pos_idx, mask, prefill)
                h, xm = fused("_f_add_norm")(h, a, L["ln2"], e)
                nxt = self.layers[i + 1]["ln1"] if i + 1 < len(self.layers) else self.final_norm
                h, x = fused("_f_add_norm")(h, self._mlp(i, xm), nxt, e)
            self.last_hidden = h
            return F.linear(x, self.lm_head)
        for i, L in enumerate(self.layers):
            x = self.norm(h, L["ln1"])
            h = h + (self._delta(i, x, prefill) if L["type"] == "delta" else self._attention(i, x, pos_idx, mask, prefill))
            h = h + self._mlp(i, self.norm(h, L["ln2"]))
        self.last_hidden = h
        return F.linear(self.norm(h, self.final_norm), self.lm_head)

    # ---- multi-row verify (follow-up H) -----------------------------------------------------------------
    def enable_verify(self, rows: int):
        """Allow forward() steps of up to `rows` rows after prefill (a self-spec verify pass): the DeltaNet
        layers keep the recurrent state after every row and the conv inputs, for commit()."""
        self.verify_rows = rows
        conv_dim = self.conv_state.shape[2]
        self.rec_hist = torch.zeros(self.n_delta, rows, self.lv, self.dk, self.dv, device=self.dev, dtype=torch.float32)
        self.conv_full = torch.zeros(self.n_delta, 1, conv_dim, self.conv_k - 1 + rows, device=self.dev, dtype=self.dtype)

    def commit(self, n):
        """After a verify pass, keep rows 0..n (n a 0-dim long tensor, graph-safe): the recurrent state becomes
        row n's and the conv state the conv_k - 1 inputs ending at row n."""
        self.rec_state.copy_(self.rec_hist.index_select(1, n.view(1)))
        idx = n + 1 + torch.arange(self.conv_k - 1, device=self.dev)
        self.conv_state.copy_(self.conv_full.index_select(3, idx))

    def prefill(self, ids: torch.Tensor, w4: bool = False) -> torch.Tensor:
        T = ids.shape[0]
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.conv_state.zero_()
        self.rec_state.zero_()
        pos = torch.arange(T, device=self.dev)
        mask = torch.full((T, self.max_len), float("-inf"), device=self.dev, dtype=self.dtype)
        mask[:, :T] = torch.triu(torch.full((T, T), float("-inf"), device=self.dev, dtype=self.dtype), 1)
        return self.forward(ids, pos, mask[None, None], prefill=True)

    def greedy(self, logits, w4=False):
        return logits.argmax(-1)

    def compile_step(self, on: bool = True):
        """Follow-up F3a: one-token steps use the fused (torch.compile'd) pure helpers: the whole DeltaNet update
        (conv, gates, recurrent rule, gated norm), q/k norm + RoPE, the output gate, SiLU-mul, norms + residuals."""
        self.fuse = on

    def quantize_int4(self, *a, **k):
        raise RuntimeError("out of memory: the Qwen3.5 decoder's weights are already INT4; no separate INT4 copy")

    def make_step(self, prompt_len: int, first_token: int, w4: bool = False):
        from radan_engine.decode import Qwen3Decoder
        return Qwen3Decoder.make_step(self, prompt_len, first_token, w4)
