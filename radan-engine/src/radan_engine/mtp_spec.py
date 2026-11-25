"""Self-speculative greedy decoding with Qwen3.5 / 3.8's own MTP layer (docs/decode_levers.md, follow-up H).

MTP (from the checkpoint's model_mtp.safetensors, BF16):
    x = fc([norm_e(embed(t_(i+1))), norm_h(h_i)])          h_i: main model's hidden at position i AFTER its final norm
    x = x + gated_attn(norm(x));  x = x + mlp(norm(x))     its own KV cache, RoPE at position i + 1
    y = norm_mtp(x); logits = lm_head(y)                   predicts t_(i+2); y (normed) is the hidden for chaining
(as llama.cpp's qwen35 MTP graph and vLLM feed it; the first run, 2026-11-25, used the pre-norm h and x:
results/processed/decode_levers/Qwen3.8-27B-INT4_tps_mtp_prenorm.json)

One round, entirely on the GPU and captured in one CUDA graph (state: `last` at position p, and the
previous verify pass's hidden rows with the tokens that follow them, n_prev + 1 of them valid):
  1  MTP over the previous verify rows (true main hidden states): refreshes the MTP KV cache at those
     positions; the last valid row gives d_1 (rows past n_prev write KV that is overwritten before use)
  2  d_2 .. d_k: MTP on its own output hidden, one row each
  3  the full model verifies [last, d_1 .. d_k] at p .. p + k; n = agreeing prefix; keep g_0 .. g_n;
     the DeltaNet states roll back to row n (Qwen35Decoder.commit)
Kept tokens are the full model's greedy choices given the kept prefix.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from radan_engine.gemma4_decode import bmm_attention


class MTPLayer:
    def __init__(self, dec, path: str, head: str = "bf16"):
        """head="int4": the draft's logits come from an INT4 copy of the shared LM head (groups of 32, clipping
        search; follow-up H2). Only the draft uses it; the full model verifies with its own BF16 head."""
        self.dec = dec
        self.head_w4 = None
        w = {}
        with safe_open(str(Path(path) / "model_mtp.safetensors"), framework="pt", device=str(dec.dev)) as st:
            for k in st.keys():
                w[k[len("mtp."):]] = st.get_tensor(k).to(dec.dtype)
        self.w = w
        self.k_cache = self.v_cache = None
        self.ensure_len()
        if head == "int4":
            from radan_engine.decode import int4_pack_chunked
            self.head_w4 = int4_pack_chunked(dec.lm_head, 32, clip=True)

    def ensure_len(self):
        """(Re)allocate the MTP KV cache to the decoder's current max_len (it changes with set_max_len)."""
        d = self.dec
        if self.k_cache is None or self.k_cache.shape[2] != d.max_len:
            self.k_cache = self.v_cache = None
            torch.cuda.empty_cache()
            self.k_cache = torch.zeros(1, d.nkv, d.max_len, d.hd, device=d.dev, dtype=d.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)

    def forward(self, tok, h, pos, mask):
        """tok (T,) tokens t_(i+1), h (T, d) main (or chained) hidden at i, pos (T,) = i + 1 -> (logits, hidden)."""
        d, w = self.dec, self.w
        T = tok.shape[0]
        e = d.norm(d.embed[tok], w["pre_fc_norm_embedding.weight"])
        hh = d.norm(h, w["pre_fc_norm_hidden.weight"])
        x = F.linear(torch.cat([e, hh], -1), w["fc.weight"])
        y = d.norm(x, w["layers.0.input_layernorm.weight"])
        qg = F.linear(y, w["layers.0.self_attn.q_proj.weight"]).view(T, d.nh, 2 * d.hd)
        q, gate = qg.chunk(2, dim=-1)
        gate = gate.reshape(T, -1)
        k = F.linear(y, w["layers.0.self_attn.k_proj.weight"]).view(T, d.nkv, d.hd)
        v = F.linear(y, w["layers.0.self_attn.v_proj.weight"]).view(T, d.nkv, d.hd)
        q = d._rope(d.norm(q, w["layers.0.self_attn.q_norm.weight"]), pos).transpose(0, 1)
        k = d._rope(d.norm(k, w["layers.0.self_attn.k_norm.weight"]), pos).transpose(0, 1)
        self.k_cache[0].index_copy_(1, pos, k)
        self.v_cache[0].index_copy_(1, pos, v.transpose(0, 1))
        if T <= 16:
            o = bmm_attention(q[None], self.k_cache, self.v_cache, mask, d.hd ** -0.5)
        else:  # prompt chunks: a memory-efficient kernel (batched matmuls would build T x max_len scores per head)
            o = F.scaled_dot_product_attention(q[None], self.k_cache, self.v_cache, attn_mask=mask, enable_gqa=True, scale=d.hd ** -0.5)
        o = o[0].transpose(0, 1).reshape(T, -1) * torch.sigmoid(gate)
        x = x + F.linear(o, w["layers.0.self_attn.o_proj.weight"])
        y = d.norm(x, w["layers.0.post_attention_layernorm.weight"])
        x = x + F.linear(F.silu(F.linear(y, w["layers.0.mlp.gate_proj.weight"])) * F.linear(y, w["layers.0.mlp.up_proj.weight"]),
                         w["layers.0.mlp.down_proj.weight"])
        y = d.norm(x, w["norm.weight"])
        if self.head_w4 is not None:
            from radan_engine.decode import int4_linear
            return int4_linear(y, self.head_w4), y
        return F.linear(y, d.lm_head), y


class MTPSelfSpec:
    def __init__(self, dec, mtp: MTPLayer, k: int, max_new: int = 512):
        self.dec, self.mtp, self.k = dec, mtp, k
        dev = dec.dev
        dec.enable_verify(k + 1)
        self.ar = torch.arange(dec.max_len, device=dev)
        self.kk = torch.arange(k + 1, device=dev)
        self.buf = {"p": torch.zeros(1, dtype=torch.long, device=dev),           # position of `last`
                    "drafts": torch.zeros(k + 1, dtype=torch.long, device=dev),   # [last, d_1 .. d_k]
                    "prev_h": torch.zeros(k + 1, dec.embed.shape[1], device=dev, dtype=dec.dtype),
                    "prev_tok": torch.zeros(k + 1, dtype=torch.long, device=dev),
                    "prev_n": torch.zeros(1, dtype=torch.long, device=dev),
                    "prev_p0": torch.zeros(1, dtype=torch.long, device=dev),      # position of the previous verify's row 0
                    "out": torch.zeros(max_new + 8 * (k + 1), dtype=torch.long, device=dev),
                    "n_out": torch.zeros(1, dtype=torch.long, device=dev),
                    "rounds": torch.zeros(1, dtype=torch.long, device=dev),
                    "accepted": torch.zeros(1, dtype=torch.long, device=dev)}
        self.graph = None

    def _mask(self, positions):
        allowed = self.ar[None, :] <= positions[:, None]
        return torch.where(allowed, 0.0, float("-inf")).to(self.dec.dtype)[None, None]

    def _round(self):
        b, dec, mtp, k = self.buf, self.dec, self.mtp, self.k
        # 1: MTP over the previous verify rows (MTP positions prev_p0 + 1 + r); d_1 from row prev_n
        pos = b["prev_p0"] + 1 + self.kk
        logits, hm = mtp.forward(b["prev_tok"], b["prev_h"], pos, self._mask(pos))
        n0 = b["prev_n"]
        b["drafts"][1:2].copy_(logits.index_select(0, n0).argmax(-1))
        h = hm.index_select(0, n0)
        # 2: chained drafts; d_j at MTP position p + j - 1
        for j in range(2, k + 1):
            pj = b["p"] + j - 1
            logits, h = mtp.forward(b["drafts"][j - 1:j], h, pj, self._mask(pj))
            b["drafts"][j:j + 1].copy_(logits.argmax(-1))
        # 3: verify [last, d_1 .. d_k] at p .. p + k
        positions = b["p"] + self.kk
        g = dec.forward(b["drafts"], positions, self._mask(positions)).argmax(-1)
        hv = dec.norm(dec.last_hidden, dec.final_norm)      # the MTP takes the main model's post-norm hidden
        match = (b["drafts"][1:] == g[:-1]).long()
        n = match.cumprod(0).sum()
        dec.commit(n)
        b["out"].index_copy_(0, b["n_out"] + self.kk, g)
        b["n_out"].add_(n + 1)
        # next round's MTP inputs: verify rows with the tokens that follow them (drafts for r < n, g_n at r = n)
        b["prev_h"].copy_(hv)
        b["prev_tok"][:-1].copy_(b["drafts"][1:])
        b["prev_tok"].index_copy_(0, n.view(1), g.index_select(0, n.view(1)))
        b["prev_n"].copy_(n.view(1))
        b["prev_p0"].copy_(b["p"])
        b["p"].add_(n + 1)
        b["drafts"][0:1].copy_(g.index_select(0, n.view(1)))
        b["rounds"].add_(1)
        b["accepted"].add_(n)

    def prefill_long(self, prompt: torch.Tensor, chunk: int = 2048):
        """Chunked prefill of the main model (Qwen35Decoder.prefill_chunked semantics) and of the MTP KV cache,
        chunk by chunk, so a long prompt never needs all its hidden states at once. Returns the first token and
        the last prompt row's hidden state, for reset() (follow-up H at long contexts)."""
        dec, mtp = self.dec, self.mtp
        P = prompt.shape[0]
        for t in ("k_cache", "v_cache"):
            getattr(dec, t).zero_()
        dec.conv_state.zero_()
        dec.rec_state.zero_()
        mtp.k_cache.zero_()
        mtp.v_cache.zero_()
        dec.extending = True
        try:
            for s0 in range(0, P, chunk):
                s1 = min(P, s0 + chunk)
                pos = torch.arange(s0, s1, device=dec.dev)
                lg = dec.forward(prompt[s0:s1], pos, self._mask(pos))
                hp = dec.norm(dec.last_hidden, dec.final_norm)
                n = min(s1, P - 1) - s0          # MTP rows i = s0 .. min(s1, P - 1) - 1: (h_i, t_(i+1)) at position i + 1
                if n > 0:
                    mp = torch.arange(s0 + 1, s0 + 1 + n, device=dec.dev)
                    mtp.forward(prompt[s0 + 1:s0 + 1 + n], hp[:n], mp, self._mask(mp))
        finally:
            dec.extending = False
        return int(lg[-1].argmax()), hp[-1].clone()

    def reset(self, P: int, first: int, h_last: torch.Tensor):
        """Round buffers for a prompt already in the caches (restore the decoder's snapshot first)."""
        b = self.buf
        b["p"].fill_(P)
        b["drafts"].zero_()
        b["drafts"][0] = first
        b["prev_h"].zero_()
        b["prev_h"][0].copy_(h_last)
        b["prev_tok"].zero_()
        b["prev_tok"][0] = first
        b["prev_n"].zero_()
        b["prev_p0"].fill_(P - 1)
        b["out"].zero_()
        b["out"][0] = first
        b["n_out"].fill_(1)
        b["rounds"].zero_()
        b["accepted"].zero_()

    def start(self, prompt: torch.Tensor):
        """Prefill the main model and the MTP KV cache over the prompt; the first new token is the model's."""
        dec, mtp, b = self.dec, self.mtp, self.buf
        P = prompt.shape[0]
        logits = dec.prefill(prompt)
        hp = dec.norm(dec.last_hidden, dec.final_norm)
        first = logits[-1].argmax()
        mtp.k_cache.zero_()
        mtp.v_cache.zero_()
        if P > 1:  # MTP rows i = 0 .. P - 2: (h_i, t_(i+1)) at positions 1 .. P - 1
            pos = torch.arange(1, P, device=dec.dev)
            mtp.forward(prompt[1:], hp[:-1], pos, self._mask(pos))
        b["p"].fill_(P)
        b["drafts"].zero_()
        b["drafts"][0] = first
        b["prev_h"].zero_()
        b["prev_h"][0].copy_(hp[-1])
        b["prev_tok"].zero_()
        b["prev_tok"][0] = first
        b["prev_n"].zero_()
        b["prev_p0"].fill_(P - 1)
        b["out"].zero_()
        b["out"][0] = first
        b["n_out"].fill_(1)
        b["rounds"].zero_()
        b["accepted"].zero_()

    def capture(self, prompt: torch.Tensor, reset_fn=None):
        """reset_fn (optional): puts the caches and buffers back to the prompt's state instead of start()."""
        again = reset_fn or (lambda: self.start(prompt))
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(2):
                again()
                self._round()
        torch.cuda.current_stream().wait_stream(warm)
        again()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._round()

    def generate(self, prompt: torch.Tensor, n_new: int, check_every: int = 1):
        self.start(prompt)
        b = self.buf
        while True:
            for _ in range(check_every):
                self.graph.replay() if self.graph is not None else self._round()
            if int(b["n_out"]) >= n_new:
                break
        return b["out"][:n_new].clone(), int(b["rounds"]), int(b["accepted"])
