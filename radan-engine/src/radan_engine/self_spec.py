"""Self-speculative greedy decoding on radan_engine.decode.Qwen3Decoder (docs/self_speculative_decoding.md).

One round, entirely on the GPU and captured in one CUDA graph:
  draft   k single-token steps of the same model with some attention / MLP sub-layers skipped,
          starting from the last accepted token: d_1..d_k
  verify  the full model on [last, d_1..d_k] at positions p..p+k in one pass -> greedy targets g_0..g_k
  accept  n = length of the agreeing prefix (d_(j+1) == g_j); keep g_0..g_n (n + 1 tokens: n accepted
          drafts and the full model's own next token), advance p by n + 1
The kept tokens are exactly the full model's greedy choices given the kept prefix. Draft and verify
share the KV cache; verify rewrites every position it covers, and stale entries past the kept prefix
are rewritten before anything attends to them.
"""

import torch


class SelfSpecDecoder:
    def __init__(self, dec, k: int, skip_attn=(), skip_mlp=(), max_new: int = 512, draft_w4: bool = False):
        """draft_w4: draft with the decoder's INT4 weight copies (dec.quantize_int4) instead of, or as
        well as, skipping sub-layers; verify always uses the BF16 weights."""
        self.dec, self.k, self.draft_w4 = dec, k, draft_w4
        self.skip_attn, self.skip_mlp = frozenset(skip_attn), frozenset(skip_mlp)
        dev = dec.dev
        self.dev = dev
        self.ar = torch.arange(dec.max_len, device=dev)
        self.kk = torch.arange(k + 1, device=dev)
        self.buf = {"p": torch.zeros(1, dtype=torch.long, device=dev),          # position of `last`
                    "drafts": torch.zeros(k + 1, dtype=torch.long, device=dev),  # [last, d_1..d_k]
                    "out": torch.zeros(max_new + 8 * (k + 1), dtype=torch.long, device=dev),  # room for check_every <= 8
                    "n_out": torch.zeros(1, dtype=torch.long, device=dev),
                    "rounds": torch.zeros(1, dtype=torch.long, device=dev),
                    "accepted": torch.zeros(1, dtype=torch.long, device=dev)}
        self.graph = None

    def _mask(self, positions):
        allowed = self.ar[None, :] <= positions[:, None]
        return torch.where(allowed, 0.0, float("-inf")).to(self.dec.dtype)[None, None]

    def _round(self):
        b, dec, k = self.buf, self.dec, self.k
        # draft
        for j in range(k):
            pos = b["p"] + j
            logits = dec.forward(b["drafts"][j:j + 1], pos, self._mask(pos),
                                 skip_attn=self.skip_attn, skip_mlp=self.skip_mlp, w4=self.draft_w4)
            tok = dec.greedy(logits, self.draft_w4) if hasattr(dec, "greedy") else logits.argmax(-1)
            b["drafts"][j + 1:j + 2].copy_(tok)
        # verify
        positions = b["p"] + self.kk
        g = dec.forward(b["drafts"], positions, self._mask(positions)).argmax(-1)
        match = (b["drafts"][1:] == g[:-1]).long()
        n = match.cumprod(0).sum()
        b["out"].index_copy_(0, b["n_out"] + self.kk, g)
        b["n_out"].add_(n + 1)
        b["p"].add_(n + 1)
        b["drafts"][0:1].copy_(g.gather(0, n.view(1)))
        b["rounds"].add_(1)
        b["accepted"].add_(n)

    def start(self, prompt: torch.Tensor, first_token=None):
        """Prefill with the full model; the first new token is the full model's greedy choice. With first_token
        given, the cache already holds the prompt (a decoder whose steps change nothing below the prompt's
        positions, e.g. Qwen3Decoder) and no prefill is run (follow-up J at long contexts)."""
        dec = self.dec
        if first_token is not None:
            return self._reset(prompt.shape[0], torch.tensor(first_token, device=self.dev))
        # chunked prefill where the decoder has it (long prompts: no T x T mask, logits of the last row only)
        logits = dec.prefill_chunked(prompt) if hasattr(dec, "prefill_chunked") and prompt.shape[0] > 2048 else dec.prefill(prompt)
        first = logits[-1].argmax()
        self._reset(prompt.shape[0], first)

    def _reset(self, P, first):
        b = self.buf
        b["p"].fill_(P)
        b["drafts"][0] = first
        b["out"].zero_()
        b["out"][0] = first
        b["n_out"].fill_(1)
        b["rounds"].zero_()
        b["accepted"].zero_()

    def capture(self, prompt: torch.Tensor, first_token=None):
        """Capture one round in a CUDA graph (after warm-up rounds on a side stream)."""
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(2):
                self.start(prompt, first_token)
                self._round()
        torch.cuda.current_stream().wait_stream(warm)
        self.start(prompt, first_token)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._round()

    def generate(self, prompt: torch.Tensor, n_new: int, check_every: int = 1):
        """Greedy-decode at least n_new tokens; returns (tokens, rounds, accepted drafts)."""
        self.start(prompt)
        b = self.buf
        while True:
            for _ in range(check_every):
                self.graph.replay() if self.graph is not None else self._round()
            if int(b["n_out"]) >= n_new:
                break
        return b["out"][:n_new].clone(), int(b["rounds"]), int(b["accepted"])
