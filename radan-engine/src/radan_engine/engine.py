"""The Radan engine's public API: load a model once, then generate exactly the model's own greedy text, fast.

    from radan_engine import Engine
    eng = Engine.load("path/to/Qwen3-4B-Base")          # 8-bit target + its own 4-bit draft, fused step
    out = eng.generate("The history of", max_new_tokens=128)
    print(out.text, out.tokens_per_s)

What `load` picks per family (override with `weights=` / `draft=`):

    family             weights (default)                     draft (default)
    Qwen3 / Llama      Q8_0 (llama.cpp's format, exact)       none; opt in to its own INT4 copy (draft="int4",
                                                              draft_codes=..., k = 6): faster only on text it predicts well
    Gemma 4 (dense)    Q8_0                                   none (its INT4 draft is not exact on Gemma)
    Qwen3.5 / 3.8      the checkpoint's packed INT4 as is     its MTP layer with an INT4 draft head, k = 4

Every setting returns the model's own greedy output for the weights it runs (for Q8_0: the Q8_0 model's);
the drafts only change speed. Batch size 1, greedy decoding, one CUDA GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

FAMILIES = {"qwen3": "qwen3", "llama": "qwen3", "gemma4": "gemma", "gemma4_text": "gemma"}


@dataclass
class Generation:
    tokens: list[int]
    text: str | None
    tokens_per_s: float
    prompt_tokens: int
    setting: str
    accept_rate: float | None = None
    info: dict = field(default_factory=dict)


def _family(config: dict) -> str:
    mt = config.get("model_type", "")
    if mt.startswith("qwen3_5") or mt.startswith("qwen3_8"):
        return "qwen35"
    text = config.get("text_config", {}).get("model_type", "")
    if mt.startswith("gemma4") or text.startswith("gemma4"):
        return "gemma"
    for key in (mt, text):
        if key in FAMILIES:
            return FAMILIES[key]
    raise ValueError(f"unsupported model_type {mt!r} (Radan engine runs Qwen3 / Llama, Gemma 4 dense, Qwen3.5 / 3.8)")


def _is_packed(config: dict) -> bool:
    q = config.get("quantization_config") or config.get("text_config", {}).get("quantization_config") or {}
    return q.get("quant_method") == "compressed-tensors" or q.get("format") == "pack-quantized"


class Engine:
    def __init__(self, model, tokenizer, dec, family: str, path: str, weights: str, draft: str, k: int, native_positions: int | None):
        self.model, self.tokenizer, self.dec = model, tokenizer, dec
        self.family, self.path, self.weights, self.draft, self.k = family, path, weights, draft, k
        self.native_positions = native_positions
        self._mtp = None

    # ---- loading ---------------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str, weights: str = "auto", draft: str = "auto", k: int | None = None,
             max_context: int = 4096, device: str = "cuda", draft_codes: str | None = None) -> "Engine":
        """weights: "auto" | "q8" (Q8_0 target) | "native" (the checkpoint's own BF16 or packed INT4).
        draft: "auto" | "none" | "int4" (Qwen3 / Llama self-draft) | "mtp" (Qwen3.5 / 3.8).
        draft_codes: saved INT4 codes of a calibrated (GPTQ + clipping, groups of 32) copy of the model, from
        the research repo's scripts/save_draft_codes.py. The INT4 self-draft pays off only with such a draft:
        a round-to-nearest copy (draft="int4" without codes) is accepted too rarely to be faster."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from radan_engine.gemma4_decode import make_decoder
        cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        family = _family(cfg)
        if _is_packed(cfg):
            from radan_engine.quant.packed import load_packed
            model, tokenizer, _ = load_packed(path, device, torch.bfloat16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        if weights == "auto":
            weights = "native" if family == "qwen35" else "q8"
        if draft == "auto":
            draft = {"qwen3": "none", "gemma": "none", "qwen35": "mtp"}[family]   # Qwen3 int4 draft: opt-in (text-dependent)
        if k is None:
            k = {"int4": 6, "mtp": 4}.get(draft, 0)
        with torch.inference_mode():
            dec = make_decoder(model, max_context)
            dec.attn_impl, dec.rope_impl, dec.step_attn = "fast", "fused", "cudnn"
            if family == "qwen3":
                if draft == "int4" and draft_codes:
                    dec.load_int4_codes(draft_codes)
                elif draft == "int4":   # built from the BF16 weights before they are replaced by Q8_0
                    dec.quantize_int4(group=32, method="rtn", clip=True)
                if weights == "q8":
                    dec.quantize_q8_target()
            elif family == "gemma":
                if draft not in ("none",):
                    raise ValueError("Gemma 4: no exact draft (its INT4 self-draft flips tokens); use draft='none'")
                if weights == "q8":
                    dec.quantize_q8_target(model, draft=False)
            elif weights == "q8":
                raise ValueError("Qwen3.5 / 3.8: the engine runs the checkpoint's own INT4 weights (weights='native')")
            dec.compile_step(True)
        tc = cfg.get("text_config", cfg)
        native = tc.get("max_position_embeddings") if family == "qwen3" else None
        eng = cls(model, tokenizer, dec, family, path, weights, draft, k, native)
        if draft == "mtp":
            from radan_engine.mtp_spec import MTPLayer
            with torch.inference_mode():
                eng._mtp = MTPLayer(dec, path, head="int4")
        return eng

    # ---- generation -------------------------------------------------------------------------------
    def encode(self, prompt, chat: bool = False) -> torch.Tensor:
        if isinstance(prompt, str):
            if chat:
                text = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                          add_generation_prompt=True)
                ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            else:
                ids = self.tokenizer(prompt)["input_ids"]
        else:
            ids = list(prompt)
        return torch.tensor(ids, dtype=torch.long, device=self.dec.dev)

    def _fit(self, need: int):
        d = self.dec
        if need <= d.max_len:
            return
        if self.family == "qwen3" and self.native_positions and need > self.native_positions:
            d.set_max_len(need, yarn=(4.0, self.native_positions))
        else:
            d.set_max_len(need)
        if self._mtp is not None:
            self._mtp.ensure_len()

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens: int = 256, chat: bool = False, draft: str | None = None,
                 stop_at_eos: bool = True) -> Generation:
        """prompt: text or token ids. draft: None = the engine's default ("auto" rules), or "none"."""
        ids = self.encode(prompt, chat)
        P, n = ids.shape[0], max_new_tokens
        use = self.draft if draft is None else draft
        if use == "int4" and P + n > 8192:   # measured: the 4-bit self-draft stops paying on 8-bit targets past ~8k
            use = "none"
        self._fit(P + n + 64 + 8 * (self.k + 1))
        chunk = 2048 if P + n < 65536 else 512
        if use == "mtp":
            toks, ms, made, acc = self._gen_mtp(ids, n, chunk)
        elif use == "int4":
            toks, ms, made, acc = self._gen_selfspec(ids, n, chunk)
        else:
            toks, ms, made, acc = self._gen_plain(ids, n, chunk)
        tps = 1000.0 * made / ms if ms > 0 else float("nan")
        if stop_at_eos:
            eos = set()
            for e in (getattr(self.tokenizer, "eos_token_id", None), getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)):
                if e is not None:
                    eos |= set(e) if isinstance(e, (list, tuple)) else {e}
            cut = next((i for i, t in enumerate(toks) if t in eos), None)
            toks = toks if cut is None else toks[:cut + 1]
        text = self.tokenizer.decode(toks, skip_special_tokens=True) if self.tokenizer is not None else None
        return Generation(toks, text, tps, P, f"{self.family}/{self.weights}/{use}" + (f"/k={self.k}" if use != "none" else ""), acc)

    def _prefill_first(self, ids, chunk):
        return int(self.dec.prefill_chunked(ids, chunk=chunk if self.family != "gemma" else min(chunk, 1024))[-1].argmax())

    @staticmethod
    def _timed(fn):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        r = fn()
        b.record()
        torch.cuda.synchronize()
        return r, a.elapsed_time(b)

    def _gen_plain(self, ids, n, chunk):
        d = self.dec
        first = self._prefill_first(ids, chunk)
        snap = d.snapshot()                 # capturing the step runs warm-up steps: recurrent states must be put back
        rep, rst, buf = d.make_step(ids.shape[0], first)
        d.restore(snap)
        rst()
        out = torch.empty(n, dtype=torch.long, device=d.dev)
        out[0] = first

        def run():
            for i in range(1, n):
                rep()
                out[i:i + 1].copy_(buf["tok"].view(-1)[:1])
        _, ms = self._timed(run)
        return out.tolist(), ms, n - 1, None

    def _gen_selfspec(self, ids, n, chunk):
        from radan_engine.self_spec import SelfSpecDecoder
        d = self.dec
        first = self._prefill_first(ids, chunk)
        snap = d.snapshot()
        ss = SelfSpecDecoder(d, self.k, (), (), max_new=n, draft_w4=True)
        ss.capture(ids, first)
        d.restore(snap)
        ss.start(ids, first)

        def run():
            while int(ss.buf["n_out"]) < n:
                ss.graph.replay()
        _, ms = self._timed(run)
        acc = int(ss.buf["accepted"]) / max(1, int(ss.buf["rounds"]) * self.k)
        made = int(ss.buf["n_out"]) - 1
        toks = ss.buf["out"][:n].tolist()
        del ss
        torch.cuda.empty_cache()
        return toks, ms, made, acc

    def _gen_mtp(self, ids, n, chunk):
        from radan_engine.mtp_spec import MTPSelfSpec
        d = self.dec
        ss = MTPSelfSpec(d, self._mtp, self.k, max_new=n)
        P = ids.shape[0]
        f1, hl = ss.prefill_long(ids, chunk=chunk)
        sm = d.snapshot()

        def reset():
            d.restore(sm)
            ss.reset(P, f1, hl)
        ss.capture(ids, reset)
        reset()

        def run():
            while int(ss.buf["n_out"]) < n:
                ss.graph.replay()
        _, ms = self._timed(run)
        acc = int(ss.buf["accepted"]) / max(1, int(ss.buf["rounds"]) * self.k)
        made = int(ss.buf["n_out"]) - 1
        toks = ss.buf["out"][:n].tolist()
        del ss
        torch.cuda.empty_cache()
        return toks, ms, made, acc
