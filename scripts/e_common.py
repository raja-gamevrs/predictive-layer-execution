"""Shared pieces for experiments E2–E5 (docs/experiments_e1_e5.md): data splits, cached ridge
predictors, a gated speculation context and the quality metrics against the exact model."""

import math
from contextlib import contextmanager
from pathlib import Path

import torch

from ple.activation_capture import capture_residual_stream
from ple.data import wikitext_chunks
from ple.predictors.fitting import fit_linear
from ple.predictors.linear import LinearDeltaPredictor
from ple.speculation_policy import attention_input_linears

SEQ = 512
FIT = (0, 256)          # train chunks for predictor fitting / training
CAL = (400, 16)         # calibration
HOLD = (600, 16)        # holdout
HEAL = (1000, 1000)     # E3 healing data
CACHE = Path("results/raw/e1_e5")


def chunks(tok, span):
    skip, n = span
    return wikitext_chunks(tok, "train", SEQ, n, skip=skip)


def ridge(model, layers, tok, stem: str, dev) -> dict[int, LinearDeltaPredictor]:
    """Per-boundary ridge predictors h_l -> h_(l+1) fitted on FIT, cached on disk (FP32)."""
    path = CACHE / f"{stem}_ridge.pt"
    if path.exists():
        saved = torch.load(path, map_location=dev)
        return {l: LinearDeltaPredictor(w, b) for l, (w, b) in saved.items()}
    d = model.config.get_text_config().hidden_size

    def traces():
        for ids in chunks(tok, FIT):
            yield [s[0, 1:] for s in capture_residual_stream(model, layers, input_ids=ids[None].to(dev))]
    out = {}
    group = 12
    for i in range(0, len(layers) - 1, group):
        out |= fit_linear(traces(), range(i, min(i + group, len(layers) - 1)), d, dev)
        torch.cuda.empty_cache()
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({l: (p.weight, p.bias) for l, p in out.items()}, path)
    return out


@contextmanager
def gated_speculation(layers, boundaries, predictor, hidden_size, tau=None, stats=None, fallback=None,
                      branch=None):
    """Speculate block l+1's attention inputs from predictor[l](h_l), uncorrected, except for tokens whose
    relative error r = ||N(h) - N(ĥ)|| / ||N(h)|| exceeds `tau`, which get the exact projection
    (`fallback(l, r)` -> bool mask may replace the threshold rule). `branch(mod, x)` replaces the
    speculative projection's kernel (E5). Position 0 stays exact. `stats` collects speculated share."""
    handles, stash, decide = [], {}, {}

    def keep(l):
        def hook(_m, args, kwargs):
            stash[l] = args[0] if args else kwargs["hidden_states"]
        return hook

    def replace(l, first):
        def hook(mod, args, out):
            x = args[0]
            if first or l not in decide:
                n_hat = layers[l + 1].input_layernorm(predictor[l](stash[l])).to(x.dtype)
                r = (x.float() - n_hat.float()).norm(dim=-1) / x.float().norm(dim=-1).clamp(min=1e-6)
                use_exact = fallback(l, r) if fallback is not None else (
                    r > tau if tau is not None else torch.zeros_like(r, dtype=torch.bool))
                use_exact[:, 0] = True
                decide[l] = (n_hat, use_exact)
                if stats is not None:
                    stats["spec"] += (~use_exact[:, 1:]).sum().item()
                    stats["total"] += use_exact[:, 1:].numel()
                    stats.setdefault("r", []).append(r[:, 1:].flatten().cpu())
            n_hat, use_exact = decide[l]
            y_hat = branch(mod, n_hat) if branch is not None else mod.forward(n_hat)
            return torch.where(use_exact[..., None], out, y_hat.to(out.dtype))
        return hook

    def clear(l):
        def hook(_m, _a, _o):
            decide.pop(l, None)
        return hook

    try:
        for l in boundaries:
            handles.append(layers[l].register_forward_pre_hook(keep(l), with_kwargs=True))
            for j, lin in enumerate(attention_input_linears(layers[l + 1], hidden_size)):
                handles.append(lin.register_forward_hook(replace(l, j == 0)))
            handles.append(layers[l + 1].register_forward_hook(clear(l)))
        yield
    finally:
        for h in handles:
            h.remove()


class Quality:
    """Accumulate KL(p_exact || p), top-1 agreement and NLL over positions 1..T-2 (predicting 2..T-1)."""

    def __init__(self):
        self.kl = self.top1 = self.nll = self.n = 0.0

    @staticmethod
    def logprobs(logits):
        return torch.log_softmax(logits[0, 1:-1].float(), dim=-1)

    def add(self, lp_exact, lp, ids):
        t = ids[0, 2:]
        self.kl += (lp_exact.exp() * (lp_exact - lp)).sum().item()
        self.top1 += (lp.argmax(-1) == lp_exact.argmax(-1)).sum().item()
        self.nll += -lp.gather(-1, t[:, None]).sum().item()
        self.n += t.numel()

    def row(self, exact_ppl):
        ppl = math.exp(self.nll / self.n)
        return {"kl": self.kl / self.n, "top1": self.top1 / self.n, "ppl": ppl,
                "ppl_change_pct": 100 * (ppl / exact_ppl - 1)}


def exact_logprobs(model, tok, span, dev):
    ids_list = [ids[None].to(dev) for ids in chunks(tok, span)]
    with torch.inference_mode():
        lps = [Quality.logprobs(model(input_ids=ids).logits) for ids in ids_list]
    nll = sum(-lp.gather(-1, ids[0, 2:, None]).sum().item() for lp, ids in zip(lps, ids_list))
    n = sum(ids.shape[1] - 2 for ids in ids_list)
    return ids_list, lps, math.exp(nll / n)
