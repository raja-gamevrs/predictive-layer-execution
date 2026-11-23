"""Prompt corpora. WikiText-2 raw (CC BY-SA), pinned revision; train and test splits are disjoint."""

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

WIKITEXT, WIKITEXT_REV = "Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3"


def wikitext_chunks(tokenizer, split: str, seq_len: int, n: int, skip: int = 0) -> list[torch.Tensor]:
    """Consecutive non-overlapping chunks of `seq_len` tokens (BOS first if the tokenizer has one)."""
    path = hf_hub_download(WIKITEXT, f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                           repo_type="dataset", revision=WIKITEXT_REV)
    text = "".join(pq.read_table(path).column("text").to_pylist())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    body = seq_len - len(bos)
    out = [torch.tensor(bos + ids[i:i + body]) for i in range(0, len(ids) - body, body)][skip:skip + n]
    if len(out) < n:
        raise ValueError(f"{split}: only {len(out)} chunks of {seq_len} tokens after skipping {skip}")
    return out
