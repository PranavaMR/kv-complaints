"""Prefill helper shared by tests 2 and 3 (not a test file)."""
import contextlib

import torch
from transformers import DynamicCache


def cache_kv(cache, layer: int):
    """(keys, values) of one layer, each (B, H_kv, seq, head_dim)."""
    if hasattr(cache, "layers"):                  # transformers >= 4.56
        return cache.layers[layer].keys, cache.layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


@torch.no_grad()
def prefill(pipe, input_ids, press=None):
    """
    One prefill done the way the kvpress pipeline does it:
        with press(model): model.model(input_ids=..., past_key_values=DynamicCache())
    Returns lists of per-layer keys and values read back out of the cache.
    """
    cache = DynamicCache()
    hooks = press(pipe.model) if press is not None else contextlib.nullcontext()
    with hooks:
        pipe.model.model(input_ids=input_ids.to(pipe.model.device), past_key_values=cache)
    n_layers = pipe.model.config.num_hidden_layers
    keys = [cache_kv(cache, l)[0].clone() for l in range(n_layers)]
    values = [cache_kv(cache, l)[1].clone() for l in range(n_layers)]
    return keys, values
