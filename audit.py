"""
Retention recorder: which context tokens did each layer and KV head keep?

ScorerPress.compress in kvpress 0.5.5 does:
    scores  = press.score(...)                      # (B, H_kv, k_len)
    n_kept  = compute_n_kept(k_len, ratio)          # int(k_len * (1 - ratio))
    indices = scores.topk(n_kept, dim=-1).indices   # top-k per KV head
    keys, values = gathered at those indices
RetentionRecorder runs exactly these steps with the wrapped press's own score()
and saves the indices, so they ARE the eviction (Test 3 proves it).
"""
import numpy as np
import torch
from kvpress import BasePress, ScorerPress
from kvpress.utils import compute_n_kept


class RetentionRecorder(BasePress):
    def __init__(self, press: ScorerPress):
        assert isinstance(press, ScorerPress), "The recorder only supports ScorerPress subclasses"
        self.press = press
        self.kept: dict[int, torch.LongTensor] = {}    # layer_idx -> (B, H_kv, n_kept)

    def reset(self) -> None:
        self.kept = {}

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        ratio = self.press.compression_ratio
        if ratio == 0:
            return keys, values

        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        n_kept = compute_n_kept(keys.shape[2], ratio)
        indices = scores.topk(n_kept, dim=-1).indices
        self.kept[module.layer_idx] = indices.detach().cpu()

        gather_idx = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, gather_idx).contiguous(), values.gather(2, gather_idx).contiguous()


def kept_mask(kept: dict[int, torch.LongTensor], k_len: int) -> torch.BoolTensor:
    """(n_layers, H_kv, k_len): True where that layer and KV head kept the token (batch 0)."""
    layers = sorted(kept)
    mask = torch.zeros(len(layers), kept[layers[0]].shape[1], k_len, dtype=torch.bool)
    for row, layer in enumerate(layers):
        mask[row].scatter_(1, kept[layer][0], True)
    return mask


def retention_by_review(kept: dict[int, torch.LongTensor], spans: list[tuple[int, int]]) -> np.ndarray:
    """(n_reviews,): share of each review's tokens kept, averaged over layers and KV heads."""
    keep_share = kept_mask(kept, spans[-1][1]).float().mean(dim=(0, 1))     # per token
    return np.array([keep_share[start:end].mean().item() for start, end in spans])
