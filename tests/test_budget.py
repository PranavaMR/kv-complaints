#Test 2: exact budget. 
import config
from audit import RetentionRecorder
from kvhelpers import prefill
from run import make_press

PRESSES = ["random", "streaming_llm", "snapkv", "knorm"]


def test_exact_budget(pipe, short_input):
    ratio = config.COMPRESSION_RATIO
    ids = short_input.context_ids
    # k_len is the WHOLE context: chat-template prefix + reviews, not the review spans alone
    k_len = ids.shape[1]
    #spec formula
    expected = int(k_len * (1 - ratio))              
    n_layers = pipe.model.config.num_hidden_layers
    n_kv_heads = pipe.model.config.num_key_value_heads

    for method in PRESSES:
        #the real cache, compressed by the bare kvpress press
        keys, _ = prefill(pipe, ids, make_press(method, ratio, seed=config.SEED))
        for layer in range(n_layers):
            assert keys[layer].shape[2] == expected, \
                f"{method} layer {layer}: cache holds {keys[layer].shape[2]} tokens, expected {expected}"

        #(B, H_kv, n_kept)
        rec = RetentionRecorder(make_press(method, ratio, seed=config.SEED))
        prefill(pipe, ids, rec)                       #
        assert len(rec.kept) == n_layers, f"{method}: a layer was not compressed"
        for layer_idx, kept in rec.kept.items():
            assert kept.shape[1] == n_kv_heads, f"{method} layer {layer_idx}: {kept.shape[1]} KV heads"
            assert kept.shape[-1] == expected, f"{method} layer {layer_idx}: kept {kept.shape[-1]}, expected {expected}"
