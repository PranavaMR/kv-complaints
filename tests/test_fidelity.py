
#Test 3: recorder fidelity. 

#all comments are made for my reference only

import torch

import config
from audit import RetentionRecorder
from kvhelpers import prefill
from run import make_press

PRESSES = ["random", "streaming_llm", "snapkv", "knorm"]


def test_recorder_fidelity(pipe, short_input):
    ids = short_input.context_ids
    ratio, seed = config.COMPRESSION_RATIO, config.SEED

    keys_full, _ = prefill(pipe, ids)                                    # 1.no press: every token
    for method in PRESSES:
        compressed_keys, _ = prefill(pipe, ids, make_press(method, ratio, seed))  # 2.bare press
        rec = RetentionRecorder(make_press(method, ratio, seed))
        prefill(pipe, ids, rec)                                          # 3.recorder, rec.kept
        # an empty rec.kept would make the loop below compare nothing and pass
        assert len(rec.kept) == pipe.model.config.num_hidden_layers, f"{method}: a layer was not recorded"

        for layer in rec.kept:                                          #4.compare, layer by layer
            idx = rec.kept[layer].to(keys_full[layer].device)            # (B, H_kv, n_kept)
            head_dim = keys_full[layer].shape[-1]
            gidx = idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            assert torch.equal(keys_full[layer].gather(2, gidx), compressed_keys[layer]), \
                f"{method} layer {layer}: recorded indices do not reproduce the compressed keys"
