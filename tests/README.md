# The four required tests (yours to write)

The task rules say the tests must be your own work, so this folder is empty
apart from this note. Put them in `tests/test_*.py` and run `pytest -q`.
Each spec below says what to assert and which functions in this repo to use.

## Test 1: span partition (`data.render`)
For a few contexts from `data.load_manifest()`, `render(ctx, tokenizer)`:
- `spans` are contiguous: each `end` equals the next `start`
- `prefix_span == (0, spans[0][0])` and `spans[-1][1] == context_ids.shape[1]`
- `tokenizer.decode(context_ids[0, s:e])` starts with `f"Review {k} ("` for review k

## Test 2: exact budget (`audit.RetentionRecorder`)
Run each press wrapped in the recorder on one short real input.
For every layer in `recorder.kept`, the last dimension is
`int(k_len * (1 - ratio))`, where `k_len = context_ids.shape[1]`
(template tokens included). Check all four presses.

## Test 3: recorder fidelity (`audit.RetentionRecorder`)
On a short real input, for each press:
1. Prefill once with no press into a `DynamicCache` and keep `keys_full` per layer.
2. Prefill again with the bare press (not wrapped) and keep the compressed keys.
3. Prefill a third time with the recorder and read `recorder.kept`.
4. Assert `torch.equal(keys_full.gather(2, idx[..., None].expand(-1, -1, -1, head_dim)), compressed_keys)`.

Hint: prefill with `pipe.model.model(input_ids=ids, past_key_values=cache)` inside
`with press(pipe.model):`, as the pipeline does. Comparing against the BARE press in
step 2 is what proves the recorder changes nothing.

## Test 4: licence guard
Walk every git-tracked file (`git ls-files`) outside `data/`. Fail if any contains
a substring of 60+ characters of any review in `data/review_texts.json`.
