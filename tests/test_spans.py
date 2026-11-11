#Test 1: span partition. Checks all 48 contexts (slots 1, 15, 30 and the controls).


def test_spans_contiguous(rendered_all):
    for ctx, r in rendered_all:
        for i in range(len(r.spans) - 1):
            assert r.spans[i][1] == r.spans[i + 1][0], f"{ctx.context_id}: gap or overlap after Review {i + 1}"


def test_spans_cover_context(rendered_all):
    for ctx, r in rendered_all:
        assert r.prefix_span == (0, r.spans[0][0]), ctx.context_id
        assert r.spans[-1][1] == r.context_ids.shape[1], ctx.context_id


def test_span_decodes_to_review_k(rendered_all, tokenizer):
    for ctx, r in rendered_all:
        for i, (start, end) in enumerate(r.spans):      #span index 0-30
            text = tokenizer.decode(r.context_ids[0, start:end])
            assert text.startswith(f"Review {i + 1} ("), f"{ctx.context_id}: span {i} decodes to {text[:30]!r}"
