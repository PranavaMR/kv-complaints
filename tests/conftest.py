"""
Shared fixtures for the four required tests.

    pytest tests/                      # everything (GPU tests need a CUDA GPU, e.g. Colab)
    pytest tests/test_licence.py       # the licence guard alone: pure Python, no GPU

Tests 1, 2, 3 and 4 need the built data (`python data.py`): results/manifest.json and
data/review_texts.json. If either is missing the tests FAIL (they do not skip), because
a skipped test is not a passing test.
"""
import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def require(path: Path, how: str) -> None:
    if not path.exists():
        pytest.fail(f"{path} is missing. {how}", pytrace=False)


# ---------------------------------------------------------------- CPU fixtures

@pytest.fixture(scope="session")
def tokenizer():
    from data import get_tokenizer
    return get_tokenizer()


@pytest.fixture(scope="session")
def contexts():
    require(config.MANIFEST, "Run `python data.py` first.")
    require(config.TEXT_CACHE, "Run `python data.py` first (it needs the Yelp files).")
    from data import load_manifest
    return load_manifest()


@pytest.fixture(scope="session")
def rendered_all(contexts, tokenizer):
    """(context, Rendered) for all 48 contexts."""
    from data import render
    return [(ctx, render(ctx, tokenizer)) for ctx in contexts]


@pytest.fixture(scope="session")
def review_texts() -> dict[str, str]:
    """review_id -> text of every organic review used in the study."""
    import json
    require(config.TEXT_CACHE, "Run `python data.py` first (it needs the Yelp files).")
    return json.loads(config.TEXT_CACHE.read_text())


# ---------------------------------------------------------------- GPU fixtures

@pytest.fixture(scope="session")
def pipe():
    """The same kvpress pipeline run.py uses (same model, dtype, attention implementation)."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU: run this on Colab before submitting")
    from run import load_pipe
    p = load_pipe()
    p.model.eval()
    return p


@pytest.fixture(scope="session")
def short_input(contexts, tokenizer):
    """
    A short REAL input: the first 4 reviews of the first context whose complaint is at
    slot 1, so it holds the planted review and 3 organic ones, rendered with the chat
    template exactly as run.py renders the full context.
    """
    from data import render
    ctx = next(c for c in contexts if c.condition == "plant" and c.plant_position == 1)
    short = dataclasses.replace(ctx, context_id=ctx.context_id + "_short", reviews=ctx.reviews[:4])
    return render(short, tokenizer)
