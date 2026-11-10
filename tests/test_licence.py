"""Test 4: licence guard. Pure Python, no GPU. Needs data/review_texts.json (from data.py)."""
import json
import subprocess

import config

WINDOW = 60


def has_60char_overlap(content: str, text: str) -> bool:
    """True if content contains any 60+ character substring of text.
    Any 60+ char match contains a 60-char window, so checking every window is enough."""
    return any(text[i:i + WINDOW] in content for i in range(len(text) - WINDOW + 1))


def _strings(obj):
    """Every string value inside parsed JSON (dicts and lists, any depth)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def searchable_text(path: str, content: str) -> str:
    """
    The text to search. For .json/.jsonl, the decoded string values: json.dumps writes
    " as \\" and ’ as \\u2019, so a raw search would miss a review containing them.
    If the file does not parse as JSON, fall back to the raw text (never skip it).
    """
    try:
        if path.endswith(".jsonl"):
            return "\n".join(s for line in content.splitlines() if line.strip()
                             for s in _strings(json.loads(line)))
        if path.endswith(".json"):
            return "\n".join(_strings(json.loads(content)))
    except ValueError:
        pass
    return content


def test_has_60char_overlap_fires():
    """Test the test: a guard that can never fire is worthless."""
    texts = json.loads(config.TEXT_CACHE.read_text(encoding="utf-8"))
    review = next(iter(texts.values()))
    # \x00 never occurs in a review, so it cannot stretch a match past the slice
    assert has_60char_overlap(f"\x00{review[20:80]}\x00", review)          # 60 chars: fires
    assert not has_60char_overlap(f"\x00{review[20:79]}\x00", review)      # 59 chars: does not
    # and a review copied into a .jsonl answer is still found after json.dumps escaping
    leaked_line = json.dumps({"q1_answer": f"1. {review[20:80]}"})
    assert has_60char_overlap(searchable_text("results/runs.jsonl", leaked_line), review)


def test_no_raw_review_text_in_git():
    texts = json.loads(config.TEXT_CACHE.read_text(encoding="utf-8"))   # review_id -> text
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=config.ROOT, text=True).splitlines()
    assert tracked, "git ls-files listed nothing: run pytest inside the repo"
    for path in tracked:
        if path.startswith("data/"):            # the git-ignored folder is exempt
            continue
        file = config.ROOT / path
        if not file.is_file():
            continue
        content = searchable_text(path, file.read_text(encoding="utf-8", errors="ignore"))
        for review_id, text in texts.items():
            assert not has_60char_overlap(content, text), f"{path} contains text from review {review_id}"
