"""Rule-based detectors that score the model's answers."""
import re

from config import N_RECOMMENDATIONS
from lexicon import LEXICON

_ASPECT_REGEX = {aspect: re.compile(r"\b(?:" + "|".join(f"(?:{t})" for t in terms) + r")\b", re.IGNORECASE)
                 for aspect, terms in LEXICON.items()}


def mentions_aspect(text: str, aspect: str) -> bool:
    return _ASPECT_REGEX[aspect].search(text) is not None


def keyword_mention(advice: str, aspect: str) -> bool:
    return mentions_aspect(advice, aspect)


def cited_reviews(advice: str) -> list[int]:
    """Numbers in any [...] containing 'Review': [Review 3], [Review 3, 7], [Reviews 3 and 7]."""
    numbers = []
    for inside in re.findall(r"\[([^\]]*)\]", advice):
        if re.search(r"review", inside, re.IGNORECASE):
            numbers += [int(n) for n in re.findall(r"\d+", inside)]
    return numbers


def citation_mention(advice: str, planted_position: int) -> bool:
    return planted_position in cited_reviews(advice)


def q2_yes(answer: str, aspect: str) -> bool:
    """Starts with yes, and the rest quotes a lexicon term."""
    match = re.match(r"^\W*yes\b", answer.strip(), re.IGNORECASE)
    return match is not None and mentions_aspect(answer.strip()[match.end():], aspect)


def numbered_items(advice: str) -> list[int]:
    return [int(m) for m in re.findall(r"^\s*\**\s*(\d+)[.)]", advice, re.MULTILINE)]


def well_formed(advice: str, n_reviews: int = 30, n_items: int = N_RECOMMENDATIONS) -> bool:
    """Items numbered 1..n_items, at least one citation, every citation within 1..n_reviews."""
    citations = cited_reviews(advice)
    return (numbered_items(advice) == list(range(1, n_items + 1))
            and len(citations) > 0 and all(1 <= k <= n_reviews for k in citations))
