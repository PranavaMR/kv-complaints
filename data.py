"""
Context builder: 12 Yelp restaurants -> 48 contexts (3 planted + 1 control each).

    python data.py   ->  results/manifest.json, data/review_texts.json, logs/data_build.log
"""
import datetime as dt
import json
import random
import statistics
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import config
from detect import mentions_aspect
from lexicon import LEXICON
from templates import FILLERS, TEMPLATES


@dataclass(frozen=True)
class ReviewRef:
    review_id: str | None      # None for a synthetic review
    stars: float
    date: str
    n_tokens: int
    synthetic: bool


@dataclass(frozen=True)
class Context:
    context_id: str
    business_id: str
    condition: Literal["plant", "control"]
    plant_aspect: str | None
    plant_position: int | None
    template_id: int | None
    reviews: tuple[ReviewRef, ...]


@dataclass
class Rendered:
    context_text: str
    context_ids: "torch.LongTensor"          # (1, n_context_tokens)
    spans: list[tuple[int, int]]             # [start, end) of each review
    prefix_span: tuple[int, int]             # chat-template tokens before Review 1


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.MODEL_NAME)


def n_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


COMMON_ENGLISH = {"the", "and", "was", "is", "to", "a", "of", "it", "for", "with", "we", "i", "my", "but"}


def looks_english(text: str) -> bool:
    """At least 97% ASCII and at least 3 very common English words."""
    ascii_share = sum(ch.isascii() for ch in text) / max(1, len(text))
    return ascii_share >= 0.97 and len(set(text.lower().split()) & COMMON_ENGLISH) >= 3


def mentions_any_aspect(text: str) -> bool:
    return any(mentions_aspect(text, aspect) for aspect in LEXICON)


def complaint_text(aspect: str, template_id: int, n_fillers: int) -> str:
    return " ".join([TEMPLATES[aspect][template_id]] + FILLERS[:n_fillers])


def make_complaint(aspect: str, template_id: int, target_tokens: float, tokenizer) -> int:
    """Token count of the complaint whose filler count gets closest to target_tokens (within +-15%)."""
    lengths = [n_tokens(complaint_text(aspect, template_id, n), tokenizer) for n in range(len(FILLERS) + 1)]
    length = min(lengths, key=lambda l: abs(l - target_tokens))
    low, high = (1 - config.LENGTH_TOLERANCE) * target_tokens, (1 + config.LENGTH_TOLERANCE) * target_tokens
    if not low <= length <= high:
        raise ValueError(f"Complaint {aspect}/{template_id} has {length} tokens; need {low:.0f}-{high:.0f}. "
                         "Shorten the template or add fillers in templates.py.")
    return length


def random_date(dates: list[str], rng: random.Random) -> str:
    first, last = (dt.date.fromisoformat(d) for d in (min(dates), max(dates)))
    return (first + dt.timedelta(days=rng.randint(0, (last - first).days))).isoformat()


def check_templates() -> None:
    """Each template hits its own aspect's lexicon and no other; fillers hit none."""
    for aspect, templates in TEMPLATES.items():
        for t, text in enumerate(templates):
            assert mentions_aspect(text, aspect), f"Template {aspect}/{t} has no '{aspect}' lexicon term"
            for other in LEXICON:
                assert other == aspect or not mentions_aspect(text, other), f"Template {aspect}/{t} matches '{other}'"
    for filler in FILLERS:
        assert not mentions_any_aspect(filler), f"Filler matches an aspect: {filler}"


def load_candidate_restaurants(yelp_dir: Path, seed: int) -> list[str]:
    """Open restaurants with >= 40 reviews, in a seeded random order."""
    ids = []
    with open(yelp_dir / config.BUSINESS_FILE, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if (b.get("is_open") == 1 and "Restaurants" in (b.get("categories") or "")
                    and b["review_count"] >= config.MIN_ENGLISH_REVIEWS):
                ids.append(b["business_id"])
    ids.sort()
    random.Random(seed).shuffle(ids)
    return ids[: config.N_CANDIDATES]


def load_reviews(yelp_dir: Path, business_ids: list[str]) -> dict[str, list[dict]]:
    """One pass over the 5 GB review file, keeping only the candidates' reviews."""
    reviews = {b: [] for b in business_ids}
    with open(yelp_dir / config.REVIEW_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["business_id"] in reviews:
                reviews[r["business_id"]].append({"review_id": r["review_id"], "stars": float(r["stars"]),
                                                  "date": r["date"][:10], "text": " ".join(r["text"].split())})
    return reviews


def build_contexts(yelp_dir: Path, seed: int) -> list[Context]:
    check_templates()
    tokenizer = get_tokenizer()
    candidates = load_candidate_restaurants(Path(yelp_dir), seed)
    reviews_by_business = load_reviews(Path(yelp_dir), candidates)

    contexts, texts, log = [], {}, []
    for business_id in candidates:
        if len(contexts) == 4 * config.N_RESTAURANTS:
            break
        i = len(contexts) // 4

        usable = []
        for r in reviews_by_business[business_id]:
            if looks_english(r["text"]):
                length = n_tokens(r["text"], tokenizer)
                if config.MIN_TOKENS <= length <= config.MAX_TOKENS:
                    usable.append({**r, "n_tokens": length})
        if len(usable) < config.MIN_ENGLISH_REVIEWS:
            log.append(f"skip {business_id}: only {len(usable)} usable English reviews")
            continue

        usable.sort(key=lambda r: r["review_id"])
        rng = random.Random(f"{seed}-{business_id}")
        sampled = rng.sample(usable, config.N_REVIEWS)

        # Every restaurant gets all three aspects, so no sampled review may mention any of them.
        if any(mentions_any_aspect(r["text"]) for r in sampled):
            log.append(f"replace {business_id}: a sampled review already mentions an aspect")
            continue

        organic = [ReviewRef(r["review_id"], r["stars"], r["date"], r["n_tokens"], False) for r in sampled]
        texts.update({r["review_id"]: r["text"] for r in sampled})
        median_len = statistics.median(ref.n_tokens for ref in organic)
        dates = [ref.date for ref in organic]

        for p, slot in enumerate(config.PLANT_SLOTS):
            aspect, template_id = config.ASPECTS[(i + p) % 3], i % 2
            length = make_complaint(aspect, template_id, median_len, tokenizer)
            planted = ReviewRef(None, config.PLANT_STARS, random_date(dates, rng), length, True)
            order = organic[: slot - 1] + [planted] + organic[slot - 1: 29]
            contexts.append(Context(f"r{i:02d}_p{slot:02d}", business_id, "plant",
                                    aspect, slot, template_id, tuple(order)))

        order = organic[:14] + [organic[29]] + organic[14:29]     # control: organic review 30 in slot 15
        contexts.append(Context(f"r{i:02d}_control", business_id, "control", None, None, None, tuple(order)))
        log.append(f"accept {business_id} as restaurant {i}")
        print(log[-1])

    log_path = config.ROOT / "logs" / "data_build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    if len(contexts) < 4 * config.N_RESTAURANTS:
        raise RuntimeError(f"Only {len(contexts) // 4} eligible restaurants; raise N_CANDIDATES in config.py")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.TEXT_CACHE.write_text(json.dumps(texts))
    return contexts


def save_manifest(contexts: list[Context], path: Path) -> None:
    """IDs, positions, aspects, templates, token counts and the synthetic texts. No raw review text."""
    tokenizer = get_tokenizer()
    rows = []
    for ctx in contexts:
        row = asdict(ctx)
        if ctx.condition == "plant":
            row["synthetic_text"] = synthetic_text_for(ctx, tokenizer)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": config.SEED, "model_tokenizer": config.MODEL_NAME,
                                "contexts": rows}, indent=1))


def load_manifest(path: Path = config.MANIFEST) -> list[Context]:
    rows = json.loads(Path(path).read_text())["contexts"]
    return [Context(row["context_id"], row["business_id"], row["condition"], row["plant_aspect"],
                    row["plant_position"], row["template_id"],
                    tuple(ReviewRef(**ref) for ref in row["reviews"])) for row in rows]


def synthetic_text_for(ctx: Context, tokenizer) -> str:
    """Rebuild the planted text: the filler count whose length matches the recorded n_tokens."""
    target = ctx.reviews[ctx.plant_position - 1].n_tokens
    for n in range(len(FILLERS) + 1):
        text = complaint_text(ctx.plant_aspect, ctx.template_id, n)
        if n_tokens(text, tokenizer) == target:
            return text
    raise ValueError(f"Cannot rebuild the planted review of {ctx.context_id}; were the templates edited?")


@lru_cache(maxsize=1)
def organic_texts() -> dict[str, str]:
    return json.loads(config.TEXT_CACHE.read_text())


def render(ctx: Context, tokenizer) -> Rendered:
    """
    The exact tokens kvpress compresses, and each review's token span.
    Mirrors kvpress 0.5.5 preprocessing: chat template up to the end of the user turn;
    the question is appended after compression.
    """
    import torch

    lines = []
    for k, ref in enumerate(ctx.reviews, start=1):
        text = synthetic_text_for(ctx, tokenizer) if ref.synthetic else organic_texts()[ref.review_id]
        lines.append(f"Review {k} (★{ref.stars:g}, {ref.date}): {text}")
    block = "\n\n".join(lines)

    separator = "#" * (len(block) + 10)
    templated = tokenizer.apply_chat_template([{"role": "user", "content": block + separator}],
                                              add_generation_prompt=True, tokenize=False)
    templated_context = templated.split(separator)[0]
    encoded = tokenizer(templated_context, add_special_tokens=False, return_offsets_mapping=True)    ids, token_starts = encoded["input_ids"], [start for start, _ in encoded["offset_mapping"]]

    # A token belongs to the review whose line contains the token's first character.
    char_starts, pos = [], templated_context.index(block)
    for line in lines:
        char_starts.append(pos)
        pos += len(line) + 2
    starts = [next(t for t, s in enumerate(token_starts) if s >= c) for c in char_starts]
    ends = starts[1:] + [len(ids)]
    return Rendered(block, torch.tensor([ids], dtype=torch.long), list(zip(starts, ends)), (0, starts[0]))

if __name__ == "__main__":
    built = build_contexts(config.YELP_DIR, config.SEED)
    save_manifest(built, config.MANIFEST)
    print(f"Built {len(built)} contexts -> {config.MANIFEST}")
