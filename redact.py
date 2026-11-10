"""
data/runs_raw.jsonl -> results/runs.jsonl, with any 60+ character copy of a real review
replaced by "[redacted review text]". Stops if redaction would change a detector result.

    python redact.py
"""
import json

import config
from data import load_manifest, organic_texts
from detect import citation_mention, keyword_mention, q2_yes, well_formed

WINDOW = 60
MARK = "[redacted review text]"


def redact(answer: str | None, review_texts: list[str]) -> str | None:
    if not answer:
        return answer
    windows = {t[i:i + WINDOW] for t in review_texts for i in range(len(t) - WINDOW + 1)}
    hit = [False] * len(answer)
    for i in range(len(answer) - WINDOW + 1):
        if answer[i:i + WINDOW] in windows:
            hit[i:i + WINDOW] = [True] * WINDOW
    out = []
    for i, ch in enumerate(answer):
        if not hit[i]:
            out.append(ch)
        elif i == 0 or not hit[i - 1]:
            out.append(MARK)
    return "".join(out)


def detector_results(row: dict) -> tuple:
    advice = row["q1_answer"]
    return (well_formed(advice), *[keyword_mention(advice, a) for a in config.ASPECTS],
            row["plant_position"] and citation_mention(advice, row["plant_position"]),
            row["q2_answer"] and q2_yes(row["q2_answer"], row["q2_aspect"]))


def main(src=config.RAW_RUNS_FILE, dst=config.RUNS_FILE):
    texts = organic_texts()
    reviews_of = {c.context_id: [texts[r.review_id] for r in c.reviews if not r.synthetic] for c in load_manifest()}
    n_redacted = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            before = detector_results(row)
            for field in ("q1_answer", "q2_answer"):
                new = redact(row[field], reviews_of[row["context_id"]])
                n_redacted += new != row[field]
                row[field] = new
            assert detector_results(row) == before, f"Redaction changed a detector result in {row['run_id']}"
            fout.write(json.dumps(row) + "\n")
    print(f"Wrote {dst}; redacted {n_redacted} answers.")


if __name__ == "__main__":
    main()
