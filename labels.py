"""
Blind labelling sheet: 30 planted Q1 outputs (6 per method) + 6 controls, shuffled.

    python labels.py  ->  labels/to_label.csv (label this)  +  labels/key.csv (don't open it)

For every row, put 1 in each aspect column the advice addresses, else 0.
Save the filled sheet as labels/labels.csv.
"""
import json
import random

import pandas as pd

import config


def main():
    runs = pd.DataFrame([json.loads(l) for l in config.RUNS_FILE.read_text().splitlines() if l.strip()])
    rng = random.Random(config.SEED)
    planted = runs[runs.condition == "plant"]
    picked = [r for m in config.METHODS for r in rng.sample(sorted(planted[planted.press == m].run_id), 6)]
    picked += rng.sample(sorted(runs[runs.condition == "control"].run_id), 6)
    rng.shuffle(picked)

    answers = runs.set_index("run_id").q1_answer
    sheet = pd.DataFrame({"label_id": [f"L{n:02d}" for n in range(len(picked))],
                          "advice": [answers[r] for r in picked],
                          **{a: "" for a in config.ASPECTS}, "notes": ""})
    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(config.LABELS_DIR / "to_label.csv", index=False)
    pd.DataFrame({"label_id": sheet.label_id, "run_id": picked}).to_csv(config.LABELS_DIR / "key.csv", index=False)
    print(f"Wrote {len(sheet)} rows to labels/to_label.csv. Fill it in and save it as labels/labels.csv.")


if __name__ == "__main__":
    main()
