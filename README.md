# Who gets heard in the cache?

Does KV-cache eviction (25% kept) drop a planted customer complaint before Qwen2.5 writes
advice for a restaurant owner? 12 Yelp restaurants, 48 contexts, 5 methods, 240 runs.

## Run it

On your laptop (needs only `pip install "transformers==4.57.*"`), with the two Yelp JSON files in `yelp/`:

```bash
python data.py          # 48 contexts -> results/manifest.json, data/review_texts.json
```

Then upload this folder (without `yelp/`) to `MyDrive/kv-complaints` and run `colab_runbook.ipynb`
on a T4. It runs, in order:

```bash
pytest -q -rs tests         # the four required tests
python run.py --calibrate   # Full cache, 12 planted contexts, Q1 only
python run.py               # 240 runs -> data/runs_raw.jsonl (resumable)
python redact.py            # -> results/runs.jsonl (committable)
python labels.py            # -> labels/to_label.csv for blind hand labels
python analysis.py          # every table and figure, from saved results
```

## Files

| File | What it does |
|---|---|
| `config.py` | Every path and setting. |
| `lexicon.py`, `templates.py` | Keyword lexicon and SYNTHETIC complaint templates. Frozen before any output. |
| `data.py` | Picks restaurants and reviews, plants complaints, renders the exact tokens kvpress compresses, with each review's token span. |
| `audit.py` | `RetentionRecorder`: wraps a press, records the tokens each layer and KV head kept; `retention_by_review`. |
| `detect.py` | Keyword, citation, Q2 and well-formed detectors. |
| `run.py` | The runs, one JSONL line each, plus calibration. |
| `redact.py` | Replaces any 60+ character copy of a real review in the answers before they are committed. |
| `labels.py` | Blind, shuffled 36-row labelling sheet. |
| `analysis.py` | Rates with restaurant-bootstrap intervals, paired drops, false positives, retention, well-formed share, failure decomposition, verdict numbers, two figures. |
| `tests/` | The four required tests, with shared fixtures (`conftest.py`) and a prefill helper (`kvhelpers.py`). |
| `logs/` | `decisions.md` (calibration, primary detector, deviations), `data_build.log` (skipped and replaced restaurants). |

## Choices to check against the pre-registration

- **Model:** Qwen2.5-1.5B-Instruct in float16 on a Colab T4, the task's Colab option. Colab runs Python 3.12, not 3.11.
- **Eligibility:** every restaurant gets all three aspects, so a restaurant is replaced if any sampled review mentions any aspect.
- **English:** at least 97% ASCII characters and at least 3 very common English words.
- **Q2 on controls:** asked about the aspect that restaurant gets at slot 15; a "yes" is a Q2 false positive.
- **Hand labels:** every row gets all three aspect columns, so planted rows and controls look alike while labelling.
- **Verdict numbers:** count StreamingLLM, SnapKV and Knorm as eviction methods; Random is the control.
- **Raw answers** stay in the git-ignored `data/`, because they can quote real reviews.

## AI use

TO FILL IN before submitting: which tools you used and for what. For example: pipeline code
drafted with Claude (Anthropic), which also reviewed the tests; say who wrote `tests/conftest.py`
and `tests/kvhelpers.py`, and which parts are your own.
