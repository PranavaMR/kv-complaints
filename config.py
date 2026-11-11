"""Every path and setting. PRE-REGISTERED values must not change after you see outputs."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Yelp files: $YELP_DIR, else a yelp/ folder in this repo, else the Colab default.
YELP_DIR = Path(os.environ.get("YELP_DIR") or
                (ROOT / "yelp" if (ROOT / "yelp").exists() else "/content/drive/MyDrive/yelp"))
BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"

# Git-ignored: raw review text and raw model answers.
DATA_DIR = ROOT / "data"
TEXT_CACHE = DATA_DIR / "review_texts.json"
RAW_RUNS_FILE = DATA_DIR / "runs_raw.jsonl"
CALIBRATION_FILE = DATA_DIR / "calibration_raw.jsonl"

# Committed: no raw review text.
MANIFEST = ROOT / "results" / "manifest.json"
RUNS_FILE = ROOT / "results" / "runs.jsonl"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"
LABELS_DIR = ROOT / "labels"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
DTYPE = "float16"                 # a T4 has no bfloat16
DEVICE = "cuda:0"

SEED = 1234
N_RESTAURANTS = 12                # time cut 2: set to 9 (run.py then uses restaurants 0-8; no rebuild needed)
N_REVIEWS = 30
MIN_ENGLISH_REVIEWS = 40
MIN_TOKENS, MAX_TOKENS = 120, 260
N_CANDIDATES = 80                 # restaurants scanned to find 12 eligible ones
PLANT_SLOTS = (1, 15, 30)         # position index p = 0, 1, 2
PLANT_STARS = 2.0
LENGTH_TOLERANCE = 0.15
ASPECTS = ("access", "double_charge", "towed")   # order matters for the balancing rule

METHODS = ("full", "random", "streaming_llm", "snapkv", "knorm")   # time cut 1: remove "knorm"
COMPRESSION_RATIO = 0.75          # fraction REMOVED: 25% kept

N_RECOMMENDATIONS = 5             # PRE-REGISTERED calibration rule: may change once (to 8 or 3); log it
Q1_MAX_NEW_TOKENS = 512
Q2_MAX_NEW_TOKENS = 48

TAU = 0.2                         # PRE-REGISTERED
N_BOOTSTRAP = 2000
PRIMARY_DETECTOR = "keyword"      # PRE-REGISTERED rule decides: "keyword", "citation", "either" or "both"
