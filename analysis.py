"""
Every table and figure, from the saved results.

    python analysis.py   ->  results/tables/*.csv, results/figures/*.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from detect import citation_mention, keyword_mention, q2_yes, well_formed

METHOD_NAMES = {"full": "Full", "random": "Random", "streaming_llm": "StreamingLLM",
                "snapkv": "SnapKV", "knorm": "Knorm"}
EVICTION_METHODS = [m for m in ["streaming_llm", "snapkv", "knorm"] if m in config.METHODS]
POSITIONS = list(config.PLANT_SLOTS)
COLORS = {"full": "#6b6a66", "random": "#2a78d6", "streaming_llm": "#eb6834",
          "snapkv": "#1baf7a", "knorm": "#6250d6"}


def report(title: str, table: pd.DataFrame, name: str) -> None:
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES_DIR / f"{name}.csv", index=False)
    print(f"\n== {title}\n{table.to_string(index=False)}")


def combine(keyword: bool, citation: bool, rule: str = config.PRIMARY_DETECTOR) -> bool:
    return {"keyword": keyword, "citation": citation,
            "either": keyword or citation, "both": keyword and citation}[rule]


def load_runs() -> pd.DataFrame:
    df = pd.DataFrame([json.loads(l) for l in config.RUNS_FILE.read_text().splitlines() if l.strip()])
    df["restaurant"] = df.context_id.str[1:3].astype(int)
    df["plant_position"] = df.plant_position.astype("Int64")
    planted = df.condition == "plant"
    df["kw"] = [keyword_mention(a, asp) if p else None for a, asp, p in zip(df.q1_answer, df.plant_aspect, planted)]
    df["cite"] = [citation_mention(a, int(pos)) if p else None for a, pos, p in zip(df.q1_answer, df.plant_position, planted)]
    df["q2"] = [q2_yes(a, asp) if a is not None else None for a, asp in zip(df.q2_answer, df.q2_aspect)]
    df["wf"] = df.q1_answer.map(well_formed)
    df["mention"] = [combine(k, c) if p else None for k, c, p in zip(df.kw, df.cite, planted)]
    return df


def bootstrap(df: pd.DataFrame, value: str) -> tuple[float, float, float]:
    """Mean with a 95% interval from resampling restaurants (runs of one restaurant are related)."""
    per = df.groupby("restaurant")[value].agg(["sum", "count"])
    sums, counts = per["sum"].to_numpy(float), per["count"].to_numpy(float)
    draws = np.random.default_rng(config.SEED).integers(0, len(per), size=(config.N_BOOTSTRAP, len(per)))
    stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return sums.sum() / counts.sum(), np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def prf(truth: pd.Series, pred: pd.Series) -> dict:
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}


def detector_validation(df: pd.DataFrame) -> None:
    if not (config.LABELS_DIR / "labels.csv").exists():
        print("\n(no labels/labels.csv yet: skipping detector validation)")
        return
    labels = pd.read_csv(config.LABELS_DIR / "labels.csv").merge(pd.read_csv(config.LABELS_DIR / "key.csv"), on="label_id")
    joined = labels.merge(df, on="run_id")

    planted = joined[joined.condition == "plant"]
    truth = pd.Series([bool(r[r.plant_aspect]) for _, r in planted.iterrows()], index=planted.index)
    rows = []
    for rule in ["keyword", "citation", "either", "both"]:
        pred = pd.Series([combine(k, c, rule) for k, c in zip(planted.kw, planted.cite)], index=planted.index)
        rows.append({"detector": rule, "rows": f"{len(planted)} planted", **prf(truth, pred)})

    controls = joined[joined.condition == "control"]
    c_truth = pd.Series([bool(r[a]) for _, r in controls.iterrows() for a in config.ASPECTS])
    c_pred = pd.Series([keyword_mention(r.q1_answer, a) for _, r in controls.iterrows() for a in config.ASPECTS])
    rows.append({"detector": "keyword", "rows": f"{len(controls)} controls x 3 aspects", **prf(c_truth, c_pred)})
    report(f"Detector validation (primary in config.py: {config.PRIMARY_DETECTOR})", pd.DataFrame(rows),
           "detector_validation")


def mention_rates(planted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in config.METHODS:
        m = planted[planted.press == method]
        for pos in POSITIONS + ["all"]:
            part = m if pos == "all" else m[m.plant_position == pos]
            mean, lo, hi = bootstrap(part, "mention")
            rows.append({"method": METHOD_NAMES[method], "position": pos,
                         "count": f"{int(part.mention.sum())} of {len(part)}",
                         "rate": round(mean, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    table = pd.DataFrame(rows)
    report(f"Q1 mention rate ({config.PRIMARY_DETECTOR} detector); compare each method with Random here",
           table, "mention_rates")
    return table


def paired_drops(planted: pd.DataFrame) -> pd.DataFrame:
    """Method minus Full, on exactly the same contexts."""
    full = planted[planted.press == "full"].set_index("context_id")["mention"].astype(float)
    rows = []
    for method in config.METHODS[1:]:
        m = planted[planted.press == method].copy()
        m["diff"] = m.mention.astype(float).to_numpy() - full.loc[m.context_id].to_numpy()
        for pos in POSITIONS + ["all"]:
            mean, lo, hi = bootstrap(m if pos == "all" else m[m.plant_position == pos], "diff")
            rows.append({"method": METHOD_NAMES[method], "position": pos,
                         "difference": round(mean, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
    table = pd.DataFrame(rows)
    report("Paired difference vs Full (method minus Full)", table, "paired_drop_vs_full")
    return table


def false_positives(df: pd.DataFrame) -> None:
    controls = df[df.condition == "control"]
    rows = []
    for method in config.METHODS:
        c = controls[controls.press == method]
        fires = [keyword_mention(a, asp) for a in c.q1_answer for asp in config.ASPECTS]
        rows.append({"method": METHOD_NAMES[method], "keyword_fires": f"{sum(fires)} of {len(fires)}",
                     "fp_rate": round(np.mean(fires), 3), "q2_yes": f"{int(c.q2.sum())} of {len(c)}"})
    report("False positives on controls (nothing planted)", pd.DataFrame(rows), "false_positives")


def retention_table(planted: pd.DataFrame) -> pd.DataFrame:
    table = (planted[planted.press != "full"]
             .groupby(["press", "plant_position"])["retention_planted"].mean().unstack().round(3))
    table = table.loc[[m for m in config.METHODS if m in table.index]]
    table.index = [METHOD_NAMES[m] for m in table.index]
    report("Mean retention r of the planted review, by slot", table.reset_index(names="method"), "retention")
    return table


def well_formed_table(df: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame([{"method": METHOD_NAMES[m], "well_formed": f"{int(df[df.press == m].wf.sum())} of {(df.press == m).sum()}",
                           "share": round(df[df.press == m].wf.mean(), 3)} for m in config.METHODS])
    report("Well-formed Q1 answers", table, "well_formed")
    return table


def failure_decomposition(planted: pd.DataFrame) -> None:
    """Misses (Full mentioned it, the method did not), split by retention r and Q2."""
    full = planted[planted.press == "full"].set_index("context_id")["mention"]
    rows = []
    for method in config.METHODS[1:]:
        m = planted[planted.press == method]
        misses = m[full.loc[m.context_id].to_numpy().astype(bool) & ~m.mention.astype(bool).to_numpy()]
        kept, q2 = misses.retention_planted >= config.TAU, misses.q2.astype(bool)
        rows.append({"method": METHOD_NAMES[method], "misses": len(misses),
                     "retained_deprioritised": int((kept & q2).sum()), "retained_unreadable": int((kept & ~q2).sum()),
                     "evicted_recovered": int((~kept & q2).sum()), "evicted": int((~kept & ~q2).sum())})
    report(f"Failure decomposition (tau = {config.TAU})", pd.DataFrame(rows), "failure_decomposition")


def verdict_check(drops: pd.DataFrame, wf: pd.DataFrame) -> None:
    """The pre-registered criteria as numbers. The verdict itself is yours to write."""
    overall = drops[drops.position == "all"].set_index("method")["difference"]
    share = wf.set_index("method")["share"]
    print("\n== Verdict criteria (numbers only)")
    meeting = 0
    for name in [METHOD_NAMES[m] for m in EVICTION_METHODS]:
        drop, gap = -overall[name], abs(share[name] - share["Full"])
        meeting += drop >= 0.10 and gap <= 0.05
        print(f"{name}: drop vs Full {drop * 100:+.1f} points, well-formed gap {gap * 100:.1f} points")
    print(f"Eviction methods meeting both: {meeting} (criterion: at least 2; Random not counted)")


def save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_mention(rates: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, method in zip(np.linspace(-0.3, 0.3, len(config.METHODS)), config.METHODS):
        part = rates[(rates.method == METHOD_NAMES[method]) & (rates.position != "all")]
        ax.errorbar(np.arange(len(POSITIONS)) + offset, part.rate, yerr=[part.rate - part.ci_low, part.ci_high - part.rate],
                    fmt="o", ms=7, lw=2, capsize=3, color=COLORS[method], label=METHOD_NAMES[method])
    ax.set_xticks(range(len(POSITIONS)), [f"Review {p}" for p in POSITIONS])
    ax.set_xlabel("Where the complaint was planted")
    ax.set_ylabel(f"Mention rate in Q1 advice\n({config.PRIMARY_DETECTOR} detector, n = {config.N_RESTAURANTS} each)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Does the planted complaint reach the advice? (25% of the KV cache kept)", fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)
    fig.text(0.01, 0.005, "Bars: 95% interval, 2,000 bootstrap draws over restaurants.", fontsize=8, color="#52514e")
    save(fig, "fig1_mention_rate.png")


def figure_retention(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    image = ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), [f"Review {p}" for p in table.columns])
    ax.set_yticks(range(len(table.index)), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="white" if v > 0.55 else "#0b0b0b")
    ax.set_xlabel("Where the complaint was planted")
    ax.set_title("Planted review's retention r\n(mean kept share over layers and KV heads)", fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.04)
    save(fig, "fig2_retention.png")


def main():
    df = load_runs()
    planted = df[df.condition == "plant"]
    detector_validation(df)
    rates = mention_rates(planted)
    drops = paired_drops(planted)
    false_positives(df)
    retention = retention_table(planted)
    wf = well_formed_table(df)
    failure_decomposition(planted)
    verdict_check(drops, wf)
    figure_mention(rates)
    figure_retention(retention)
    print(f"\nTables in {config.TABLES_DIR}, figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
