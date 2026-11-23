"""Collect the results of every test into one summary table and plot."""
# Save as pipeline/collect_results.py. Run from the project root after the
# clustering and classification jobs:
#   pixi run python pipeline/collect_results.py
# Writes output/experiments/summary.csv and output/experiments/summary_f1.png.

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent / "output" / "experiments"


def main():
    paths = sorted(EXPERIMENT_ROOT.glob("*/*/results.json"))
    if not paths:
        raise SystemExit(f"No results.json files in {EXPERIMENT_ROOT}. Run the tests first.")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = pd.DataFrame(rows).drop(columns=["class_counts"], errors="ignore")
    summary["r_values"] = summary["r_values"].map(lambda values: " ".join(map(str, values)))
    summary = summary.sort_values(["task", "variant"])
    summary.to_csv(EXPERIMENT_ROOT / "summary.csv", index=False, float_format="%.4f")
    print(f"Saved {len(summary)} results to {EXPERIMENT_ROOT / 'summary.csv'}")

    if "f1_macro_mean" not in summary:
        return
    scores = summary[summary["task"] == "classification"].sort_values("f1_macro_mean")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(scores))))
    ax.barh(scores["variant"], scores["f1_macro_mean"], xerr=scores["f1_macro_std"],
            color="#1C3041", capsize=3)
    ax.set_xlabel("Macro F1 (mean ± std over 5 folds)")
    ax.set_xlim(0, 1)
    ax.axvline(1 / 3, color="grey", linestyle="--", linewidth=0.8, label="Random guess (3 classes)")
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title("Classification F1 per embedding variant")
    fig.tight_layout()
    fig.savefig(EXPERIMENT_ROOT / "summary_f1.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
