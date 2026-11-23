"""Classify cities as low, medium or high accessibility for each embedding variant."""
# Save as supervised_learning/classification.py and run from the project root:
#   pixi run python supervised_learning/classification.py --list
#   pixi run python supervised_learning/classification.py --variant mean__city__all
#   pixi run python supervised_learning/classification.py --all
#   pixi run python supervised_learning/classification.py --variant-index $SLURM_ARRAY_TASK_ID
# Run pipeline/graph_feature_calculation.py and pipeline/embedding_variants.py first.
# Uses the city-level variants in input/variants.txt.
# Label: mean average_distance_m over kept nodes (output/POI_distances), split
# into three equal groups. Model: logistic regression, stratified 5-fold CV.
# Results: output/experiments/<variant>/classification/

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.embedding_variants import (
    EXPERIMENT_ROOT, add_variant_arguments, chosen_variants, list_variants, load_city_table,cv_folds,
    load_setup, safe_filename, #######!!!! cv_folds
)

SEED = 42
FOLDS = cv_folds() #normally it's 5###!!!!!!
DPI = 300
CLASSES = ["low", "medium", "high"]  # accessibility; low = longest distances
DISTANCE_ROOT = PROJECT_ROOT / "output" / "POI_distances"


def accessibility_score(place):
    """Mean average_distance_m over the nodes FEATHER kept. Lower = more accessible."""
    name = safe_filename(place)
    path = DISTANCE_ROOT / name / f"{name}_poi_distances.csv"
    table = pd.read_csv(path, usecols=["average_distance_m", "feather_exclusion"])
    kept = table.loc[table["feather_exclusion"].fillna("") == "", "average_distance_m"]
    if kept.empty:
        raise ValueError(f"{path.name}: no nodes kept after FEATHER filtering.")
    return float(kept.mean())


def make_labels(places):
    scores = {}
    for place in places:
        try:
            scores[place] = accessibility_score(place)
        except (OSError, ValueError, KeyError) as error:
            print(f"No label for {place} (skipped): {error}", flush=True)
    ranked = pd.Series(scores).sort_values(ascending=False)  # longest distance first
    labels = pd.Series(index=ranked.index, dtype=object)
    for name, group in zip(CLASSES, np.array_split(np.arange(len(ranked)), len(CLASSES))):
        labels.iloc[group] = name
    smallest = labels.value_counts().reindex(CLASSES, fill_value=0).min()
    if smallest < FOLDS:
        raise ValueError(f"{len(ranked)} labelled cities give {smallest} in the smallest class. "
                         f"{FOLDS}-fold needs at least {FOLDS} per class ({FOLDS * len(CLASSES)} cities)."
                         f"Add cities, or set folds = {max(smallest, 2)} in input/variants.txt.")
    return ranked, labels


def model():
    return Pipeline([
        ("constant", VarianceThreshold(0.0)),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000, random_state=SEED)),
    ])


def fold_metrics(y_true, y_pred, probabilities):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "roc_auc_ovr": roc_auc_score([CLASSES.index(v) for v in y_true], probabilities,
                                     multi_class="ovr", labels=list(range(len(CLASSES)))),
    }


def save_plots(y_true, y_pred, probabilities, final, columns, out, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=CLASSES), display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted accessibility")
    ax.set_ylabel("True accessibility")
    ax.set_title(f"Confusion matrix (5-fold)\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for index, name in enumerate(CLASSES):
        RocCurveDisplay.from_predictions(y_true == name, probabilities[:, index], name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"ROC, one class vs. rest (5-fold)\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=DPI)
    plt.close(fig)

    kept = final[:-1].get_feature_names_out(columns)
    logreg = final.named_steps["logreg"]
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(6 * len(CLASSES), 6))
    for ax, name in zip(axes, CLASSES):
        weights = pd.Series(logreg.coef_[list(logreg.classes_).index(name)], index=kept)
        top = weights.reindex(weights.abs().sort_values().index[-15:])
        ax.barh(top.index, top.to_numpy(), color=np.where(top > 0, "#1C6E8C", "#C0504D"))
        ax.set_title(f"{name} accessibility")
        ax.tick_params(axis="y", labelsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
    fig.suptitle(f"Largest coefficients (final model, scaled features)\n{title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "top_coefficients.png", dpi=DPI)
    plt.close(fig)


def run_variant(variant, setup, ranked, labels):
    out = EXPERIMENT_ROOT / variant.name / "classification"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Classification {variant.name} (r = {list(variant.r_values)}) ===", flush=True)
    features, _ = load_city_table(variant, setup)
    places = [place for place in labels.index if place in features.index]
    X, y = features.loc[places], labels.loc[places].to_numpy()
    counts = pd.Series(y).value_counts().reindex(CLASSES, fill_value=0)
    if counts.min() < FOLDS:
        raise ValueError(f"Only {counts.to_dict()} cities per class have features; need {FOLDS} each.")

    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    predicted = np.empty(len(y), dtype=object)
    probabilities = np.zeros((len(y), len(CLASSES)))
    fold_of = np.zeros(len(y), dtype=int)
    rows = []
    for number, (train, test) in enumerate(folds.split(X, y), start=1):
        fitted = model().fit(X.iloc[train], y[train])
        order = [list(fitted.classes_).index(name) for name in CLASSES]
        probabilities[test] = fitted.predict_proba(X.iloc[test])[:, order]
        predicted[test] = fitted.predict(X.iloc[test])
        fold_of[test] = number
        rows.append({"fold": str(number), "cities": len(test),
                     **fold_metrics(y[test], predicted[test], probabilities[test])})
    log = pd.DataFrame(rows)
    metrics = log.columns[2:]
    log = pd.concat([
        log,
        pd.DataFrame([{"fold": "mean", "cities": len(y), **log[metrics].mean()},
                      {"fold": "std", "cities": len(y), **log[metrics].std(ddof=0)},
                      {"fold": "pooled", "cities": len(y), **fold_metrics(y, predicted, probabilities)}]),
    ], ignore_index=True)
    log.to_csv(out / "results_log.csv", index=False, float_format="%.4f")

    pd.DataFrame({
        "distance_score_m": ranked.loc[places].round(1), "true": y, "predicted": predicted,
        **{f"probability_{name}": probabilities[:, i].round(3) for i, name in enumerate(CLASSES)},
        "fold": fold_of, "correct": predicted == y,
    }, index=pd.Index(places, name="place")).to_csv(out / "predictions.csv")
    (out / "classification_report.txt").write_text(
        classification_report(y, predicted, labels=CLASSES, zero_division=0), encoding="utf-8")

    final = model().fit(X, y)
    joblib.dump({"model": final, "columns": list(X.columns), "classes": CLASSES}, out / "model.joblib")
    save_plots(y, predicted, probabilities, final, list(X.columns), out, variant.name)

    mean, std = log.set_index("fold").loc["mean"], log.set_index("fold").loc["std"]
    per_class = f1_score(y, predicted, labels=CLASSES, average=None, zero_division=0)
    result = {
        "task": "classification", "variant": variant.name, "categories": variant.categories,
        "level": variant.level, "r_selection": variant.r_mode, "r_values": list(variant.r_values),
        "samples": len(y), "features": X.shape[1], "folds": FOLDS,
        **{f"{m}_mean": float(mean[m]) for m in metrics}, **{f"{m}_std": float(std[m]) for m in metrics},
        **{f"f1_{name}": float(score) for name, score in zip(CLASSES, per_class)},
        "class_counts": counts.to_dict(), "seed": SEED,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(log.round(3).to_string(index=False), flush=True)
    print(f"Saved results to {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_variant_arguments(parser)
    args = parser.parse_args()
    setup = load_setup()
    variants = chosen_variants(args, list_variants(setup, levels=("city",)))
    if variants is None:
        return
    if not variants:
        raise ValueError("No city-level variants in input/variants.txt.")
    ranked, labels = make_labels(setup.places)
    print("Cities per class:", labels.value_counts().reindex(CLASSES).to_dict(), flush=True)
    for variant in variants:
        run_variant(variant, setup, ranked, labels)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
