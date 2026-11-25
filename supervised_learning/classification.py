"""Classify cities from precomputed graph-level FEATHER feature tables.

r is NOT an experiment dimension here. Classification uses every feature column in
output/graph_features, so the r values are exactly those selected upstream by
input/config.txt -> orders.
"""

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
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline_common import load_config, safe_filename

CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
VARIANTS_PATH = PROJECT_ROOT / "input" / "variants.txt"
GRAPH_FEATURE_ROOT = PROJECT_ROOT / "output" / "graph_features"
EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "experiments"
DISTANCE_ROOT = PROJECT_ROOT / "output" / "POI_distances"

SEED = 42
DPI = 300
CLASSES = ["low", "medium", "high"]


def flatten_orders(raw):
    values = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            values.extend(item)
        else:
            values.append(item)
    orders = sorted({int(value) for value in values})
    if not orders:
        raise ValueError("No orders configured in input/config.txt.")
    return orders


def read_variants_file(path):
    settings = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Invalid line in {path}: {raw_line}")
        key, value = [part.strip() for part in line.split("=", 1)]
        settings[key] = value

    embeddings = [v.strip() for v in settings.get("embeddings", "").split(",") if v.strip()]
    reductions = [v.strip() for v in settings.get("graph_reductions", "").split(",") if v.strip()]
    only = [v.strip() for v in settings.get("only", "").split(",") if v.strip()]
    folds = int(settings.get("folds", "5"))

    if not embeddings:
        raise ValueError("input/variants.txt must define embeddings = ...")
    if not reductions:
        raise ValueError("input/variants.txt must define graph_reductions = ...")
    if folds < 2:
        raise ValueError("folds must be at least 2.")

    return embeddings, reductions, only, folds


def accessibility_score(place):
    name = safe_filename(place)
    path = DISTANCE_ROOT / name / f"{name}_poi_distances.csv"
    table = pd.read_csv(path, usecols=["average_distance_m", "feather_exclusion"])
    kept = table.loc[
        table["feather_exclusion"].fillna("") == "",
        "average_distance_m",
    ]
    if kept.empty:
        raise ValueError(f"{path.name}: no nodes kept after FEATHER filtering.")
    return float(kept.mean())


def make_labels(places, folds):
    scores = {place: accessibility_score(place) for place in places}
    ranked = pd.Series(scores).sort_values(ascending=False)
    labels = pd.Series(index=ranked.index, dtype=object)

    for name, group in zip(CLASSES, np.array_split(np.arange(len(ranked)), len(CLASSES))):
        labels.iloc[group] = name

    counts = labels.value_counts().reindex(CLASSES, fill_value=0)
    if counts.min() < folds:
        raise ValueError(
            f"{len(labels)} cities give class counts {counts.to_dict()}; "
            f"{folds}-fold CV requires at least {folds} cities in each class."
        )
    return ranked, labels


def build_model():
    return Pipeline([
        ("constant", VarianceThreshold(0.0)),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=5000, random_state=SEED)),
    ])


def fold_metrics(y_true, y_pred, probabilities):
    encoded = [CLASSES.index(value) for value in y_true]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "roc_auc_ovr": roc_auc_score(
            encoded,
            probabilities,
            multi_class="ovr",
            labels=list(range(len(CLASSES))),
        ),
    }


def load_graph_table(embedding_variant, reduction, configured_places, labels):
    path = GRAPH_FEATURE_ROOT / embedding_variant / f"{reduction}.csv"
    table = pd.read_csv(path)
    if "place" not in table.columns:
        raise ValueError(f"{path}: missing place column.")
    if table["place"].duplicated().any():
        raise ValueError(f"{path}: duplicate places.")

    table = table.set_index("place")
    places = [place for place in configured_places if place in table.index and place in labels.index]
    if not places:
        raise ValueError(f"{path}: no configured labelled cities found.")

    X = table.loc[places].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy(dtype=float, copy=False)).all():
        raise ValueError(f"{path}: graph features contain non-finite values.")
    return X, places


def save_plots(y, predicted, probabilities, final_model, columns, out, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y, predicted, labels=CLASSES),
        display_labels=CLASSES,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted accessibility")
    ax.set_ylabel("True accessibility")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for index, name in enumerate(CLASSES):
        RocCurveDisplay.from_predictions(
            (np.asarray(y) == name).astype(int),
            probabilities[:, index],
            name=name,
            ax=ax,
        )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=DPI)
    plt.close(fig)

    selector = final_model.named_steps["constant"]
    kept_columns = np.asarray(columns)[selector.get_support()]
    coefficients = final_model.named_steps["logreg"].coef_
    importance = np.max(np.abs(coefficients), axis=0)
    top_count = min(20, len(importance))
    top = np.argsort(importance)[-top_count:]

    fig, ax = plt.subplots(figsize=(9, max(5, top_count * 0.32)))
    ax.barh(np.arange(top_count), importance[top])
    ax.set_yticks(np.arange(top_count), labels=kept_columns[top], fontsize=7)
    ax.set_xlabel("Maximum absolute logistic-regression coefficient")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "top_coefficients.png", dpi=DPI)
    plt.close(fig)


def run_experiment(embedding_variant, reduction, config, ranked, labels, folds):
    variant_name = f"{embedding_variant}__{reduction}"
    out = EXPERIMENT_ROOT / variant_name / "classification"
    out.mkdir(parents=True, exist_ok=True)

    X, places = load_graph_table(embedding_variant, reduction, config["places"], labels)
    y = labels.loc[places].to_numpy()
    counts = pd.Series(y).value_counts().reindex(CLASSES, fill_value=0)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    predicted = np.empty(len(y), dtype=object)
    probabilities = np.zeros((len(y), len(CLASSES)))
    fold_of = np.zeros(len(y), dtype=int)
    rows = []

    for fold_number, (train, test) in enumerate(splitter.split(X, y), start=1):
        fitted = build_model().fit(X.iloc[train], y[train])
        class_order = [list(fitted.classes_).index(name) for name in CLASSES]
        probabilities[test] = fitted.predict_proba(X.iloc[test])[:, class_order]
        predicted[test] = fitted.predict(X.iloc[test])
        fold_of[test] = fold_number
        rows.append({
            "fold": str(fold_number),
            "cities": len(test),
            **fold_metrics(y[test], predicted[test], probabilities[test]),
        })

    log = pd.DataFrame(rows)
    metric_columns = list(log.columns[2:])
    log = pd.concat([
        log,
        pd.DataFrame([
            {"fold": "mean", "cities": len(y), **log[metric_columns].mean().to_dict()},
            {"fold": "std", "cities": len(y), **log[metric_columns].std(ddof=0).to_dict()},
            {"fold": "pooled", "cities": len(y), **fold_metrics(y, predicted, probabilities)},
        ]),
    ], ignore_index=True)
    log.to_csv(out / "results_log.csv", index=False, float_format="%.4f")

    pd.DataFrame({
        "distance_score_m": ranked.loc[places].round(1),
        "true": y,
        "predicted": predicted,
        **{f"probability_{name}": probabilities[:, i].round(3) for i, name in enumerate(CLASSES)},
        "fold": fold_of,
        "correct": predicted == y,
    }, index=pd.Index(places, name="place")).to_csv(out / "predictions.csv")

    (out / "classification_report.txt").write_text(
        classification_report(y, predicted, labels=CLASSES, zero_division=0),
        encoding="utf-8",
    )

    final_model = build_model().fit(X, y)
    joblib.dump(
        {"model": final_model, "columns": list(X.columns), "classes": CLASSES},
        out / "model.joblib",
    )
    save_plots(y, predicted, probabilities, final_model, list(X.columns), out, variant_name)

    mean_row = log.set_index("fold").loc["mean"]
    std_row = log.set_index("fold").loc["std"]
    per_class = f1_score(y, predicted, labels=CLASSES, average=None, zero_division=0)
    orders = flatten_orders(config["orders"])

    result = {
        "task": "classification",
        "variant": variant_name,
        "embedding_variant": embedding_variant,
        "graph_reduction": reduction,
        "orders": orders,
        "samples": len(y),
        "features": X.shape[1],
        "folds": folds,
        **{f"{metric}_mean": float(mean_row[metric]) for metric in metric_columns},
        **{f"{metric}_std": float(std_row[metric]) for metric in metric_columns},
        **{f"f1_{name}": float(score) for name, score in zip(CLASSES, per_class)},
        "class_counts": counts.to_dict(),
        "seed": SEED,
    }
    (out / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved results to {out}", flush=True)
    return result


def choose_experiments(all_names, default_names, args):
    if args.list:
        for index, name in enumerate(all_names):
            print(f"{index}: {name}")
        return None
    if args.variant:
        if args.variant not in all_names:
            raise ValueError(f"Unknown variant: {args.variant}")
        return [args.variant]
    if args.variant_index is not None:
        if args.variant_index < 0 or args.variant_index >= len(all_names):
            raise ValueError("variant-index is out of range.")
        return [all_names[args.variant_index]]
    if args.all:
        return all_names
    return default_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--variant")
    parser.add_argument("--variant-index", type=int)
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    embeddings, reductions, only, folds = read_variants_file(VARIANTS_PATH)
    experiments = [f"{embedding}__{reduction}" for embedding in embeddings for reduction in reductions]
    defaults = only or experiments
    selected = choose_experiments(experiments, defaults, args)
    if selected is None:
        return

    ranked, labels = make_labels(config["places"], folds)
    print("Cities per class:", labels.value_counts().reindex(CLASSES).to_dict(), flush=True)
    print("Configured orders:", flatten_orders(config["orders"]), flush=True)

    summary_rows = []
    for name in selected:
        embedding_variant, reduction = name.rsplit("__", 1)
        summary_rows.append(
            run_experiment(embedding_variant, reduction, config, ranked, labels, folds)
        )

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    # Store lists compactly in the CSV.
    summary["orders"] = summary["orders"].map(lambda values: " ".join(map(str, values)))
    summary["class_counts"] = summary["class_counts"].map(json.dumps)
    summary.to_csv(EXPERIMENT_ROOT / "summary.csv", index=False, float_format="%.4f")
    print(f"Saved summary: {EXPERIMENT_ROOT / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
