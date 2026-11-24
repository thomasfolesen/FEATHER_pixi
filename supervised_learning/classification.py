"""Classify cities as low, medium or high accessibility with logistic regression.

This script does not read node-level FEATHER embedding CSVs. Run
``pipeline/graph_feature_calculation.py`` first; classification then consumes the
prepared one-row-per-city files from ``output/graph_features``.

Commands from the project root:

    pixi run python supervised_learning/classification.py --list
    pixi run python supervised_learning/classification.py
    pixi run python supervised_learning/classification.py --all
    pixi run python supervised_learning/classification.py \
        --variant r_categoryAll_evalAll__mean
    pixi run python supervised_learning/classification.py \
        --variant-index $SLURM_ARRAY_TASK_ID

Results are written to:

    output/experiments/<embedding>__<graph_reduction>/classification/
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

from pipeline.pipeline_common import CONFIG_PATH, EMBEDDING_VARIANTS, load_config, safe_filename

VARIANTS_PATH = PROJECT_ROOT / "input" / "variants.txt"
GRAPH_FEATURE_ROOT = PROJECT_ROOT / "output" / "graph_features"
DISTANCE_ROOT = PROJECT_ROOT / "output" / "POI_distances"
EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "experiments"

GRAPH_REDUCTIONS = ("mean", "std", "mean_std")
CLASSES = ["low", "medium", "high"]  # low accessibility = longest distances
SEED = 42
DPI = 300


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_variants_file(path=VARIANTS_PATH):
    """Read classification experiment settings from variants.txt."""
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Invalid line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    embeddings = comma_list(
        values.get("embeddings", ", ".join(EMBEDDING_VARIANTS))
    )
    reductions = comma_list(
        values.get("graph_reductions", ", ".join(GRAPH_REDUCTIONS))
    )
    only = comma_list(values.get("only", ""))

    try:
        folds = int(values.get("folds", "5"))
    except ValueError as error:
        raise ValueError("folds in input/variants.txt must be an integer.") from error

    unknown_embeddings = [
        value for value in embeddings if value not in EMBEDDING_VARIANTS
    ]
    if unknown_embeddings:
        raise ValueError(
            "Unknown embedding variant(s): "
            + ", ".join(unknown_embeddings)
            + ". Expected: "
            + ", ".join(EMBEDDING_VARIANTS)
        )

    unknown_reductions = [
        value for value in reductions if value not in GRAPH_REDUCTIONS
    ]
    if unknown_reductions:
        raise ValueError(
            "Unknown graph_reductions value(s): "
            + ", ".join(unknown_reductions)
            + ". Expected: mean, std, mean_std."
        )

    if not embeddings:
        raise ValueError("Configure at least one embedding in input/variants.txt.")
    if not reductions:
        raise ValueError(
            "Configure at least one graph_reduction in input/variants.txt."
        )
    if folds < 2:
        raise ValueError("folds must be at least 2.")

    all_experiments = [
        f"{embedding}__{reduction}"
        for embedding in embeddings
        for reduction in reductions
    ]

    unknown_only = [name for name in only if name not in all_experiments]
    if unknown_only:
        raise ValueError(
            "Unknown experiment name(s) in 'only': " + ", ".join(unknown_only)
        )

    configured = only if only else all_experiments

    return {
        "embeddings": embeddings,
        "graph_reductions": reductions,
        "folds": folds,
        "all_experiments": all_experiments,
        "configured_experiments": configured,
    }


def split_experiment(name):
    try:
        embedding, reduction = name.rsplit("__", 1)
    except ValueError as error:
        raise ValueError(f"Invalid experiment name: {name!r}") from error
    if embedding not in EMBEDDING_VARIANTS or reduction not in GRAPH_REDUCTIONS:
        raise ValueError(f"Invalid experiment name: {name!r}")
    return embedding, reduction


def graph_feature_path(experiment_name):
    embedding, reduction = split_experiment(experiment_name)
    return GRAPH_FEATURE_ROOT / embedding / f"{reduction}.csv"


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list",
        action="store_true",
        help="List configured experiment names and zero-based indices.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all embedding x graph-reduction combinations, ignoring 'only'.",
    )
    group.add_argument(
        "--variant",
        help="Run one experiment by name, e.g. r_categoryAll_evalAll__mean.",
    )
    group.add_argument(
        "--variant-index",
        type=int,
        help="Run one configured experiment by zero-based index.",
    )


def choose_experiments(args, variants):
    all_experiments = variants["all_experiments"]
    configured = variants["configured_experiments"]

    if args.list:
        for index, name in enumerate(configured):
            print(f"{index}: {name}")
        return None

    if args.all:
        return all_experiments

    if args.variant is not None:
        if args.variant not in all_experiments:
            raise ValueError(
                f"Unknown variant {args.variant!r}. Run with --list to see names."
            )
        return [args.variant]

    if args.variant_index is not None:
        index = args.variant_index
        if index < 0 or index >= len(configured):
            raise ValueError(
                f"variant-index {index} is outside 0..{len(configured) - 1}. "
                "Run with --list to see indices."
            )
        return [configured[index]]

    return configured


def accessibility_score(place):
    """Mean average_distance_m over nodes retained by FEATHER."""
    name = safe_filename(place)
    path = DISTANCE_ROOT / name / f"{name}_poi_distances.csv"
    table = pd.read_csv(
        path,
        usecols=["average_distance_m", "feather_exclusion"],
    )
    kept = table.loc[
        table["feather_exclusion"].fillna("") == "",
        "average_distance_m",
    ]
    if kept.empty:
        raise ValueError(f"{path.name}: no nodes kept after FEATHER filtering.")
    return float(kept.mean())


def make_labels(places, folds):
    scores = {}
    for place in places:
        try:
            scores[place] = accessibility_score(place)
        except (OSError, ValueError, KeyError) as error:
            print(f"No label for {place} (skipped): {error}", flush=True)

    ranked = pd.Series(scores, dtype=float).sort_values(ascending=False)
    labels = pd.Series(index=ranked.index, dtype=object)

    for class_name, group in zip(
        CLASSES,
        np.array_split(np.arange(len(ranked)), len(CLASSES)),
    ):
        labels.iloc[group] = class_name

    smallest = int(labels.value_counts().reindex(CLASSES, fill_value=0).min())
    if smallest < folds:
        raise ValueError(
            f"{len(ranked)} labelled cities give {smallest} in the smallest class. "
            f"{folds}-fold needs at least {folds} per class "
            f"({folds * len(CLASSES)} cities). "
            f"Add cities, or set folds = {max(smallest, 2)} in input/variants.txt."
        )

    return ranked, labels


def load_city_features(experiment_name):
    """Load a prepared one-row-per-city graph-feature table."""
    path = graph_feature_path(experiment_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing graph-feature file: {path}\n"
            "Run: pixi run python pipeline/graph_feature_calculation.py"
        )

    table = pd.read_csv(path, dtype={"place": str})
    if "place" not in table.columns:
        raise ValueError(f"Missing 'place' column: {path}")
    if table["place"].duplicated().any():
        raise ValueError(f"Duplicate city rows: {path}")
    if len(table.columns) < 2:
        raise ValueError(f"No feature columns: {path}")

    table = table.set_index("place")
    table = table.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(table.to_numpy(dtype=float, copy=False)).all():
        raise ValueError(f"Non-finite graph features: {path}")

    return table, path


def model():
    return Pipeline(
        [
            ("constant", VarianceThreshold(0.0)),
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=5000, random_state=SEED)),
        ]
    )


def fold_metrics(y_true, y_pred, probabilities):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="weighted",
            zero_division=0,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=CLASSES,
            average="macro",
            zero_division=0,
        ),
        "roc_auc_ovr": roc_auc_score(
            [CLASSES.index(value) for value in y_true],
            probabilities,
            multi_class="ovr",
            labels=list(range(len(CLASSES))),
        ),
    }


def save_plots(y_true, y_pred, probabilities, final, columns, out, title, folds):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=CLASSES),
        display_labels=CLASSES,
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Predicted accessibility")
    ax.set_ylabel("True accessibility")
    ax.set_title(f"Confusion matrix ({folds}-fold)\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for index, class_name in enumerate(CLASSES):
        RocCurveDisplay.from_predictions(
            y_true == class_name,
            probabilities[:, index],
            name=class_name,
            ax=ax,
        )
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"ROC, one class vs. rest ({folds}-fold)\n{title}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "roc_curves.png", dpi=DPI)
    plt.close(fig)

    kept = final[:-1].get_feature_names_out(columns)
    logreg = final.named_steps["logreg"]
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(6 * len(CLASSES), 6))
    for ax, class_name in zip(axes, CLASSES):
        weights = pd.Series(
            logreg.coef_[list(logreg.classes_).index(class_name)],
            index=kept,
        )
        top = weights.reindex(weights.abs().sort_values().index[-15:])
        ax.barh(top.index, top.to_numpy())
        ax.set_title(f"{class_name} accessibility")
        ax.tick_params(axis="y", labelsize=7)
        ax.axvline(0, linewidth=0.8)
    fig.suptitle(
        f"Largest coefficients (final model, scaled features)\n{title}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / "top_coefficients.png", dpi=DPI)
    plt.close(fig)


def run_experiment(experiment_name, ranked, labels, folds):
    embedding, reduction = split_experiment(experiment_name)
    out = EXPERIMENT_ROOT / experiment_name / "classification"
    out.mkdir(parents=True, exist_ok=True)

    features, feature_path = load_city_features(experiment_name)
    places = [place for place in labels.index if place in features.index]
    if not places:
        raise ValueError(
            f"No labelled cities overlap with graph features in {feature_path}."
        )

    X = features.loc[places]
    y = labels.loc[places].to_numpy()
    counts = pd.Series(y).value_counts().reindex(CLASSES, fill_value=0)
    if counts.min() < folds:
        raise ValueError(
            f"Only {counts.to_dict()} cities per class have features; "
            f"need {folds} each."
        )

    print(
        f"\n=== Classification {experiment_name} ===\n"
        f"Input: {feature_path}\n"
        f"Cities: {len(X):,}; features: {X.shape[1]:,}",
        flush=True,
    )

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    predicted = np.empty(len(y), dtype=object)
    probabilities = np.zeros((len(y), len(CLASSES)))
    fold_of = np.zeros(len(y), dtype=int)
    rows = []

    for number, (train, test) in enumerate(cv.split(X, y), start=1):
        fitted = model().fit(X.iloc[train], y[train])
        class_order = [list(fitted.classes_).index(name) for name in CLASSES]
        probabilities[test] = fitted.predict_proba(X.iloc[test])[:, class_order]
        predicted[test] = fitted.predict(X.iloc[test])
        fold_of[test] = number
        rows.append(
            {
                "fold": str(number),
                "cities": len(test),
                **fold_metrics(y[test], predicted[test], probabilities[test]),
            }
        )

    log = pd.DataFrame(rows)
    metric_columns = list(log.columns[2:])
    log = pd.concat(
        [
            log,
            pd.DataFrame(
                [
                    {
                        "fold": "mean",
                        "cities": len(y),
                        **log[metric_columns].mean(),
                    },
                    {
                        "fold": "std",
                        "cities": len(y),
                        **log[metric_columns].std(ddof=0),
                    },
                    {
                        "fold": "pooled",
                        "cities": len(y),
                        **fold_metrics(y, predicted, probabilities),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    log.to_csv(out / "results_log.csv", index=False, float_format="%.4f")

    pd.DataFrame(
        {
            "distance_score_m": ranked.loc[places].round(1),
            "true": y,
            "predicted": predicted,
            **{
                f"probability_{name}": probabilities[:, index].round(3)
                for index, name in enumerate(CLASSES)
            },
            "fold": fold_of,
            "correct": predicted == y,
        },
        index=pd.Index(places, name="place"),
    ).to_csv(out / "predictions.csv")

    (out / "classification_report.txt").write_text(
        classification_report(
            y,
            predicted,
            labels=CLASSES,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    final = model().fit(X, y)
    joblib.dump(
        {
            "model": final,
            "columns": list(X.columns),
            "classes": CLASSES,
        },
        out / "model.joblib",
    )
    save_plots(
        y,
        predicted,
        probabilities,
        final,
        list(X.columns),
        out,
        experiment_name,
        folds,
    )

    mean_row = log.set_index("fold").loc["mean"]
    std_row = log.set_index("fold").loc["std"]
    per_class = f1_score(
        y,
        predicted,
        labels=CLASSES,
        average=None,
        zero_division=0,
    )
    result = {
        "task": "classification",
        "variant": experiment_name,
        "embedding": embedding,
        "graph_reduction": reduction,
        "graph_feature_file": str(feature_path.relative_to(PROJECT_ROOT)),
        "samples": len(y),
        "features": X.shape[1],
        "folds": folds,
        **{
            f"{metric}_mean": float(mean_row[metric])
            for metric in metric_columns
        },
        **{
            f"{metric}_std": float(std_row[metric])
            for metric in metric_columns
        },
        **{
            f"f1_{name}": float(score)
            for name, score in zip(CLASSES, per_class)
        },
        "class_counts": counts.to_dict(),
        "seed": SEED,
    }
    (out / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(log.round(3).to_string(index=False), flush=True)
    print(f"Saved results to {out}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    variants = read_variants_file()
    experiments = choose_experiments(args, variants)
    if experiments is None:
        return
    if not experiments:
        raise ValueError("No classification experiments selected.")

    config = load_config(CONFIG_PATH)
    ranked, labels = make_labels(config["places"], variants["folds"])
    print(
        "Cities per class:",
        labels.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        flush=True,
    )

    for experiment_name in experiments:
        run_experiment(
            experiment_name,
            ranked,
            labels,
            variants["folds"],
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
