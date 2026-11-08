"""Explore accessibility patterns using every saved FEATHER_new walk order."""
# Save as classification/FEATHER_patterns.py and run from the project root:
# pixi run python classification/FEATHER_patterns.py
# Run pipeline/geofabrik_final.py first if embedding files are missing.
# All places must use the same categories, theta grid, distance scale and trimming.

import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "FEATHER_patterns"
SEED = 42
MAX_CLUSTERS = 6


def safe_filename(place):
    return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()


def load_config():
    settings = {}
    for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    places = [value.strip() for value in re.findall(r"\[([^\[\]]+)\]", settings["place"])]
    categories = [key for key, value in settings.items() if key != "place" and value.startswith("[[")]
    order, points = int(settings["order"]), int(settings["eval_points"])
    if not places or not categories or min(order, points) < 1:
        raise ValueError("Configure places, categories, and positive order/eval_points.")
    names = [safe_filename(place) for place in places]
    if len(set(names)) != len(names):
        raise ValueError("Some places produce the same filename; remove duplicate places.")
    columns = [
        f"r{r}_{category}_eval{point}_{component}"
        for r in range(1, order + 1)
        for component in ("real", "imaginary")
        for category in categories
        for point in range(1, points + 1)
    ]
    return settings, places, columns


def summarise_graph(path, columns):
    """Read one chunk at a time and combine its mean and variance."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if set(header) != {"node_id", *columns} or len(header) != len(columns) + 1:
        raise ValueError(f"{path.name}: embedding columns do not match config.txt. Rerun the pipeline.")
    count = 0
    mean = np.zeros(len(columns))
    squared_deviations = np.zeros(len(columns))
    # Allow space for several temporary arrays, even with many embedding columns.
    chunk_size = max(1, min(2000, 16_000_000 // (8 * len(columns) * 4)))
    for chunk in pd.read_csv(path, usecols=columns, dtype=float, chunksize=chunk_size):
        values = chunk[columns].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: embeddings contain missing or infinite values.")
        size = len(values)
        if not size:
            continue
        chunk_mean = values.mean(axis=0)
        chunk_squared = ((values - chunk_mean) ** 2).sum(axis=0)
        delta = chunk_mean - mean
        total = count + size
        squared_deviations += chunk_squared + delta ** 2 * (count * size / total)
        mean += delta * (size / total)
        count = total
    if not count:
        raise ValueError(f"{path.name}: no vertices found.")
    std = np.sqrt(np.maximum(squared_deviations / count, 0))
    return np.concatenate([mean, std]), count


def choose_clusters(values):
    rows, best = [], None
    distinct = len(np.unique(values, axis=0))
    for k in range(2, min(MAX_CLUSTERS, len(values) // 2, distinct) + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(values)
        sizes = np.bincount(model.labels_)
        if len(sizes) != k or (sizes == 0).any():
            continue
        score = silhouette_score(values, model.labels_)
        rows.append({"clusters": k, "silhouette": score,
                     "smallest_cluster": int(sizes.min())})
        if best is None or score > best[0]:
            best = (score, model)
    if best is None:
        raise ValueError("Not enough distinct graph summaries to form clusters.")
    return best[1], pd.DataFrame(rows)


def save_plots(values, memberships, scores):
    # PCA is only for this two-dimensional picture; clustering uses all features.
    pca = PCA(n_components=min(2, values.shape[1]), svd_solver="full")
    coordinates = pca.fit_transform(values)
    if coordinates.shape[1] == 1:
        coordinates = np.column_stack([coordinates, np.zeros(len(values))])
    fig, ax = plt.subplots(figsize=(12, 8))
    for cluster in sorted(memberships["cluster"].unique()):
        selected = memberships["cluster"].eq(cluster).to_numpy()
        ax.scatter(*coordinates[selected].T, label=f"Cluster {cluster}", s=55,
                   color=plt.get_cmap("tab10")(int(cluster)))
    for position, place in zip(coordinates, memberships.index):
        ax.annotate(place, position, xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} of variance)")
    second = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
    ax.set_ylabel(f"PC2 ({second:.1%} of variance)")
    ax.set_title("Accessibility embedding patterns — all walk orders")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "clusters_pca.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scores["clusters"], scores["silhouette"], marker="o")
    ax.set_xticks(scores["clusters"])
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Exploratory comparison of cluster counts")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "cluster_scores.png", dpi=600)
    plt.close(fig)


def main():
    settings, places, columns = load_config()
    summaries, counts, missing = {}, {}, []
    for place in places:
        name = safe_filename(place)
        path = PROJECT_ROOT / "output" / "FEATHER_embeddings" / name / f"{name}_feather_embeddings.csv"
        if not path.is_file():
            print(f"{place}: missing {path.name}; run pixi run python pipeline/geofabrik_final.py")
            missing.append(place)
            continue
        print(f"Reading {place}...", flush=True)
        summaries[place], counts[place] = summarise_graph(path, columns)
    if len(summaries) < 4:
        raise ValueError("At least four places with saved embeddings are needed for this exploration.")

    names = [f"{stat}__{column}" for stat in ("mean", "std") for column in columns]
    features = pd.DataFrame.from_dict(summaries, orient="index", columns=names)
    features.index.name = "place"
    # Each city is one sample, regardless of its number of vertices.
    varying = features.columns[np.ptp(features.to_numpy(), axis=0) > 1e-12].tolist()
    if not varying:
        raise ValueError("All graph summaries are identical; there is no variation to cluster.")
    scaler = StandardScaler()
    values = scaler.fit_transform(features[varying])
    model, scores = choose_clusters(values)
    memberships = pd.DataFrame({
        "cluster": model.labels_, "vertices": [counts[place] for place in features.index],
        "silhouette": silhouette_samples(values, model.labels_),
    }, index=features.index)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_ROOT / "graph_features.csv")
    memberships.to_csv(OUTPUT_ROOT / "cluster_membership.csv")
    scores.to_csv(OUTPUT_ROOT / "cluster_scores.csv", index=False)
    features.groupby(memberships["cluster"]).mean().to_csv(OUTPUT_ROOT / "cluster_feature_means.csv")
    joblib.dump({"scaler": scaler, "model": model, "columns": varying,
                 "embedding_columns": columns, "config": settings}, OUTPUT_ROOT / "clustering.joblib")
    save_plots(values, memberships, scores)
    notes = {
        "config": settings, "included_places": list(features.index), "missing_places": missing,
        "chosen_clusters": model.n_clusters, "seed": SEED,
        "notes": [
            "Uses all orders: mean and population standard deviation of every embedding column.",
            "Uses vertices retained in the pipeline embeddings, including its outlier filtering.",
            "Column checks cannot verify theta_max, distance scale or cache provenance; regenerate consistently.",
            "Cluster labels are arbitrary. Silhouette selection does not prove real pattern classes exist.",
            "Singleton clusters may be unusual places rather than reusable pattern types.",
            "Graph summaries lose vertex locations; interpret groups using the accessibility maps.",
            "Correlated features and repeated walk orders can affect distances between cities.",
        ],
    }
    (OUTPUT_ROOT / "run_notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(memberships.to_string())
    print(f"\nSaved {model.n_clusters} exploratory clusters to {OUTPUT_ROOT}")
    print("Inspect the maps before naming clusters. Adding places may change the groups.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
