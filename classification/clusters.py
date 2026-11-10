"""Explore accessibility patterns using every saved FEATHER_new walk order."""
# Save as classification/FEATHER_patterns.py and run from the project root:
# pixi run python classification/FEATHER_patterns.py
# Run pipeline/geofabrik_final.py first if embedding files are missing.
# Reads every city folder in output/FEATHER_embeddings/.
# All embeddings must use the same categories, theta grid, scale and trimming.
# In input/config.txt, use pattern_clusters = 4 for four clusters,
# or pattern_clusters = auto to select a count using silhouette scores.

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
    selection = settings.setdefault("pattern_clusters", "auto").lower()
    if selection != "auto":
        try:
            count = int(selection)
        except ValueError:
            raise ValueError("pattern_clusters must be auto or an integer of at least 2.") from None
        if count < 2:
            raise ValueError("pattern_clusters must be auto or an integer of at least 2.")
    settings["pattern_clusters"] = selection
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


def choose_clusters(values, selection="auto"):
    rows, best = [], None
    distinct = len(np.unique(values, axis=0))
    if selection == "auto":
        candidates = range(2, min(MAX_CLUSTERS, len(values) // 2, distinct) + 1)
    else:
        k = int(selection)
        # Silhouette scores need more places than clusters.
        if not 2 <= k < len(values) or k > distinct:
            raise ValueError(
                f"pattern_clusters={k} needs at least {k + 1} available places "
                f"and {k} distinct graph summaries; found {len(values)} and {distinct}."
            )
        candidates = [k]
    for k in candidates:
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
    # PCA plots are for display; clustering uses all features.
    centered = values - values.mean(axis=0)
    rank = np.linalg.matrix_rank(centered)
    if rank < 5:
        raise ValueError(
            f"Only {rank} independent PCA directions are available; "
            "at least 5 are needed. Add more varied places."
        )
    component_count = 5
    pca = PCA(n_components=component_count, svd_solver="full")
    coordinates = pca.fit_transform(values)
    pairs = [(0, 1)]
    if component_count >= 3:
        pairs.extend([(0, 2), (1, 2)])
    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6), squeeze=False)
    for ax, (x_index, y_index) in zip(axes[0], pairs):
        for cluster in sorted(memberships["cluster"].unique()):
            selected = memberships["cluster"].eq(cluster).to_numpy()
            ax.scatter(coordinates[selected, x_index], coordinates[selected, y_index],
                       label=f"Cluster {cluster}", s=55,
                       color=plt.get_cmap("tab10")(int(cluster)))
        for index, place in enumerate(memberships.index):
            ax.annotate(place, (coordinates[index, x_index], coordinates[index, y_index]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        ax.set_xlabel(f"PC{x_index + 1} ({pca.explained_variance_ratio_[x_index]:.1%})")
        ax.set_ylabel(f"PC{y_index + 1} ({pca.explained_variance_ratio_[y_index]:.1%})")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Accessibility embedding patterns — all walk orders")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "clusters_pca.png", dpi=600)
    plt.close(fig)

    coordinates_table = pd.DataFrame(
        coordinates, index=memberships.index,
        columns=[f"PC{index + 1}" for index in range(component_count)],
    )
    coordinates_table.to_csv(OUTPUT_ROOT / "pca_coordinates.csv")
    variance = pd.DataFrame({
        "component": [f"PC{index + 1}" for index in range(component_count)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    variance.to_csv(OUTPUT_ROOT / "pca_variance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_x = np.arange(1, component_count + 1)
    plot_y = pca.explained_variance_ratio_ * 100
    bars = ax.bar(plot_x, plot_y, align="center", color="#1C3041",
                  edgecolor="#000000", linewidth=1.2)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3,
                f"{value:.1f}%", ha="center", va="bottom")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Percentage of Explained Variance")
    ax.set_title("Variance Explained per Principal Component", loc="left",
                 fontweight="bold", y=1.06)
    ax.set_xticks(plot_x)
    ax.set_ylim(0, max(plot_y) * 1.18)
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "pca_explained_variance.png", dpi=600)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scores["clusters"], scores["silhouette"], marker="o")
    ax.set_xticks(scores["clusters"])
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Mean silhouette score")
    ax.set_title("Silhouette scores for the evaluated cluster counts")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "cluster_scores.png", dpi=600)
    plt.close(fig)


def main():
    settings, places, columns = load_config()
    embedding_root = PROJECT_ROOT / "output" / "FEATHER_embeddings"
    paths = sorted(embedding_root.glob("*/*_feather_embeddings.csv"))
    if not paths:
        raise ValueError(f"No saved embeddings found in {embedding_root}. Run the pipeline first.")

    place_names = {safe_filename(place): place for place in places}
    summaries, counts, skipped = {}, {}, []
    print(f"Found {len(paths)} saved embedding files.", flush=True)
    for path in paths:
        folder_name = path.parent.name
        place = place_names.get(folder_name, folder_name.replace("_", " ").title())
        try:
            print(f"Reading {place}...", flush=True)
            summaries[place], counts[place] = summarise_graph(path, columns)
        except ValueError as error:
            print(f"Skipping {place}: {error}")
            skipped.append({"place": place, "file": str(path), "reason": str(error)})
    if len(summaries) < 6:
        raise ValueError(
            f"Found only {len(summaries)} usable embedding files; "
            "at least six places are needed for five PCA directions."
        )

    names = [f"{stat}__{column}" for stat in ("mean", "std") for column in columns]
    features = pd.DataFrame.from_dict(summaries, orient="index", columns=names)
    features.index.name = "place"
    # Each city is one sample, regardless of its number of vertices.
    varying = features.columns[np.ptp(features.to_numpy(), axis=0) > 1e-12].tolist()
    if not varying:
        raise ValueError("All graph summaries are identical; there is no variation to cluster.")
    scaler = StandardScaler()
    values = scaler.fit_transform(features[varying])
    model, scores = choose_clusters(values, settings["pattern_clusters"])
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
        "config": settings, "included_places": list(features.index), "skipped_files": skipped,
        "chosen_clusters": model.n_clusters, "seed": SEED,
        "notes": [
            "Uses all orders: mean and population standard deviation of every embedding column.",
            "Uses vertices retained in the pipeline embeddings, including its outlier filtering.",
            "Column checks cannot verify theta_max, distance scale or cache provenance; regenerate consistently.",
            "Cluster labels are arbitrary. Silhouette scores do not prove real pattern classes exist.",
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
