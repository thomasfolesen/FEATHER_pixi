"""Cluster cities using the mean and standard deviation of their POI distances.

Save in classification/ and run from the project root:
    pixi run python classification/cluster_baseline.py

Reads output/POI_distances and writes output/FEATHER_patterns_baseline.
Uses the same retained nodes and, when available, cluster count as FEATHER.
Use --columns to specify distance columns if their names differ from config.txt.
"""

import argparse
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
OUTPUT_ROOT = PROJECT_ROOT / "output" / "FEATHER_patterns_baseline"
SEED = 42
MAX_CLUSTERS = 6


def safe_filename(place):
    return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()


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
    if rank < 2:
        raise ValueError("At least two independent directions are needed for PCA plots.")
    component_count = min(5, rank)
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

    fig.suptitle("Raw POI distance baseline — mean and standard deviation")
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


def read_settings():
    settings = {}
    for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    places = [p.strip() for p in re.findall(r"\[([^\[\]]+)\]", settings["place"])]
    categories = [k for k, v in settings.items() if k != "place" and v.startswith("[[")]
    return settings, places, categories


def id_column(path):
    names = pd.read_csv(path, nrows=0).columns.tolist()
    found = [c for c in ("node_id", "osmid") if c in names]
    if len(found) != 1:
        raise ValueError(f"{path}: expected one ID column named node_id or osmid; found {names}")
    return found[0]


def input_for_city(root, city):
    folder = root / city
    candidates = sorted(folder.glob("*.csv")) if folder.is_dir() else []
    candidates += [p for p in (root / f"{city}_poi_distances.csv", root / f"{city}.csv") if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"{city}: expected exactly one POI CSV, found {candidates}. "
                         f"Use one CSV in {folder}, or {root / (city + '_poi_distances.csv')}.")
    return candidates[0]


def select_columns(path, categories, explicit):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if explicit:
        columns = explicit
    else:
        columns = []
        for category in categories:
            matches = [
                c for c in (category, f"distance_to_{category}", f"{category}_mean_distance_m")
                if c in header
            ]
            if len(matches) != 1:
                raise ValueError(f"{path}: cannot select distance column for {category!r}. "
                                 f"Use --columns followed by the distance column names. Header: {header}")
            columns.extend(matches)
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("Specify at least one unique distance column.")
    if set(columns) - set(header) or set(columns) & {"node_id", "osmid"}:
        raise ValueError(f"Invalid distance columns {columns}; CSV header: {header}")
    return columns


def summarise_distances(path, embedding, columns, distance_limit):
    """Calculate column means and population standard deviations for FEATHER nodes."""
    embedding_id = id_column(embedding)
    expected = pd.read_csv(
        embedding, usecols=[embedding_id], dtype={embedding_id: "string"}
    )[embedding_id]
    if expected.isna().any() or expected.duplicated().any():
        raise ValueError(f"{embedding}: missing or duplicate node IDs.")
    expected = set(expected)

    node_id = id_column(path)
    seen = set()
    count = 0
    mean = np.zeros(len(columns))
    squared_deviations = np.zeros(len(columns))

    chunks = pd.read_csv(
        path, usecols=[node_id, *columns], dtype={node_id: "string"}, chunksize=2000
    )
    for chunk in chunks:
        chunk = chunk.loc[chunk[node_id].isin(expected)]
        ids = chunk[node_id]
        if ids.duplicated().any() or seen.intersection(ids):
            raise ValueError(f"{path}: duplicate retained node IDs.")
        seen.update(ids)
        if chunk.empty:
            continue
        values = chunk[columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError(f"{path}: distances contain infinite or negative values.")
        # Match fill_distances() in the pipeline before summarising the city.
        values = np.minimum(values, distance_limit)
        values = np.where(np.isnan(values), distance_limit, values)
        # Merge this chunk into the running mean and squared deviations.
        size = len(values)
        batch_mean = values.mean(axis=0)
        delta = batch_mean - mean
        total = count + size
        squared_deviations += ((values - batch_mean) ** 2).sum(axis=0) + delta ** 2 * (count * size / total)
        mean += delta * (size / total)
        count = total

    if seen != expected or not count:
        raise ValueError(f"{path}: missing {len(expected - seen)} FEATHER node IDs or empty input.")
    std = np.sqrt(np.maximum(squared_deviations / count, 0))
    return np.concatenate([mean, std]), count


def main():
    global OUTPUT_ROOT, CONFIG_PATH
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "output" / "POI_distances")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--columns", nargs="+", help="Explicit raw distance columns, in consistent order")
    parser.add_argument("--clusters", help="Integer or auto; default matches existing FEATHER cluster count")
    args = parser.parse_args()
    OUTPUT_ROOT, CONFIG_PATH = args.output_dir, args.config
    comparison = PROJECT_ROOT / "output" / "FEATHER_patterns"
    if OUTPUT_ROOT.resolve() == comparison.resolve():
        raise ValueError("Choose a different output directory from FEATHER_patterns.")
    settings, places, categories = read_settings()
    distance_limit = float(settings["CLIP_BUFFER_M"])
    if not np.isfinite(distance_limit) or distance_limit <= 0:
        raise ValueError("CLIP_BUFFER_M must be a positive finite distance.")
    name_map = {safe_filename(p): p for p in places}
    if len(name_map) != len(places):
        raise ValueError("Configured places have duplicate normalized names.")

    membership_path = comparison / "cluster_membership.csv"
    reference = pd.read_csv(membership_path, index_col=0) if membership_path.exists() else None
    if reference is not None:
        if not reference.index.is_unique:
            raise ValueError("Reference city names must be unique.")
        cohort = {safe_filename(str(p)): str(p) for p in reference.index}
        if len(cohort) != len(reference):
            raise ValueError("Reference city names have duplicate normalized names.")
    else:
        cohort = name_map

    if args.clusters:
        selection = args.clusters
    elif reference is not None:
        selection = str(reference["cluster"].nunique())
    else:
        selection = settings.get("pattern_clusters", "auto")
    selection = selection.lower()
    if selection != "auto" and (not selection.isdigit() or int(selection) < 2):
        raise ValueError("Cluster count must be auto or an integer of at least 2.")

    summaries, counts, sources = {}, {}, []
    columns = None
    for city, place in cohort.items():
        path = input_for_city(args.input_dir, city)
        selected = select_columns(path, categories, args.columns)
        if columns is not None and selected != columns:
            raise ValueError("Distance column names/order differ between cities; use --columns.")
        columns = selected
        folder = PROJECT_ROOT / "output" / "FEATHER_embeddings" / city
        embeddings = sorted(folder.glob("*_feather_embeddings.csv"))
        if len(embeddings) != 1:
            raise ValueError(f"Expected exactly one saved FEATHER embedding in {folder}.")
        print(f"Reading {place}...", flush=True)
        summaries[place], counts[place] = summarise_distances(path, embeddings[0], columns, distance_limit)
        if reference is not None and "vertices" in reference and counts[place] != reference.loc[place, "vertices"]:
            raise ValueError(f"{place}: vertex count differs from FEATHER clustering; rerun it first.")
        sources.append({"place": place, "distances": str(path), "retained_node_source": str(embeddings[0])})

    if len(summaries) < 3:
        raise ValueError("At least three usable cities are needed.")
    names = [f"{stat}__{column}" for stat in ("mean", "std") for column in columns]
    features = pd.DataFrame.from_dict(summaries, orient="index", columns=names)
    features.index.name = "place"
    varying = features.columns[np.ptp(features.to_numpy(), axis=0) > 1e-12].tolist()
    if not varying:
        raise ValueError("All city summaries are identical.")

    scaler = StandardScaler()
    values = scaler.fit_transform(features[varying])
    if np.linalg.matrix_rank(values - values.mean(axis=0)) < 2:
        raise ValueError("Need at least two independent directions for comparable PCA plots.")
    model, scores = choose_clusters(values, selection)
    memberships = pd.DataFrame({"cluster": model.labels_, "vertices": [counts[p] for p in features.index],
                               "silhouette": silhouette_samples(values, model.labels_)}, index=features.index)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_ROOT / "graph_features.csv")
    memberships.to_csv(OUTPUT_ROOT / "cluster_membership.csv")
    scores.to_csv(OUTPUT_ROOT / "cluster_scores.csv", index=False)
    features.groupby(memberships["cluster"]).mean().to_csv(OUTPUT_ROOT / "cluster_feature_means.csv")
    joblib.dump({"scaler": scaler, "model": model, "columns": varying,
                 "distance_columns": columns, "config": settings}, OUTPUT_ROOT / "clustering.joblib")
    save_plots(values, memberships, scores)

    notes = {"config": settings, "included_places": list(features.index), "skipped_files": [],
             "chosen_clusters": model.n_clusters, "cluster_selection": selection, "seed": SEED,
             "sources": sources, "distance_columns": columns,
             "notes": ["Raw POI distance means and population standard deviations; no FEATHER transform.",
                       "Uses exactly the node IDs retained in each saved FEATHER embedding.",
                       "Distances are capped at CLIP_BUFFER_M; missing values are replaced by that limit, as in the pipeline.",
                       "Summaries stay in metres; the pipeline divides inputs by CLIP_BUFFER_M. Across-city standardization cancels this common scale factor.",
                       "K-means uses all varying standardized summary columns; PCA is visualization only.",
                       "PCA retains up to five independent directions; fewer may exist for the baseline.",
                       "Cluster labels are arbitrary; silhouette values across different representations are not proof of superiority."]}
    (OUTPUT_ROOT / "run_notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    print(memberships.to_string())
    print(f"\nSaved {model.n_clusters} baseline clusters to {OUTPUT_ROOT}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from None
