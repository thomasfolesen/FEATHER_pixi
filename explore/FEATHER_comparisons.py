"""Run both FEATHER implementations on the pipeline's cached city data."""

import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "FEATHER_comparisons"


def parse_config_list(value):
    return [item.strip() for item in re.findall(r"\[([^\[\]]+)\]", value)]


def load_config(path):
    config = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key == "pattern_clusters":
            continue  # Used only by classification/FEATHER_patterns.py.
        if key == "place":
            config["places"] = parse_config_list(value)
        elif key in ("theta_max", "CLIP_BUFFER_M", "eval_points", "order"):
            config[key] = float(value) if key == "theta_max" else int(value)
        elif key not in ("high_low_percent", "NEAREST_POIS", "outlier_bottom_percent", "outlier_top_percent"):
            config.setdefault("categories", []).append(key)
    return config


def safe_filename(place):
    return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()


def city_inputs(place, config):
    city_name = safe_filename(place)
    city_cache = CACHE_ROOT / city_name
    graph_path = city_cache / (
        f"city_graph_walking_buffer_{config['CLIP_BUFFER_M']}m_"
        "simplified_undirected.pkl"
    )
    distance_path = (
        PROJECT_ROOT / "output" / "POI_distances" / city_name
        / f"{city_name}_poi_distances.csv"
    )
    missing = [path for path in (graph_path, distance_path) if not path.is_file()]
    if missing:
        print(f"{place}: required pipeline files are missing:")
        for path in missing:
            print(f"  {path}")
        print("Run pipeline/geofabrik_final.py first.\n")
        return None

    with graph_path.open("rb") as file:
        graph = pickle.load(file)
    distances = pd.read_csv(distance_path, index_col="node_id")
    distances.index = pd.to_numeric(distances.index, errors="raise")
    return graph, distances


def prepare_inputs(graph, distances, categories, buffer_m):
    columns = [f"{category}_mean_distance_m" for category in categories]
    missing = [column for column in columns if column not in distances]
    if missing:
        raise ValueError(f"Distance CSV is missing columns: {', '.join(missing)}")

    features = distances[columns].copy()
    features.columns = categories
    if np.isinf(features.to_numpy(dtype=float)).any() or (features < 0).any().any():
        raise ValueError("Distances must be finite or missing, and cannot be negative.")
    features = features.clip(upper=buffer_m).fillna(buffer_m)
    node_ids = [node for node in graph.nodes if node in features.index]
    if "feather_exclusion" in distances:
        eligible = distances["feather_exclusion"].fillna("").eq("")
        node_ids = [node for node in node_ids if eligible.get(node, False)]

    graph = graph.subgraph(node_ids).copy()
    # Isolated vertices cannot contribute to a neighbour-based walk.
    isolated = list(nx.isolates(graph))
    graph.remove_nodes_from(isolated)
    node_ids = list(graph.nodes)
    if len(node_ids) < 2:
        raise ValueError("Fewer than two usable nodes remain after filtering.")

    node_numbers = {node_id: index for index, node_id in enumerate(node_ids)}
    graph = nx.relabel_nodes(graph, node_numbers, copy=True)
    matrix = features.loc[node_ids].to_numpy(dtype=float) / buffer_m
    return graph, matrix, node_ids


def fit_model(model_class, graph, features, config):
    model = model_class(
        theta_max=config["theta_max"],
        eval_points=config["eval_points"],
        order=config["order"],
    )
    model.fit(graph=graph.copy(), X=features.copy())
    return np.asarray(model.get_embedding(), dtype=float)


def node_positions(graph, node_ids):
    positions = []
    for node_id in node_ids:
        node = graph.nodes[node_id]
        if "x" not in node or "y" not in node:
            raise ValueError("Cached graph nodes need x and y coordinates for plotting.")
        positions.append((node["x"], node["y"]))
    return np.asarray(positions)


def comparison_scores(old, new):
    """Compare corresponding embedding rows."""
    denominator = np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1)
    similarity = np.divide(
        np.sum(old * new, axis=1), denominator,
        out=np.full(len(old), np.nan), where=denominator > 0,
    )
    # Zero vectors have no direction, so their similarity stays missing.
    similarity = np.clip(similarity, -1, 1)
    difference = np.mean(np.abs(old - new), axis=1)
    return similarity, difference


def comparison_blocks(old_embedding, new_embedding, config):
    """Split the layout used by both FEATHER implementations."""
    shape = (len(old_embedding), config["order"], 2,
             len(config["categories"]), config["eval_points"])
    # Within each order: all cosine values, followed by all sine values.
    old = old_embedding.reshape(shape)
    new = new_embedding.reshape(shape)
    for order in [None, *range(config["order"])]:
        scope = "overall" if order is None else f"r{order + 1}"
        a = old if order is None else old[:, order:order + 1]
        b = new if order is None else new[:, order:order + 1]
        for component, index in (("both", None), ("cosine", 0), ("sine", 1)):
            left = a if index is None else a[:, :, index]
            right = b if index is None else b[:, :, index]
            yield scope, component, left.reshape(len(old), -1), right.reshape(len(new), -1)


def save_difference_map(place, title, positions, difference, limit, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    points = ax.scatter(
        *positions.T, c=difference, s=2,
        cmap="magma", vmin=0, vmax=limit,
    )
    fig.colorbar(points, ax=ax, shrink=0.75, label="Mean absolute embedding difference")
    ax.set_title(f"{place}: FEATHER vs FEATHER_new\n{title}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_order_plot(place, summary, folder):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    orders = summary[summary["scope"] != "overall"].copy()
    orders["r"] = orders["scope"].str[1:].astype(int)
    for component, color in (("both", "black"), ("cosine", "tab:blue"), ("sine", "tab:orange")):
        rows = orders[orders["component"] == component]
        label = "Cosine and sine" if component == "both" else component.title()
        ax.plot(rows["r"], rows["mean_absolute_embedding_difference"],
                marker="o", label=label, color=color)
    ax.set_xticks(sorted(orders["r"].unique()))
    ax.set_xlabel("Walk order r")
    ax.set_ylabel("Mean absolute embedding difference across vertices")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend()
    ax.set_title(f"{place}: differences by walk order")
    fig.tight_layout()
    fig.savefig(folder / "comparison_by_order.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_comparison(place, folder, graph, node_ids, old_embedding, new_embedding, config):
    expected = config["order"] * 2 * len(config["categories"]) * config["eval_points"]
    for embedding in (old_embedding, new_embedding):
        if embedding.shape != (len(node_ids), expected) or not np.isfinite(embedding).all():
            raise ValueError("Embeddings must be finite and match the configured dimensions.")
    positions = node_positions(graph, node_ids)
    results = pd.DataFrame(index=pd.Index(node_ids, name="node_id"))
    rows = []
    for scope, component, old, new in comparison_blocks(old_embedding, new_embedding, config):
        similarity, difference = comparison_scores(old, new)
        prefix = "" if (scope, component) == ("overall", "both") else f"{scope}_{component}_"
        results[prefix + "cosine_similarity"] = similarity
        results[prefix + "mean_absolute_embedding_difference"] = difference
        valid = np.isfinite(similarity)
        rows.append({
            "scope": scope, "component": component,
            "mean_cosine_similarity": similarity[valid].mean() if valid.any() else np.nan,
            "mean_absolute_embedding_difference": difference.mean(),
            "vertices": len(node_ids), "valid_similarity_vertices": int(valid.sum()),
        })
    summary = pd.DataFrame(rows)
    folder.mkdir(parents=True, exist_ok=True)
    results.to_csv(folder / "node_comparison.csv")
    summary.to_csv(folder / "comparison_summary.csv", index=False)

    # Share the difference scale across components and orders within this city.
    difference_columns = [name for name in results if name.endswith("mean_absolute_embedding_difference")]
    limit = results[difference_columns].max().max()
    limit = float(limit) if limit > 0 else 1.0
    for row in rows:
        scope, component = row["scope"], row["component"]
        if scope == "overall":
            continue
        prefix = f"{scope}_{component}_"
        destination = folder / scope
        filename = "embedding_comparison.png" if component == "both" else f"{component}_comparison.png"
        title = f"{scope} — {'cosine and sine' if component == 'both' else component} components"
        save_difference_map(
            place, title, positions,
            results[prefix + "mean_absolute_embedding_difference"].to_numpy(),
            limit, destination / filename,
        )
    save_order_plot(place, summary, folder)
    # Remove the three overall maps produced by earlier versions.
    for filename in ("embedding_comparison.png", "cosine_comparison.png", "sine_comparison.png"):
        (folder / filename).unlink(missing_ok=True)
    text = (
        f"Place: {place}\nCompared nodes: {len(node_ids)}\n"
        "Cosine similarity is applied to both, cosine-only and sine-only components.\n"
        "Each comparison includes all categories and evaluation points in its scope.\n"
        "Maps show mean absolute differences; undefined similarities are blank in the CSV.\n"
        "Map difference scales are shared within this city, not across cities.\n\n"
        + summary.to_string(index=False) + "\n"
    )
    (folder / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main():
    if not CONFIG_PATH.is_file():
        print(f"Config file not found: {CONFIG_PATH}")
        return
    config = load_config(CONFIG_PATH)
    required = ("places", "categories", "theta_max", "eval_points", "order", "CLIP_BUFFER_M")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config is missing settings: {', '.join(missing)}")

    try:
        from FEATHER.feather import FEATHER, FEATHER_new
    except ImportError as error:
        print(f"Could not import FEATHER and FEATHER_new: {error}")
        print("Check that FEATHER/feather.py defines both classes.")
        return

    for place in config["places"]:
        loaded = city_inputs(place, config)
        if loaded is None:
            continue
        graph, distances = loaded
        try:
            graph, features, node_ids = prepare_inputs(
                graph, distances, config["categories"], config["CLIP_BUFFER_M"]
            )
            old_embedding = fit_model(FEATHER, graph, features, config)
            new_embedding = fit_model(FEATHER_new, graph, features, config)
            if old_embedding.shape != new_embedding.shape:
                raise ValueError(
                    f"Embedding shapes differ: {old_embedding.shape} and "
                    f"{new_embedding.shape}."
                )
            # Keep coordinates from the original cached graph for the map.
            save_comparison(
                place, OUTPUT_ROOT / safe_filename(place),
                loaded[0], node_ids, old_embedding, new_embedding, config,
            )
        except (KeyError, ValueError) as error:
            print(f"{place}: comparison skipped: {error}")


if __name__ == "__main__":
    main()
