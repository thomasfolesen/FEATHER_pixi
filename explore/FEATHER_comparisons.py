"""Run both FEATHER implementations on the pipeline's cached city data.
pixi run python explore/FEATHER_comparisons.py
runs only for places given in config file
see output/FEATHER_comparisons for files
"""


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


def save_comparison(place, folder, graph, node_ids, old_embedding, new_embedding):
    old_norm = np.linalg.norm(old_embedding, axis=1)
    new_norm = np.linalg.norm(new_embedding, axis=1)
    denominator = old_norm * new_norm
    cosine = np.divide(
        np.sum(old_embedding * new_embedding, axis=1), denominator,
        out=np.full(len(node_ids), np.nan), where=denominator > 0,
    )
    difference = np.mean(np.abs(old_embedding - new_embedding), axis=1)
    positions = node_positions(graph, node_ids)

    folder.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        "node_id": node_ids,
        "cosine_similarity": cosine,
        "mean_absolute_embedding_difference": difference,
    })
    results.to_csv(folder / "node_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, values, title, cmap, limits in (
        (axes[0], cosine, "Cosine similarity", "viridis", (-1, 1)),
        (axes[1], difference, "Mean absolute embedding difference", "magma", None),
    ):
        points = ax.scatter(
            positions[:, 0], positions[:, 1], c=values, s=2,
            cmap=cmap, vmin=limits[0] if limits else None,
            vmax=limits[1] if limits else None,
        )
        fig.colorbar(points, ax=ax, shrink=0.75)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_axis_off()
    fig.suptitle(f"{place}: FEATHER vs FEATHER_new")
    fig.tight_layout()
    fig.savefig(folder / "embedding_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    summary = (
        f"Place: {place}\n"
        f"Compared nodes: {len(node_ids)}\n"
        f"Embedding dimensions: {old_embedding.shape[1]}\n"
        f"Mean cosine similarity: {np.nanmean(cosine):.6f}\n"
        f"Mean absolute embedding difference: {difference.mean():.6f}\n"
    )
    (folder / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


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
                loaded[0], node_ids, old_embedding, new_embedding,
            )
        except (KeyError, ValueError) as error:
            print(f"{place}: comparison skipped: {error}")


if __name__ == "__main__":
    main()
