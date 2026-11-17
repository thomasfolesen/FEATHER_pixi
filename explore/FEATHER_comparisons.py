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
    config = {"high_low_percent": 1.0}
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
        elif key in ("theta_max", "high_low_percent", "CLIP_BUFFER_M", "eval_points", "order"):
            config[key] = float(value) if key in ("theta_max", "high_low_percent") else int(value)
        elif key not in ("high_low_percent", "NEAREST_POIS", "outlier_bottom_percent", "outlier_top_percent", "outlier_iqr_multiplier"):
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
        print("Run pipeline/FEATHER_embeddings.py first.\n")
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

def format_curve_axis(ax, title, ylabel):
    """Apply labels and a zero reference line to a characteristic-function panel."""
    ax.set_title(title)
    ax.set_xlabel("Evaluation point θ")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.grid(alpha=0.2)

def save_figure(fig, destination, dpi=300):
    """Create the output folder, save the figure, and release it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def save_curves(series, theta, title, destination, statistic="Mean"):
    # Each entry contains a label and its cosine/sine curves.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for index, (label, values) in enumerate(series.items()):
        for component, ax in enumerate(axes):
            ax.plot(
                theta, values[component],
                color=plt.get_cmap("tab10")(index % 10),
                linestyle=["-", "--", "-.", ":"][(index // 10) % 4],
                marker="o", markersize=3, label=label,
            )
    for ax, label in zip(axes, ["Real (cosine)", "Imaginary (sine)"]):
        format_curve_axis(ax, label, f"{statistic} across nodes")
        ax.set_ylim(-1.05, 1.05)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, destination)

def high_low_summary(values, distances, node_ids, categories, percent):
    # Choose nodes by original distances, before missing values are filled.
    summaries = {}
    node_index = pd.Index(node_ids)
    for category_index, category in enumerate(categories):
        scores = distances[category].reindex(node_ids)
        scores = scores[np.isfinite(scores)].sort_index().sort_values(kind="stable")
        if len(scores) < 2:
            print(f"{category}: fewer than two reachable nodes; skipping high/low plots.", flush=True)
            continue
        # Round down, but keep at least one node in each group.
        count = max(1, int(len(scores) * percent / 100))
        groups = {"low": scores.index[:count], "high": scores.index[-count:]}
        summaries[category] = {}
        for group, selected in groups.items():
            positions = node_index.get_indexer(selected)
            # Shape: selected nodes, walk order, component, evaluation point.
            block = values[positions, :, :, category_index, :]
            summaries[category][group] = {
                "mean": block.mean(axis=0),
                "std": block.std(axis=0, ddof=0),
                "count": count,
                "distance_min": float(scores.loc[selected].min()),
                "distance_max": float(scores.loc[selected].max()),
            }
    return summaries

def save_high_low_city(place, summaries, theta, percent, folder):
    for category, groups in summaries.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        for row, group in enumerate(("high", "low")):
            data = groups[group]
            for component, ax in enumerate(axes[row]):
                for r in range(len(data["mean"])):
                    ax.plot(theta, data["mean"][r, component], label=f"r = {r + 1}")
                format_curve_axis(ax, ["Real (Cosine)", "Imaginary (Sine)"][component],
                                  "Mean across selected nodes")
                ax.set_ylim(-1.05, 1.05)
            axes[row, 0].legend(fontsize=8)
            fig.text(0.5, 0.91 if row == 0 else 0.46,
                     f"{'Highest' if group == 'high' else 'Lowest'} {percent:g}% — "
                     f"n={data['count']} — {data['distance_min']:.1f}–{data['distance_max']:.1f} m",
                     ha="center")
        fig.suptitle(f"{place} — {category.replace('_', ' ')}")
        fig.subplots_adjust(top=0.85, bottom=0.08, hspace=0.55, wspace=0.22)
        save_figure(fig, folder / f"feather_{category}.png")


def save_difference_bars(place, old, new, distances, node_ids, config, folder):
    """Mean/median of node-wise absolute differences; tails selected by POI distance.

    First average abs(new-old) over components and theta per node and order.
    Mean and median summarize these node scores, not differences of mean curves.
    High/low bars average the same scores within distance-selected groups.
    """
    shape = (len(node_ids), config["order"], 2, len(config["categories"]), config["eval_points"])
    scores = np.abs(new.reshape(shape) - old.reshape(shape)).mean(axis=(2, 4))
    records = []
    for c, category in enumerate(config["categories"]):
        raw = distances[f"{category}_mean_distance_m"].reindex(node_ids)
        finite = raw[np.isfinite(raw)].sort_index().sort_values(kind="stable")
        groups = {}
        if len(finite) >= 2:
            count = max(1, int(len(finite) * config["high_low_percent"] / 100))
            for group, selected in (("high", finite.iloc[-count:]), ("low", finite.iloc[:count])):
                groups[group] = (pd.Index(node_ids).get_indexer(selected.index), selected)
        for r in range(config["order"]):
            values = scores[:, r, c]
            record = {"category": category, "r": r + 1,
                      "mean_abs_embedding_difference": values.mean(),
                      "median_abs_embedding_difference": np.median(values), "vertices": len(values)}
            for group in ("high", "low"):
                positions, selected = groups.get(group, ([], pd.Series(dtype=float)))
                record[f"{group}_mean_abs_embedding_difference"] = values[positions].mean() if len(positions) else np.nan
                record[f"{group}_vertices"] = len(positions)
                record[f"{group}_distance_min_m"] = selected.min()
                record[f"{group}_distance_max_m"] = selected.max()
            records.append(record)
        rows = pd.DataFrame(records).query("category == @category")
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(config["order"])
        for j, (column, label) in enumerate((
            ("mean_abs_embedding_difference", "Mean"),
            ("median_abs_embedding_difference", "Median"),
            ("high_mean_abs_embedding_difference", "High-distance mean"),
            ("low_mean_abs_embedding_difference", "Low-distance mean"),
        )):
            ax.bar(x + (j - 1.5) * 0.2, rows[column], width=0.2, label=label)
        ax.set_xticks(x, [str(r + 1) for r in x])
        ax.set_xlabel("Walk order r")
        ax.set_ylabel("Absolute embedding difference")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.set_title(f"{place} — {category.replace('_', ' ')}\nFEATHER_new vs FEATHER")
        ax.grid(axis="y", alpha=0.2)
        details = [f"High/low: highest/lowest {config['high_low_percent']:g}% of finite distances among retained nodes."]
        for group, (_, selected) in groups.items():
            details.append(f"{group.title()}: n={len(selected)}, {selected.min():.1f}–{selected.max():.1f} m")
        if not groups:
            details.append("High/low unavailable: fewer than two finite distances.")
        fig.text(0.5, 0.01, "\n".join(details), ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        save_figure(fig, folder / "difference_bars" / f"{category}.png")
    pd.DataFrame(records).to_csv(folder / "difference_statistics.csv", index=False)


def save_original_curves(place, embedding, distances, node_ids, config):
    categories = config["categories"]
    order = config["order"]
    theta = np.linspace(0.01, config["theta_max"], config["eval_points"])
    values = embedding.reshape(len(node_ids), order, 2, len(categories), config["eval_points"])
    means = values.mean(axis=0)
    raw_distances = distances.reindex(node_ids)[
        [f"{category}_mean_distance_m" for category in categories]
    ].copy()
    raw_distances.columns = categories
    groups = high_low_summary(values, raw_distances, node_ids, categories, config["high_low_percent"])
    curve_root = PROJECT_ROOT / "output" / "FEATHER_curves_original"
    mean_folder = curve_root / "FEATHER_curves_mean" / safe_filename(place)
    high_folder = curve_root / "FEATHER_curves_high_low" / safe_filename(place)
    for index, category in enumerate(categories):
        save_curves(
            {f"r = {r + 1}": means[r, :, index, :] for r in range(order)},
            theta, f"Original FEATHER — {place} — {category.replace('_', ' ')}",
            mean_folder / f"feather_{category}.png",
        )
    save_high_low_city("Original FEATHER — " + place, groups, theta,
                       config["high_low_percent"], high_folder)
    last_groups = {
        category: {
            group: {**data, "mean": data["mean"][-1].copy(), "std": data["std"][-1].copy()}
            for group, data in category_groups.items()
        }
        for category, category_groups in groups.items()
    }
    return means[-1].copy(), last_groups

def save_median_curves(place, embedding, node_ids, config, algorithm):
    """Save per-place median curves for one FEATHER implementation."""
    categories = config["categories"]
    order = config["order"]
    eval_points = config["eval_points"]
    theta = np.linspace(0.01, config["theta_max"], eval_points)
    values = embedding.reshape(
        len(node_ids), order, 2, len(categories), eval_points
    )
    medians = np.median(values, axis=0)
    folder = (
        PROJECT_ROOT / "output" / "FEATHER_curves_original"
        / "FEATHER_curves_median" / safe_filename(place)
    )
    for index, category in enumerate(categories):
        series = {
            f"r = {r + 1}": medians[r, :, index, :]
            for r in range(order)
        }
        save_curves(
            series,
            theta,
            f"{algorithm} — {place} — {category.replace('_', ' ')} median",
            folder / f"feather_{category}.png",
            statistic="Median",
        )
    return medians[-1].copy()


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
    if not 0 < config["high_low_percent"] <= 50:
        raise ValueError("high_low_percent must be greater than 0 and at most 50.")
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
            save_original_curves(
                place, old_embedding, distances, node_ids, config
            )
            new_embedding = fit_model(FEATHER_new, graph, features, config)
            save_median_curves(
                place, old_embedding, node_ids, config, "FEATHER"
            )
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
            save_difference_bars(place, old_embedding, new_embedding, distances,
                                 node_ids, config, OUTPUT_ROOT / safe_filename(place))
        except (KeyError, ValueError) as error:
            print(f"{place}: comparison skipped: {error}")

if __name__ == "__main__":
    main()
