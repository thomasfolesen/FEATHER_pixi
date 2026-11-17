"""Render all walk orders together: two-panel curves and 2×2 high/low plots."""
import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pipeline.pipeline_common import (
    CONFIG_PATH, OUTPUT_ROOT, load_config, safe_filename, embedding_columns,
    output_paths, read_node_csv, read_metadata,
)


def save_chart(series, theta, title, destination, statistic="Mean", bands=False):
    """Plot all labelled orders on matching real/imaginary panels."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for index, (label, (values, std)) in enumerate(series.items()):
        color = plt.get_cmap("tab10")(index % 10)
        style = ("-", "--", "-.", ":")[(index // 10) % 4]
        for component, ax in enumerate(axes):
            ax.plot(theta, values[component], color=color, linestyle=style,
                    linewidth=1.6, label=label)
            if std is not None:
                ax.fill_between(theta, values[component] - std[component],
                                values[component] + std[component], color=color,
                                alpha=0.12)
    for ax, component_name in zip(axes, ("Real (Cosine)", "Imaginary (Sine)")):
        ax.set(title=component_name, xlabel="Evaluation point θ")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.grid(alpha=0.2)
        if not bands:
            ax.set_ylim(-1.05, 1.05)
    axes[0].set_ylabel(f"{statistic} across nodes")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(title)
    if bands:
        fig.text(0.05, 0.01, "Shading: mean ± 1 population standard deviation; bands are not clipped.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04 if bands else 0, 1, 0.94))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_high_low_chart(groups, theta, title, destination, percent):
    """High distances above low distances; one range label per row."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.82,
                        hspace=0.48, wspace=0.12)
    for row, group in enumerate(("high", "low")):
        summary = groups[group]
        for r in range(summary["mean"].shape[0]):
            color = plt.get_cmap("tab10")(r % 10)
            style = ("-", "--", "-.", ":")[(r // 10) % 4]
            for component, ax in enumerate(axes[row]):
                mean = summary["mean"][r, component]
                ax.plot(theta, mean, color=color, linestyle=style,
                        linewidth=1.6, label=f"r = {r + 1}")
        for ax in axes[row]:
            ax.axhline(0, color="grey", linewidth=0.6)
            ax.grid(alpha=0.2)
        axes[row, 0].set_ylabel("Mean across selected nodes")
        # Center a single group/range label above both panels of this row.
        left = axes[row, 0].get_position()
        right = axes[row, 1].get_position()
        group_label = "Highest" if group == "high" else "Lowest"
        fig.text((left.x0 + right.x1) / 2, left.y1 + 0.018,
                 f"{group_label} {percent:g}% (n={summary['count']}) — "
                 f"Mean walking distance range: "
                 f"{summary['distance_min']:,.1f}–{summary['distance_max']:,.1f} m",
                 ha="center", va="bottom", fontsize=11)
    for component, label in enumerate(("Real (Cosine)", "Imaginary (Sine)")):
        box = axes[0, component].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.90, label, ha="center", fontsize=12)
        axes[1, component].set_xlabel("Evaluation point θ")
    # A shared legend keeps r colours consistent across all four panels.
    axes[0, 1].legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(title, y=0.98)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


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

def load_city(place, percent=None):
    embedding_path, distance_path = output_paths(place)
    meta = read_metadata(embedding_path)
    categories = meta["categories"]
    table = read_node_csv(embedding_path)
    columns = embedding_columns(categories, meta["order"], meta["eval_points"])
    if set(table.columns) != set(columns):
        raise ValueError(f"Embedding columns disagree with metadata: {embedding_path}")
    values = table.loc[:, columns].to_numpy(dtype=float).reshape(
        len(table), meta["order"], 2, len(categories), meta["eval_points"]
    )
    if not len(table) or not np.isfinite(values).all():
        raise ValueError(f"Empty or non-finite embeddings: {embedding_path}")
    distances = read_node_csv(distance_path)
    if not table.index.isin(distances.index).all():
        raise ValueError(f"Embedding nodes are missing from {distance_path}")
    retained = distances.index[distances["feather_exclusion"].fillna("") == ""]
    if set(retained) != set(table.index):
        raise ValueError(f"Retained nodes differ between CSV files for {place}")
    distances = distances.rename(columns={f"{c}_mean_distance_m": c for c in categories})
    percent = meta["high_low_percent"] if percent is None else percent
    if not 0 < percent <= 50:
        raise ValueError("high_low_percent must be greater than 0 and at most 50.")
    summaries = high_low_summary(values, distances, table.index, categories, percent)
    return meta, values.mean(axis=0), np.median(values, axis=0), summaries, percent


def run_plots(places, percent=None):
    root = OUTPUT_ROOT / "FEATHER_curves"
    for place in places:
        meta, means, medians, groups, selected_percent = load_city(place, percent)
        theta = np.linspace(0.01, meta["theta_max"], meta["eval_points"])
        name = safe_filename(place)
        for c, category in enumerate(meta["categories"]):
            title = f"{place} — {category.replace('_', ' ')} — all walk orders"
            filename = f"feather_{category}_all_r.png"
            for statistic, data in (("Mean", means), ("Median", medians)):
                series = {f"r = {r + 1}": (data[r, :, c, :], None)
                          for r in range(meta["order"])}
                save_chart(series, theta, title,
                           root / f"FEATHER_curves_{statistic.lower()}" / name / filename,
                           statistic=statistic)
            if category in groups:
                save_high_low_chart(
                    groups[category], theta, title,
                    root / "FEATHER_curves_high_low" / name / filename,
                    selected_percent,
                )
        # Release this city's arrays before loading the next city.
        del means, medians, groups, series
        gc.collect()
        print(f"Saved curves for {place}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place", action="append", help="Exact place name; repeat for multiple cities.")
    parser.add_argument("--percent", type=float, help="Override the high/low percentage.")
    args = parser.parse_args()
    places = args.place or load_config(CONFIG_PATH)["places"]
    run_plots(places, args.percent)


if __name__ == "__main__":
    main()
