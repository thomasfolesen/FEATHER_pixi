"""Render amenity-distance heatmaps and IQR diagnostics from saved data."""
import argparse
import gc
import pickle
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pipeline.pipeline_common import (
    CONFIG_PATH, CACHE_ROOT, OUTPUT_ROOT, load_config, safe_filename,
    output_paths, read_node_csv, read_metadata,
)


def save_distance_map(place, category, graph_data, distances, folder, combined=False):
    nodes = graph_data["nodes"].copy()
    nodes["distance"] = distances.reindex(nodes.index)
    category_name = category.replace("_", " ")
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"distance_to_{category}.png"
    title = f"Mean distance to up to {NEAREST_POIS} nearest {category_name} POIs"
    note = (
        f"Mean over up to {NEAREST_POIS} POIs reachable within "
        f"{CLIP_BUFFER_M:,} m; unavailable neighbours are excluded.\n"
    )
    if combined:
        title = "Average walking distance across all amenity categories"
        note = (
            "Each category has equal weight; missing category distances "
            f"are set to {CLIP_BUFFER_M:,} m.\n"
        )
        note += (
            "Only nodes retained for FEATHER are shown after IQR filtering "
            "and isolated-node removal.\n"
            "Distances were calculated on the original buffered road network.\n"
        )
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(14, 11))
        display_edges = graph_data["edges"]
        if not display_edges.empty:
            display_edges.plot(
                ax=ax,
                color="#dddddd",
                linewidth=0.4,
                zorder=1,
            )
        missing = nodes[nodes["distance"].isna()]
        available = nodes[nodes["distance"].notna()]
        if not missing.empty:
            missing.plot(
                ax=ax,
                color="#999999",
                markersize=2,
                label=f"No reachable POI within {CLIP_BUFFER_M:,} m",
                zorder=2,
            )
            ax.legend()
        if not available.empty:
            available.plot(
                ax=ax,
                column="distance",
                cmap="viridis",
                vmin=0,
                vmax=CLIP_BUFFER_M,
                markersize=2,
                zorder=3,
                legend=True,
                legend_kwds={
                    "label": "Mean walking distance (metres)",
                    "ticks": np.linspace(0, CLIP_BUFFER_M, 5),
                    "shrink": 0.7,
                },
            )
        ax.set_title(
            f"{place}\n{title}",
            fontsize=15,
        )
        ax.set_axis_off()
        fig.text(
            0.5,
            0.02,
            (
                note + "Routes and POIs may extend outside the municipality. "
                "POI-to-network snapping distance is excluded."
            ),
            ha="center",
            fontsize=10,
        )
        # Explicitly keep an opaque background for the map.
        fig.savefig(image_path, dpi=600, bbox_inches="tight",
                    facecolor="white", transparent=False)
        plt.close(fig)
    return image_path

def save_iqr_figure(place, scores, iqr_multiplier):
    """Plot IQR thresholds using the same scores as outlier filtering."""

    filename = re.sub(r"[^\w-]+", "_", place).strip("_")
    output_dir = (
        Path(CONFIG_PATH).parent.parent / "output" / "graph_IQR" / filename
    )
    output_path = output_dir / f"{filename}_IQR.png"


    q1 = scores.quantile(0.25)
    median = scores.quantile(0.50)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    below = int((scores < lower).sum())
    above = int((scores > upper).sum())

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(
        scores,
        bins=50,
        color="#94a3b8",
        edgecolor="white",
        label="Node scores",
    )

    ax.axvspan(
        q1, q3, color="#22c55e", alpha=0.2,
        label=f"IQR = {iqr:.1f} m",
    )

    for value, label, color, style in [
        (lower, "Lower fence", "#dc2626", "--"),
        (q1, "Q1 (25%)", "#15803d", ":"),
        (median, "Median (50%)", "#1d4ed8", "-"),
        (q3, "Q3 (75%)", "#15803d", ":"),
        (upper, "Upper fence", "#dc2626", "--"),
    ]:
        ax.axvline(
            value,
            color=color,
            linestyle=style,
            linewidth=2,
            label=f"{label}: {value:.1f} m",
        )

    ax.set_title(
        f"{place}\n"
        f"IQR multiplier: {iqr_multiplier:g} | "
        f"Nodes: {len(scores):,} | "
        f"Below fence: {below:,} | Above fence: {above:,}"
    )
    ax.set_xlabel("Average distance across amenity categories (m)")
    ax.set_ylabel("Number of nodes")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved IQR figure: {output_path}", flush=True)
    return output_path

def run_plots(places):
    global CLIP_BUFFER_M, NEAREST_POIS
    for place in places:
        embedding_path, distance_path = output_paths(place)
        meta = read_metadata(embedding_path)
        CLIP_BUFFER_M = meta["CLIP_BUFFER_M"]
        NEAREST_POIS = meta["NEAREST_POIS"]
        path = CACHE_ROOT / "interactive_map_data" / f"{safe_filename(place)}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run the refactored pipeline first.")
        # Only load pickle files generated by your own pipeline.
        with path.open("rb") as file:
            data = pickle.load(file)
        if data["CLIP_BUFFER_M"] != CLIP_BUFFER_M or data["NEAREST_POIS"] != NEAREST_POIS:
            raise ValueError(f"Map cache and embedding metadata disagree for {place}")
        summary = read_node_csv(distance_path)
        node_ids = data["nodes"].index.map(str)
        retained = summary.index[summary["feather_exclusion"].fillna("") == ""]
        if set(node_ids) != set(retained):
            raise ValueError(f"Map cache and distance CSV nodes disagree for {place}")
        # Align CSV IDs as strings, then restore the geometry index for plotting.
        selected = summary.loc[node_ids].copy()
        selected.index = data["nodes"].index
        folder = OUTPUT_ROOT / "graph_POis" / safe_filename(place)
        for category in meta["categories"]:
            save_distance_map(place, category, data,
                              selected[f"{category}_mean_distance_m"], folder)
        save_distance_map(place, "all_categories_average", data,
                          selected["average_distance_m"], folder, combined=True)
        save_iqr_figure(place, summary["average_distance_m"], meta["outlier_iqr_multiplier"])
        del data, summary, selected
        gc.collect()
        print(f"Saved maps for {place}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--place", action="append", help="Exact place name; repeat for multiple cities.")
    args = parser.parse_args()
    run_plots(args.place or load_config(CONFIG_PATH)["places"])


if __name__ == "__main__":
    main()
