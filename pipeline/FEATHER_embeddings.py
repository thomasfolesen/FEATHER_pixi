"""Create POI distance CSVs and FEATHER embeddings from cached OSM data."""

import gc
import pickle
import sys
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pandana as pdna
from pyrosm import OSM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline_common import (
    FULL_EMBEDDING_VARIANT,
    embedding_columns,
    embedding_variant_path,
    load_config,
    safe_filename,
)


CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "output"
NETWORK_TYPE = "walking"

print("Starting pipeline/FEATHER_embeddings.py")


def city_cache_paths(place):
    """Return the cache paths created by pipeline/OSM_cache.py."""
    place_name = safe_filename(place)
    city_dir = CACHE_ROOT / place_name

    return {
        "place": place,
        "place_name": place_name,
        "city_dir": city_dir,
        "exact_boundary_path": city_dir / "municipality_boundary.geojson",
        "clip_boundary_path": (
            city_dir / f"municipality_clip_boundary_{CLIP_BUFFER_M}m.geojson"
        ),
        "city_pbf_path": (
            city_dir / f"{place_name}_buffer_{CLIP_BUFFER_M}m.osm.pbf"
        ),
        "graph_cache_path": (
            city_dir
            / (
                f"city_graph_{NETWORK_TYPE}_buffer_{CLIP_BUFFER_M}m_"
                "simplified_undirected.pkl"
            )
        ),
        "poi_cache_path": (
            city_dir / f"accessibility_pois_buffer_{CLIP_BUFFER_M}m.geojson"
        ),
    }


def require_files(place, paths, purpose):
    """Raise one clear error listing every missing cache file."""
    missing = [Path(path) for path in paths if not Path(path).is_file()]
    if not missing:
        return

    missing_text = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        f"Missing cached files for {place} while {purpose}:\n"
        f"{missing_text}\n"
        "Run: pixi run python pipeline/OSM_cache.py"
    )


def load_graph_cache(place, graph_cache_path):
    """Load the cached graph and create projected node/edge GeoDataFrames."""
    started = time.time()
    print(f"Loading cached walking graph for {place}...", flush=True)

    with graph_cache_path.open("rb") as file:
        graph = pickle.load(file)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError(f"Cached walking graph is empty: {graph_cache_path}")
    if "crs" not in graph.graph:
        raise ValueError(f"Cached walking graph has no CRS: {graph_cache_path}")

    projected_graph = ox.project_graph(graph)
    nodes, edges = ox.graph_to_gdfs(projected_graph)

    print(
        f"Graph ready for {place} in {time.time() - started:.1f} seconds.",
        flush=True,
    )
    return {
        "graph": graph,
        "projected_graph": projected_graph,
        "nodes": nodes,
        "edges": edges,
    }


def extract_pois_from_city_pbf(
    place,
    city_pbf_path,
    poi_cache_path,
    boundary_path,
    resolved_categories,
):
    """Load cached POIs or extract the configured POIs from the local city PBF."""
    if poi_cache_path.exists():
        print(f"Loading cached POIs for {place}: {poi_cache_path}", flush=True)
        features = gpd.read_file(poi_cache_path)
        if "category" not in features.columns:
            raise ValueError(
                f"Cached POI file has no 'category' column: {poi_cache_path}"
            )
        return features

    tags_by_key = {}
    for mappings in resolved_categories.values():
        for item in mappings:
            tags_by_key.setdefault(item["osm_key"], []).append(item["osm_tag"])

    print(f"Extracting POIs for {place} from the cached city PBF...", flush=True)
    osm = OSM(str(city_pbf_path), engine="out_of_core", workers="auto")
    raw = osm.get_data_by_custom_criteria(
        custom_filter=tags_by_key,
        osm_keys_to_keep=list(tags_by_key),
        tags_as_columns=list(tags_by_key) + ["name"],
        keep_nodes=True,
        keep_ways=True,
        keep_relations=True,
        keep_other_tags=False,
    )

    rows = []
    if raw is not None and not raw.empty:
        raw = raw.dropna(subset=["geometry"]).to_crs("EPSG:4326")
        raw["geometry"] = raw.geometry.representative_point()

        boundary = (
            gpd.read_file(boundary_path)
            .to_crs("EPSG:4326")
            .geometry.union_all()
        )
        raw = raw[raw.geometry.apply(boundary.covers)]

        for category, mappings in resolved_categories.items():
            for item in mappings:
                key = item["osm_key"]
                tag = item["osm_tag"]

                if key not in raw.columns:
                    continue

                matches = raw[raw[key] == tag]
                for _, poi in matches.iterrows():
                    rows.append(
                        {
                            "osm_type": poi.get("osm_type"),
                            "osm_id": poi.get("id"),
                            "category": category,
                            "osm_key": key,
                            "osm_tag": tag,
                            "name": poi.get("name"),
                            "geometry": poi["geometry"],
                        }
                    )

    features = gpd.GeoDataFrame(
        rows,
        columns=[
            "osm_type",
            "osm_id",
            "category",
            "osm_key",
            "osm_tag",
            "name",
            "geometry",
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    features = features.drop_duplicates(
        subset=["osm_type", "osm_id", "category", "osm_key", "osm_tag"]
    ).reset_index(drop=True)

    if not features.empty:
        features.to_file(poi_cache_path, driver="GeoJSON")
        print(f"Saved POI cache: {poi_cache_path}", flush=True)
    else:
        print(f"No configured POIs were found for {place}.", flush=True)

    return features


def calculate_poi_distances(graph_data, pois, categories):
    """Calculate mean network distance to up to NEAREST_POIS per category."""
    categories = list(categories)
    nodes = graph_data["nodes"]
    roads = (
        graph_data["edges"]
        .reset_index()
        .sort_values("length")
        .drop_duplicates(["u", "v"])
    )

    pois = pois.to_crs(nodes.crs)
    if (
        not np.isfinite(roads["length"]).all()
        or (roads["length"] < 0).any()
    ):
        raise ValueError("Road lengths must be finite, non-negative metres.")

    print("Preparing the buffered walking network...", flush=True)
    network = pdna.Network(
        node_x=nodes.geometry.x,
        node_y=nodes.geometry.y,
        edge_from=roads["u"],
        edge_to=roads["v"],
        edge_weights=roads[["length"]],
        twoway=True,
    )

    distances = pd.DataFrame(index=nodes.index, columns=categories, dtype=float)
    poi_counts = pd.DataFrame(0, index=nodes.index, columns=categories, dtype=int)

    for category in categories:
        destinations = (
            pois[pois["category"] == category]
            .drop_duplicates(subset=["osm_type", "osm_id"])
        )

        if destinations.empty:
            print(f"{category}: no POIs found.", flush=True)
            continue

        print(
            f"{category}: finding up to {NEAREST_POIS} nearest POIs "
            f"within {CLIP_BUFFER_M:,} metres...",
            flush=True,
        )
        network.set_pois(
            category=category,
            maxdist=CLIP_BUFFER_M,
            maxitems=NEAREST_POIS,
            x_col=destinations.geometry.x,
            y_col=destinations.geometry.y,
        )
        nearest = network.nearest_pois(
            distance=CLIP_BUFFER_M,
            category=category,
            num_pois=NEAREST_POIS,
            max_distance=-1,
        )
        nearest = nearest.where(
            (nearest >= 0) & (nearest <= CLIP_BUFFER_M)
        )

        distances[category] = nearest.mean(axis=1)
        poi_counts[category] = nearest.notna().sum(axis=1)

    return distances, poi_counts


def fill_distances(distances):
    """Use the same missing-distance replacement for totals and FEATHER."""
    features = distances.copy()

    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Distances contain infinite values.")
    if (features < 0).any().any():
        raise ValueError("Distances cannot be negative.")

    return features.clip(upper=CLIP_BUFFER_M).fillna(CLIP_BUFFER_M)


def remove_distance_outliers(graph, scores, iqr_multiplier=1.5):
    """Remove nodes outside Tukey's IQR fences, then isolated nodes."""
    if not np.isfinite(iqr_multiplier) or iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be finite and positive.")
    if scores.empty or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Outlier scores must be non-empty and finite.")

    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    excluded = pd.Series("", index=scores.index, name="feather_exclusion")
    excluded.loc[scores < lower] = "bottom_outlier"
    excluded.loc[scores > upper] = "top_outlier"

    bottom_count = int((excluded == "bottom_outlier").sum())
    top_count = int((excluded == "top_outlier").sum())

    filtered = graph.subgraph(excluded.index[excluded == ""]).copy()
    isolated = list(nx.isolates(filtered))
    excluded.loc[isolated] = "no_neighbours"
    filtered.remove_nodes_from(isolated)

    print(
        f"IQR={iqr:.2f} m; fences=[{lower:.2f}, {upper:.2f}] m. "
        f"Excluded {bottom_count} bottom and {top_count} top outliers, "
        f"plus {len(isolated)} isolated nodes; "
        f"{len(filtered)} nodes remain.",
        flush=True,
    )
    return filtered, excluded


def prepare_feather_inputs(graph, distances, categories):
    """Relabel graph nodes and create the normalized FEATHER feature matrix."""
    node_ids = list(graph.nodes)
    categories = list(categories)
    features = fill_distances(
        distances.reindex(index=node_ids, columns=categories)
    )
    node_numbers = {
        node_id: position
        for position, node_id in enumerate(node_ids)
    }
    numbered_graph = nx.relabel_nodes(graph, node_numbers, copy=True)

    return {
        "graph": numbered_graph,
        "features": features.to_numpy(dtype=float) / CLIP_BUFFER_M,
        "node_ids": node_ids,
    }


def municipality_data(graph_data, boundary_path):
    """Restrict the buffered graph to nodes inside the municipality boundary."""
    boundary = (
        gpd.read_file(boundary_path)
        .to_crs(graph_data["nodes"].crs)
        .geometry.union_all()
    )
    nodes = graph_data["nodes"]
    node_ids = nodes.index[nodes.geometry.apply(boundary.covers)]

    return {
        "graph": graph_data["graph"].subgraph(node_ids).copy(),
        "nodes": nodes.loc[node_ids].copy(),
    }


def load_distance_summary(summary_path, categories, expected_node_ids):
    """Load a saved summary and reconstruct FEATHER's category distance table."""
    print(f"Loading saved POI distances: {summary_path}", flush=True)
    print(
        "The existing distance CSV is being reused as-is. Delete it before "
        "rerunning if CLIP_BUFFER_M, NEAREST_POIS, categories or POI tags changed.",
        flush=True,
    )

    header = pd.read_csv(summary_path, nrows=0)
    if header.columns.empty:
        raise ValueError(f"Saved distance CSV has no columns: {summary_path}")

    index_column = header.columns[0]
    table = pd.read_csv(
        summary_path,
        dtype={index_column: str},
        index_col=0,
    )
    table.index = table.index.map(str)

    if not table.index.is_unique or not table.columns.is_unique:
        raise ValueError(
            f"Saved distance CSV contains duplicate rows or columns: {summary_path}"
        )

    expected_node_ids = list(expected_node_ids)
    expected_strings = [str(node_id) for node_id in expected_node_ids]
    if len(expected_strings) != len(set(expected_strings)):
        raise ValueError("Graph node IDs are not unique after conversion to strings.")

    saved_nodes = set(table.index)
    expected_nodes = set(expected_strings)
    if saved_nodes != expected_nodes:
        missing_count = len(expected_nodes - saved_nodes)
        extra_count = len(saved_nodes - expected_nodes)
        raise ValueError(
            "Saved distance CSV nodes do not match the current municipality graph: "
            f"{summary_path}. Missing={missing_count}, extra={extra_count}. "
            "Delete or move the CSV to calculate fresh distances."
        )

    table = table.loc[expected_strings].copy()
    table.index = pd.Index(expected_node_ids, name="node_id")

    categories = list(categories)
    mean_columns = [f"{category}_mean_distance_m" for category in categories]
    missing_columns = [
        column for column in mean_columns if column not in table.columns
    ]
    if missing_columns:
        raise ValueError(
            "Saved distance CSV is missing required mean-distance columns: "
            f"{missing_columns}. Delete or move {summary_path} to recalculate it."
        )

    distances = table[mean_columns].copy()
    distances.columns = categories
    distances = distances.apply(pd.to_numeric, errors="raise")

    # Recalculate these two summaries from the current category list. The saved
    # POI-count columns stay in the CSV but are not FEATHER input features.
    filled = fill_distances(distances)
    table["total_distance_m"] = filled.sum(axis=1)
    table["average_distance_m"] = filled.mean(axis=1)

    return table, distances


def calculate_distance_summary(city, graph_data, municipality, categories):
    """Calculate a new POI distance summary from cached local OSM data."""
    pois = extract_pois_from_city_pbf(
        city["place"],
        city["city_pbf_path"],
        city["poi_cache_path"],
        city["clip_boundary_path"],
        categories,
    )

    print("Calculating walking distances...", flush=True)
    distances, poi_counts = calculate_poi_distances(
        graph_data,
        pois,
        categories,
    )

    distances = distances.reindex(municipality["nodes"].index)
    poi_counts = poi_counts.reindex(municipality["nodes"].index)

    distance_summary = pd.DataFrame(index=distances.index)
    distance_summary.index.name = "node_id"

    for category in categories:
        distance_summary[f"{category}_mean_distance_m"] = distances[category]
        distance_summary[
            f"{category}_poi_count_up_to_{NEAREST_POIS}"
        ] = poi_counts[category]

    filled = fill_distances(distances[list(categories)])
    distance_summary["total_distance_m"] = filled.sum(axis=1)
    distance_summary["average_distance_m"] = filled.mean(axis=1)

    return distance_summary, distances



def validate_saved_embedding(embedding_path, node_ids, expected_columns):
    """Validate an existing embedding CSV without loading its numeric matrix."""
    print(f"Using saved FEATHER embeddings: {embedding_path}", flush=True)

    header = pd.read_csv(embedding_path, nrows=0)
    if header.columns.empty:
        raise ValueError(f"Saved embedding CSV has no columns: {embedding_path}")

    index_column = header.columns[0]
    saved_columns = list(header.columns[1:])
    if saved_columns != list(expected_columns):
        raise ValueError(
            "Saved embedding columns do not match the current categories, "
            f"orders or evaluation points: {embedding_path}. Delete or move "
            "the CSV to calculate fresh embeddings."
        )

    saved_ids = pd.read_csv(
        embedding_path,
        usecols=[index_column],
        dtype={index_column: str},
    )[index_column]

    if saved_ids.duplicated().any():
        raise ValueError(f"Saved embeddings contain duplicate node IDs: {embedding_path}")

    expected_ids = [str(node_id) for node_id in node_ids]
    if saved_ids.tolist() != expected_ids:
        raise ValueError(
            "Saved embedding nodes or row order do not match the current "
            f"filtered graph: {embedding_path}. Delete or move the CSV to "
            "calculate fresh embeddings."
        )

    print(
        "No embedding JSON is used. Delete this CSV before rerunning if "
        "theta_max, distance settings, POI inputs, or IQR settings changed.",
        flush=True,
    )

def configured_orders(config):
    """Return the unique configured FEATHER orders as sorted positive integers.

    Accepts both the config syntax used by this project, for example
    ``orders = [[7]]`` or ``orders = [[1], [7]]``, and an already-flattened
    list such as ``[1, 7]`` returned by a config loader.
    """
    raw_orders = config["orders"]
    orders = []

    for item in raw_orders:
        if isinstance(item, (list, tuple, set)):
            orders.extend(item)
        else:
            orders.append(item)

    try:
        orders = sorted({int(order) for order in orders})
    except (TypeError, ValueError) as error:
        raise ValueError(
            "orders must contain positive integers, for example [[7]] or [[1], [7]]."
        ) from error

    if not orders or any(order <= 0 for order in orders):
        raise ValueError(
            "orders must contain positive integers, for example [[7]] or [[1], [7]]."
        )

    return orders


def process_city(place, config):
    """Create or reuse distances and FEATHER embeddings for one city."""
    categories = config["resolved_categories"]
    orders = configured_orders(config)
    max_order = max(orders)
    eval_points = config["eval_points"]
    city = city_cache_paths(place)
    place_name = city["place_name"]

    summary_folder = OUTPUT_ROOT / "POI_distances" / place_name
    summary_path = summary_folder / f"{place_name}_poi_distances.csv"

    embedding_path = embedding_variant_path(
        place,
        FULL_EMBEDDING_VARIANT,
    )
    embedding_folder = embedding_path.parent

    print(f"\nProcessing {place}...", flush=True)
    require_files(
        place,
        [city["exact_boundary_path"], city["graph_cache_path"]],
        "loading the municipality graph",
    )

    buffered = load_graph_cache(place, city["graph_cache_path"])
    municipality = municipality_data(buffered, city["exact_boundary_path"])

    if summary_path.is_file():
        distance_summary, distances = load_distance_summary(
            summary_path,
            categories,
            municipality["nodes"].index,
        )
    else:
        require_files(
            place,
            [city["city_pbf_path"], city["clip_boundary_path"]],
            "calculating POI distances",
        )
        distance_summary, distances = calculate_distance_summary(
            city,
            buffered,
            municipality,
            categories,
        )

    del buffered
    gc.collect()

    graph, excluded = remove_distance_outliers(
        municipality["graph"],
        distance_summary["average_distance_m"],
        config["outlier_iqr_multiplier"],
    )
    distance_summary["feather_exclusion"] = excluded

    summary_folder.mkdir(parents=True, exist_ok=True)
    distance_summary.to_csv(summary_path, index=True)
    print(f"Saved distance CSV: {summary_path}", flush=True)

    node_ids = list(graph.nodes)
    if not node_ids:
        raise ValueError(f"No nodes remain after filtering for {place}.")

    distances = distances.loc[node_ids].copy()
    del distance_summary, excluded, municipality
    gc.collect()

    # FEATHER_new calculates every order through max_order. The output CSV
    # retains only the orders explicitly requested in input/config.txt.
    all_columns = list(
        embedding_columns(categories, max_order, eval_points)
    )
    expected_columns = list(
        embedding_columns(categories, orders, eval_points)
    )

    if embedding_path.is_file():
        validate_saved_embedding(
            embedding_path,
            node_ids,
            expected_columns,
        )
        del graph
    else:
        selected_orders = ", ".join(f"r{order}" for order in orders)
        print(
            f"Calculating FEATHER embeddings through r = {max_order}; "
            f"saving only {selected_orders}...",
            flush=True,
        )
        inputs = prepare_feather_inputs(graph, distances, categories)
        del graph

        from FEATHER.feather import FEATHER_new

        model = FEATHER_new(
            theta_max=config["theta_max"],
            eval_points=eval_points,
            order=max_order,
        )
        model.fit(graph=inputs["graph"], X=inputs["features"])
        embedding = model.get_embedding()
        node_ids = list(inputs["node_ids"])

        if not np.isfinite(embedding).all():
            raise ValueError(f"Calculated embeddings contain non-finite values for {place}.")

        if embedding.shape[1] != len(all_columns):
            raise ValueError(
                "Calculated embedding width does not match the expected "
                f"columns through r={max_order}: got {embedding.shape[1]}, "
                f"expected {len(all_columns)}."
            )

        # Attach names to the complete FEATHER output, then keep only the
        # configured r blocks. For orders = [[1], [7]], the CSV contains only
        # columns beginning with r1_ and r7_.
        table = pd.DataFrame(
            embedding,
            index=pd.Index(node_ids, name="node_id"),
            columns=all_columns,
        ).loc[:, expected_columns]

        embedding_folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(embedding_path, index=True)

        del table, inputs, model, embedding
        print(
            f"Saved {FULL_EMBEDDING_VARIANT}: {embedding_path}",
            flush=True,
        )


    print(f"Finished {place} (distance and embedding CSVs ready).", flush=True)


def run_pipeline():
    global CLIP_BUFFER_M, NEAREST_POIS

    config = load_config(CONFIG_PATH)
    CLIP_BUFFER_M = config["CLIP_BUFFER_M"]
    NEAREST_POIS = config["NEAREST_POIS"]

    multiplier = config["outlier_iqr_multiplier"]
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("outlier_iqr_multiplier must be finite and positive.")

    orders = configured_orders(config)

    if min(
        CLIP_BUFFER_M,
        NEAREST_POIS,
        config["eval_points"],
        min(orders),
    ) <= 0:
        raise ValueError(
            "Buffer, POI count, orders and evaluation count must be positive."
        )

    if not config["places"] or not config["resolved_categories"]:
        raise ValueError(
            "Configure at least one place and amenity category in input/config.txt."
        )

    print("\nFEATHER embedding pipeline", flush=True)
    print(
        "Using cached OSM files and walking graphs; this stage does not download OSM data.",
        flush=True,
    )

    for place in config["places"]:
        process_city(place, config)
        gc.collect()

    print(
        "Pipeline finished. Run pipeline/embedding_variants.py to create "
        "the three mean-based node-level embedding variants.",
        flush=True,
    )


if __name__ == "__main__":
    run_pipeline()
