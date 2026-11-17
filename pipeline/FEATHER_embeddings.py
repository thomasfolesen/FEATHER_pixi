"""Generate FEATHER embeddings, POI distance CSVs and geometry cache; no plots."""
import gc
import json
import pickle
import re
import subprocess
import time
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pandana as pdna
import requests
from pyrosm import OSM
from shapely.geometry import shape
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.pipeline_common import (load_config, safe_filename, embedding_columns)

CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "output"
GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
GEOFABRIK_WAIT_SECONDS = 60 # change this if server connection fails due to repeated tries / overload to server
NETWORK_TYPE = "walking"
TARGET_CONTINENTS = {"north-america", "europe", "asia"}
print("Starting pipeline/FEATHER_embeddings.py")

def get_geofabrik_index():
    """

    Get geofabrik index

    put into: cache/geofabrik/index-v1.json

    """
    index_path = CACHE_ROOT / "geofabrik" / "index-v1.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_path.exists():
        print(
            f"Waiting {GEOFABRIK_WAIT_SECONDS} seconds "
            "before downloading the Geofabrik index...",
            flush=True,
        )
        time.sleep(GEOFABRIK_WAIT_SECONDS)
        response = requests.get(GEOFABRIK_INDEX_URL, timeout=60)
        response.raise_for_status()
        index_path.write_bytes(response.content)
        print("Geofabrik index downloaded.", flush=True)
    else:
        print("Using cached Geofabrik index.", flush=True)
    return json.loads(index_path.read_text(encoding="utf-8"))

def city_boundaries(place):
    """

    Load cached boundaries for the given place, or create and save them.

    OSMnx fetches the municipality boundary if it is not cached.

    A second boundary is expanded outward by CLIP_BUFFER_M metres (2000).

    Files are saved in cache/place:

    - municipality_boundary.geojson

    - municipality_clip_boundary_{CLIP_BUFFER_M}m.geojson

    """
    city_dir = CACHE_ROOT / safe_filename(place)
    city_dir.mkdir(parents=True, exist_ok=True)

    exact_path = city_dir / "municipality_boundary.geojson"
    clip_path = city_dir / f"municipality_clip_boundary_{CLIP_BUFFER_M}m.geojson"

    if exact_path.exists():
        exact = gpd.read_file(exact_path).to_crs("EPSG:4326")
    else:
        geocoded = ox.geocoder.geocode_to_gdf(place).to_crs("EPSG:4326")
        exact = gpd.GeoDataFrame(
            {"place": [place]},
            geometry=[geocoded.geometry.union_all()],
            crs="EPSG:4326",
        )
        exact.to_file(exact_path, driver="GeoJSON")

    if clip_path.exists():
        clip = gpd.read_file(clip_path).to_crs("EPSG:4326")
    else:
        clip = exact.to_crs(exact.estimate_utm_crs())
        clip.geometry = clip.geometry.buffer(CLIP_BUFFER_M)
        clip = clip.to_crs("EPSG:4326")
        clip.to_file(clip_path, driver="GeoJSON")

    return {
        "city_dir": city_dir,
        "exact_boundary_path": exact_path,
        "clip_boundary": clip,
        "clip_boundary_path": clip_path,
    }

def select_source(index, clip_boundary):
    """

    Choose a European Geofabrik extract for the buffered city boundary.

    Prefer the smallest region covering the whole boundary.

    If none exist, choose the smallest region covering a point inside it.

    """
    # Look up each region's properties by its ID.
    properties = {
        feature["properties"]["id"]: feature["properties"]
        for feature in index["features"]
    }
    def under_continent(region_id, targets):
        # Follow parent regions upwards, f.x. Denmark -> Europe.
        while region_id:
            if region_id in targets:
                return True
            region_id = properties[region_id].get("parent")
        return False
    # Combine the buffered boundary into one geometry.
    city_geometry = clip_boundary.geometry.union_all()
    # Choose a point guaranteed to lie inside the boundary.
    city_point = city_geometry.representative_point()
    candidates = []
    for feature in index["features"]:
        prop = feature["properties"]
        region_id = prop["id"]
        # Read the PBF download URL, if available.
        pbf_url = prop.get("urls", {}).get("pbf")
        # Skip regions without a PBF download or outside Europe.
        if not pbf_url or not under_continent(region_id, TARGET_CONTINENTS):
            continue
        # Convert the region's GeoJSON geometry into a Shapely geometry.
        region = shape(feature["geometry"])
        # Keep regions covering the point inside the city.
        if region.covers(city_point):
            candidates.append({
                "source_id": region_id,
                "source_name": prop["name"],
                "source_url": pbf_url,
                "covers_full_city": region.covers(city_geometry),
                "area": region.area,
            })
    # Find candidates covering the entire buffered boundary.
    full_city = [
        candidate
        for candidate in candidates
        if candidate["covers_full_city"]
    ]
    # Use full coverage when available; otherwise use the point-based candidates.
    # Select by geographic area, not download file size.
    return min(full_city or candidates, key=lambda candidate: candidate["area"])

def build_download_plan(places, index):
    """Create a table of source regions, cache paths, and cached city PBF files."""
    rows = []
    for place in places:
        # Load or create the city's original and buffered boundaries.
        boundary = city_boundaries(place)
        # Choose a Geofabrik region using the buffered boundary.
        source = select_source(index, boundary["clip_boundary"])
        # Set the path where the clipped city data will be stored.
        city_pbf_path = (
            boundary["city_dir"]
            / f"{safe_filename(place)}_buffer_{CLIP_BUFFER_M}m.osm.pbf"
        )
        # Add one row describing this city's download plan.
        rows.append({
            "place": place,
            "city_dir": boundary["city_dir"],
            "exact_boundary_path": boundary["exact_boundary_path"],
            "clip_boundary_path": boundary["clip_boundary_path"],
            **source,  # Insert all fields returned by select_source().
            "city_pbf_path": city_pbf_path,
            "city_pbf_exists": city_pbf_path.exists(),
        })
    return pd.DataFrame(rows)

def download_pbf(url, destination):
    if destination.exists():
        print(f"Using cached file for {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Waiting {GEOFABRIK_WAIT_SECONDS} seconds before downloading {destination.name}", flush=True)
    time.sleep(GEOFABRIK_WAIT_SECONDS) 
    with requests.get(url, stream=True, timeout=900) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8000000): #investigate chunksize 
                file.write(chunk)
    print(f"Downloaded: {destination}", flush=True)

def clip_pbf(regional_pbf, polygon_path, output_pbf):
    """Clip a regional PBF to the boundary in polygon_path.

    Using osmium via a subprocess  https://osmcode.org/libosmium/ """
    subprocess.run(
        [
            "osmium", "extract",
            "--polygon", str(polygon_path),
            "--strategy=complete_ways",
            "--set-bounds",
            "--output", str(output_pbf),
            "--overwrite",
            str(regional_pbf),
        ],
        check=True,
    )

def prepare_city_pbfs(download_plan):
    """Create missing city PBFs, return summary metadata"""
    plan = download_plan.copy()
    regional_dir = CACHE_ROOT / "geofabrik" / "regional_temp"

    plan["status"] = "cached_city_pbf"

    for row_index, city in plan.iterrows():
        city_pbf = city["city_pbf_path"]
        if city_pbf.exists():
            continue
        regional_pbf = (
            regional_dir / f"{safe_filename(city['source_id'])}.osm.pbf"
        )
        download_pbf(city["source_url"], regional_pbf)
        print(f"Clipping {city['place']}")
        clip_pbf(regional_pbf, city["clip_boundary_path"], city_pbf)
        plan.loc[row_index, "status"] = "downloaded_clipped"

    plan["city_pbf_mb"] = plan["city_pbf_path"].map(
        lambda path: path.stat().st_size / 1024**2
    )

    return plan[[
        "place",
        "status",
        "source_name",
        "source_url",
        "city_pbf_path",
        "city_pbf_mb",
    ]]

def build_graph_from_pbf(place, city_pbf_path, city_dir):
    """Build or load a simplified, undirected walking graph."""
    started = time.time()
    cache_path = (
        city_dir
        / f"city_graph_{NETWORK_TYPE}_buffer_{CLIP_BUFFER_M}m_simplified_undirected.pkl"
    )

    if cache_path.exists():
        print(f"Loading cached graph for {place}...", flush=True)

        with cache_path.open("rb") as file:
            graph = pickle.load(file)

        status = "cached"

    else:
        print(f"Building graph for {place}...", flush=True)

        osm = OSM(
            str(city_pbf_path),
            engine="out_of_core",
            workers="auto",
        )
        network_nodes, network_edges = osm.get_network(
            network_type=NETWORK_TYPE,
            nodes=True,
        )

        # Create the graph
        graph = osm.to_graph(
            network_nodes,
            network_edges,
            graph_type="networkx",
            network_type=NETWORK_TYPE,
            simplify=False,
        )
        graph.graph["crs"] = "EPSG:4326"

        # simplify
        graph = ox.simplification.simplify_graph(graph)

        # make undirected for less data to compute
        graph = ox.convert.to_undirected(graph)

        # Remove self loops
        graph.remove_edges_from(list(nx.selfloop_edges(graph, keys=True)))

        # Keep only the largest connected component
        largest = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()

        # Cache the graph
        with cache_path.open("wb") as file:
            pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)

        status = "built"

    # Convert longitude and latitude to projection in metres.
    projected_graph = ox.project_graph(graph)

    # Create spatial tables of nodes and edges
    nodes, edges = ox.graph_to_gdfs(projected_graph)
    runtime = time.time() - started

    print(f"Graph ready for {place} in {runtime:.1f} seconds.", flush=True)

    return {
        "graph": graph,
        "projected_graph": projected_graph,
        "nodes": nodes,
        "edges": edges,
        "graph_cache_path": cache_path,
        "graph_status": status,
        "graph_runtime_seconds": runtime,
    }

def extract_pois_from_city_pbf(
    place, city_pbf_path, city_dir, exact_boundary_path, resolved_categories
):
    """Extract configured POIs inside the buffered boundary and graph, or load cached POIs."""
    poi_path = city_dir / f"accessibility_pois_buffer_{CLIP_BUFFER_M}m.geojson"

    if poi_path.exists():
        print(f"Loading cached POIs for {place}")
        features = gpd.read_file(poi_path)
        status = "cached"

    else:
        # Combine tags from all categories into one extraction filter.
        tags_by_key = {}
        for mappings in resolved_categories.values():
            for item in mappings:
                tags_by_key.setdefault(item["osm_key"], []).append(item["osm_tag"])

        print(f"Extracting POIs for {place}")
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

            # Represent buildings and other areas by a point inside them.
            raw["geometry"] = raw.geometry.representative_point()

            # Keep POIs whose point lies within or on the municipality boundary.
            boundary = (
                gpd.read_file(exact_boundary_path)
                .to_crs("EPSG:4326")
                .geometry.union_all()
            )
            raw = raw[raw.geometry.apply(boundary.covers)]

            # Create a row for each matching category and tag.
            for category, mappings in resolved_categories.items():
                for item in mappings:
                    key = item["osm_key"]
                    tag = item["osm_tag"]

                    if key not in raw.columns:
                        continue

                    matches = raw[raw[key] == tag]
                    for _, poi in matches.iterrows():
                        rows.append({
                            "osm_type": poi.get("osm_type"),
                            "osm_id": poi.get("id"),
                            "category": category,
                            "osm_key": key,
                            "osm_tag": tag,
                            "name": poi.get("name"),
                            "geometry": poi["geometry"],
                        })

        features = gpd.GeoDataFrame(
            rows,
            columns=[
                "osm_type", "osm_id", "category",
                "osm_key", "osm_tag", "name", "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )

        # Remove repeated matches while preserving different categories and tags.
        features = features.drop_duplicates(
            subset=["osm_type", "osm_id", "category", "osm_key", "osm_tag"]
        ).reset_index(drop=True)

        if not features.empty:
            features.to_file(poi_path, driver="GeoJSON")

        status = "extracted"

    features_by_category = {
        category: features[features["category"] == category].copy()
        for category in resolved_categories
    }

    # Include categories with zero matching POIs.
    category_counts = pd.DataFrame({
        "category": list(features_by_category),
        "count": [len(pois) for pois in features_by_category.values()],
    })

    tag_counts = (
        features.groupby(["category", "osm_key", "osm_tag"])
        .size()
        .reset_index(name="count")
    )

    return {
        "features": features,
        "features_by_category": features_by_category,
        "category_counts": category_counts,
        "tag_counts": tag_counts,
        "poi_status": status,
        "poi_cache_path": poi_path,
    }

def calculate_poi_distances(graph_data, pois, categories):
    categories = list(categories)
    nodes = graph_data["nodes"]
    # Keep the shortest parallel edge for each node pair.
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
        raise ValueError(
            "Road lengths must be finite, non-negative metres."
        )
    print("Preparing the buffered walking network...", flush=True)
    network = pdna.Network(
        node_x=nodes.geometry.x,
        node_y=nodes.geometry.y,
        edge_from=roads["u"],
        edge_to=roads["v"],
        edge_weights=roads[["length"]],
        twoway=True,
    )
    distances = pd.DataFrame(
        index=nodes.index,
        columns=list(categories),
        dtype=float,
    )
    poi_counts = pd.DataFrame(
        0, index=nodes.index, columns=list(categories), dtype=int
    )
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
        # Missing neighbours are returned as -1.
        nearest = nearest.where(
            (nearest >= 0) & (nearest <= CLIP_BUFFER_M)
        )
        # Average available neighbours; no neighbours gives NaN.
        distances[category] = nearest.mean(axis=1)
        # Count only the reachable POIs used in the mean, up to NEAREST_POIS.
        poi_counts[category] = nearest.notna().sum(axis=1)
    return distances, poi_counts

def fill_distances(distances):
    """Use the same missing-distance replacement for totals and FEATHER."""
    features = distances.copy()
    if np.isinf(features.to_numpy()).any():
        raise ValueError("Distances contain infinite values.")
    if (features < 0).any().any():
        raise ValueError("Distances cannot be negative.")
    # Treat missing amenities and distances beyond the search limit equally.
    return features.clip(
        upper=CLIP_BUFFER_M
    ).fillna(CLIP_BUFFER_M)

def remove_distance_outliers(graph, scores, iqr_multiplier=1.5):
    """Remove nodes outside Tukey's IQR fences, then isolated nodes."""
    if not np.isfinite(iqr_multiplier) or iqr_multiplier <= 0:
        raise ValueError("iqr_multiplier must be finite and positive.")

    if scores.empty or not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Outlier scores must be non-empty and finite.")

    q1 = scores.quantile(0.25)
    q3 = scores.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr

    excluded = pd.Series(
        "", index=scores.index, name="feather_exclusion"
    )
    excluded.loc[scores < lower] = "bottom_outlier"
    excluded.loc[scores > upper] = "top_outlier"

    bottom_count = int((excluded == "bottom_outlier").sum())
    top_count = int((excluded == "top_outlier").sum())

    filtered = graph.subgraph(
        excluded.index[excluded == ""]
    ).copy()

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
    node_ids = list(graph.nodes)
    categories = list(categories)
    features = fill_distances(distances.reindex(index=node_ids, columns=categories))
    node_numbers = {
        node_id: position
        for position, node_id in enumerate(node_ids)
    }
    numbered_graph = nx.relabel_nodes(graph, node_numbers, copy=True)
    return {
        "graph": numbered_graph,
        "features": features.to_numpy(dtype=float) / CLIP_BUFFER_M,
        "node_ids": node_ids,
        "categories": list(categories),
    }

def municipality_data(graph_data, boundary_path):
    boundary = (
        gpd.read_file(boundary_path)
        .to_crs(graph_data["nodes"].crs)
        .geometry.union_all()
    )
    nodes = graph_data["nodes"]
    node_ids = nodes.index[nodes.geometry.apply(boundary.covers)]
    edges = graph_data["edges"]
    keep = (
        edges.index.get_level_values("u").isin(node_ids)
        & edges.index.get_level_values("v").isin(node_ids)
    )
    return {
        "graph": graph_data["graph"].subgraph(node_ids).copy(),
        "nodes": nodes.loc[node_ids].copy(),
        "edges": edges.loc[keep].copy(),
    }

def save_interactive_data(place, graph_data, distances, boundary_path):
    nodes = graph_data["nodes"][["geometry"]]
    boundary = (
        gpd.read_file(boundary_path)
        .to_crs(nodes.crs)
        .geometry.union_all()
    )
    data = {
        "place": place,
        "nodes": nodes,
        "edges": gpd.clip(graph_data["edges"][["geometry"]], boundary),
        "distances": distances,
        "CLIP_BUFFER_M": CLIP_BUFFER_M,
        "NEAREST_POIS": NEAREST_POIS,
    }
    folder = CACHE_ROOT / "interactive_map_data"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{safe_filename(place)}.pkl"
    with path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def process_city(city, config):
    place = city["place"]
    categories = config["resolved_categories"]
    order = config["order"]
    eval_points = config["eval_points"]
    place_name = safe_filename(place)


    summary_folder = OUTPUT_ROOT / "POI_distances" / place_name
    summary_path = summary_folder / f"{place_name}_poi_distances.csv"


    embedding_folder = OUTPUT_ROOT / "FEATHER_embeddings" / place_name
    embedding_path = (
        embedding_folder / f"{place_name}_feather_embeddings.csv"
    )


    metadata_path = embedding_path.with_suffix(".json")
    if embedding_path.is_file() and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("theta_max", "CLIP_BUFFER_M", "NEAREST_POIS", "outlier_iqr_multiplier"):
            if previous[key] != config[key]:
                raise ValueError(
                    f"Saved embeddings use a different {key}. "
                    f"Move or delete {embedding_path} to recompute them."
                )
    elif embedding_path.is_file():
        print("Warning: legacy embeddings have no parameter metadata; "
              "their theta and distance settings cannot be verified. "
              "Reuse assumes the current config produced this CSV.", flush=True)

    print(f"\nProcessing {place}...", flush=True)

    buffered = build_graph_from_pbf(
        place, city["city_pbf_path"], city["city_dir"]
    )

    poi_data = extract_pois_from_city_pbf(
        place,
        city["city_pbf_path"],
        city["city_dir"],
        city["clip_boundary_path"],
        categories,
    )

    municipality = municipality_data(
        buffered, city["exact_boundary_path"]
    )

    # Calculate distances on the original buffered road network.
    print("Calculating walking distances...", flush=True)

    distances, poi_counts = calculate_poi_distances(
        buffered, poi_data["features"], categories
    )

    distances = distances.reindex(municipality["nodes"].index)
    poi_counts = poi_counts.reindex(municipality["nodes"].index)

    distance_summary = pd.DataFrame(index=distances.index)
    distance_summary.index.name = "node_id"

    for category in categories:
        distance_summary[f"{category}_mean_distance_m"] = (
            distances[category]
        )
        distance_summary[
            f"{category}_poi_count_up_to_{NEAREST_POIS}"
        ] = poi_counts[category]

    # Give each amenity category equal weight.
    filled = fill_distances(distances[list(categories)])
    distance_summary["total_distance_m"] = filled.sum(axis=1)
    distance_summary["average_distance_m"] = filled.mean(axis=1)

    del filled, poi_counts, buffered, poi_data
    gc.collect()

    # Remove IQR outliers, then any isolated nodes.
    graph, excluded = remove_distance_outliers(
        municipality["graph"],
        distance_summary["average_distance_m"],
        config["outlier_iqr_multiplier"],
    )

    # Retain all original nodes in the CSV with current exclusion labels.
    distance_summary["feather_exclusion"] = excluded

    summary_folder.mkdir(parents=True, exist_ok=True)
    distance_summary.to_csv(summary_path, index=True)
    print(f"Saved distance CSV: {summary_path}", flush=True)

    # Restrict both map nodes and edges to the filtered graph.
    node_ids = list(graph.nodes)
    edges = municipality["edges"]

    keep_edges = (
        edges.index.get_level_values("u").isin(node_ids)
        & edges.index.get_level_values("v").isin(node_ids)
    )

    municipality = {
        "graph": graph,
        "nodes": municipality["nodes"].loc[node_ids].copy(),
        "edges": edges.loc[keep_edges].copy(),
    }
    del edges, keep_edges

    # Keep original-network distances for the retained nodes.
    distances = distances.loc[node_ids].copy()

    # Store filtered geometry for graph_pics.py; no pictures are rendered here.
    save_interactive_data(place, municipality, distances, city["exact_boundary_path"])
    if not node_ids:
        raise ValueError(f"No nodes remain after filtering for {place}.")

    del distance_summary, excluded

    expected_columns = list(
        embedding_columns(categories, order, eval_points)
    )

    # Load saved embeddings instead of running FEATHER again.
    if embedding_path.is_file():
        print(
            f"Loading saved FEATHER embeddings: {embedding_path}",
            flush=True,
        )

        # Preserve node IDs when reading the CSV.
        index_column = pd.read_csv(
            embedding_path, nrows=0
        ).columns[0]

        table = pd.read_csv(
            embedding_path,
            dtype={index_column: str},
            index_col=0,
        )
        table.index = table.index.map(str)

        expected_index = pd.Index([str(node) for node in node_ids])

        if not table.index.is_unique or not table.columns.is_unique:
            raise ValueError(
                "Saved embeddings contain duplicate rows or columns: "
                f"{embedding_path}"
            )

        if set(table.columns) != set(expected_columns):
            raise ValueError(
                "Saved embedding columns do not match the current "
                f"categories, order, or evaluation points: {embedding_path}. "
                "Move or delete this CSV to calculate fresh embeddings."
            )

        if set(table.index) != set(expected_index):
            raise ValueError(
                "Saved embedding nodes do not match the current "
                f"filtered graph: {embedding_path}. "
                "Move or delete this CSV to calculate fresh embeddings."
            )

        # Match saved rows and columns to the current expected order.
        embedding = table.loc[
            expected_index, expected_columns
        ].to_numpy(dtype=float)

        if not np.isfinite(embedding).all():
            raise ValueError(
                "Saved embeddings contain missing or non-finite "
                f"values: {embedding_path}"
            )

        del table, municipality, graph

    else:
        print(
            f"Calculating FEATHER embeddings through r = {order}...",
            flush=True,
        )

        inputs = prepare_feather_inputs(graph, distances, categories)
        del municipality, graph

        from FEATHER.feather import FEATHER_new

        model = FEATHER_new(
            theta_max=config["theta_max"],
            eval_points=eval_points,
            order=order,
        )
        model.fit(graph=inputs["graph"], X=inputs["features"])

        embedding = model.get_embedding()
        node_ids = list(inputs["node_ids"])

        table = pd.DataFrame(
            embedding,
            index=pd.Index(node_ids, name="node_id"),
            columns=expected_columns,
        )

        embedding_folder.mkdir(parents=True, exist_ok=True)
        table.to_csv(embedding_path, index=True)

        del table, inputs, model

        print(
            f"Saved FEATHER embeddings: {embedding_path}",
            flush=True,
        )

    metadata = {
        "schema_version": 1,
        "place": place,
        "categories": list(categories),
        "order": order,
        "eval_points": eval_points,
        "theta_max": config["theta_max"],
        "CLIP_BUFFER_M": CLIP_BUFFER_M,
        "NEAREST_POIS": NEAREST_POIS,
        "outlier_iqr_multiplier": config["outlier_iqr_multiplier"],
        "high_low_percent": config.get("high_low_percent", 10),
    }
    embedding_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Finished {place} (CSV files and plotting cache saved).", flush=True)


def run_pipeline():
    global CLIP_BUFFER_M, NEAREST_POIS

    config = load_config(CONFIG_PATH)

    multiplier = config["outlier_iqr_multiplier"]
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(
            "outlier_iqr_multiplier must be finite and positive."
        )

    CLIP_BUFFER_M = config["CLIP_BUFFER_M"]
    NEAREST_POIS = config["NEAREST_POIS"]

    if min(
        CLIP_BUFFER_M,
        NEAREST_POIS,
        config["order"],
        config["eval_points"],
    ) <= 0:
        raise ValueError(
            "Buffer, POI count, order and evaluation count must be positive."
        )

    if not config["places"] or not config["resolved_categories"]:
        raise ValueError(
            "Configure at least one place and amenity category."
        )

    print("\nFEATHER pipeline", flush=True)
    print(
        "Preparing city files; existing downloads will be reused...",
        flush=True,
    )

    index = get_geofabrik_index()
    plan = build_download_plan(config["places"], index)

    # Prepare missing city files before processing cities individually.
    prepare_city_pbfs(plan)
    del index

    for city in plan.to_dict(orient="records"):
        process_city(city, config)
        gc.collect()
    print("Pipeline finished. Run the plotting scripts pipeline/graph_pics.py and pipeline/FEATHER_curves_pics.py pipeline/FEATHER_curves_pics.py separately.", flush=True)


if __name__ == "__main__":
    run_pipeline()
