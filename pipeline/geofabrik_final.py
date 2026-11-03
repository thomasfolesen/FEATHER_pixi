"""Pipeline document which takes a config file "input/config.txt" and outputs FEATHER_new embeddings etc. in "output/" folder."""
import gc
import json
import pickle
import re
import subprocess
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pandana as pdna
import requests
from pyrosm import OSM
from shapely.geometry import shape
from FEATHER.feather import FEATHER_new
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "output"
GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
GEOFABRIK_WAIT_SECONDS = 60 # change this if server connection fails due to repeated tries / overload to server
NETWORK_TYPE = "walking"
FORCE_REBUILD_DERIVED = False
DELETE_REGIONAL_PBF = False  # Keep regions for future cities.

def parse_config_list(value):
    """Read items inside [[...], [...]], preserving commas within each item. using regex"""
    return [
        item.strip()
        for item in re.findall(r"\[([^\[\]]+)\]", value)
    ]

def load_config(path):
    config = {"resolved_categories": {}, "high_low_percent": 10.0}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "place":
            config["places"] = parse_config_list(value)
        elif key in ("theta_max", "high_low_percent"):
            config[key] = float(value)
        elif key in ("eval_points", "order", "CLIP_BUFFER_M", "NEAREST_POIS"):
            config[key] = int(value)
        else:
            tags = []
            for entry in parse_config_list(value):
                osm_key, osm_tag = entry.split("=", 1)
                tags.append({
                    "osm_key": osm_key.strip(),
                    "osm_tag": osm_tag.strip(),
                })
            config["resolved_categories"][key] = tags
    return config

def safe_filename(place):
    return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()

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
    osm_cache = city_dir / "osmnx"
    osm_cache.mkdir(parents=True, exist_ok=True)
    exact_path = city_dir / "municipality_boundary.geojson"
    clip_path = city_dir / f"municipality_clip_boundary_{CLIP_BUFFER_M}m.geojson"
    if exact_path.exists():
        exact = gpd.read_file(exact_path).to_crs("EPSG:4326")
    else:
        # Use this city's cache while looking up its boundary.
        previous_cache = ox.settings.cache_folder
        previous_use_cache = ox.settings.use_cache
        try:
            ox.settings.cache_folder = str(osm_cache)
            ox.settings.use_cache = True
            geocoded = ox.geocoder.geocode_to_gdf(place).to_crs("EPSG:4326")
        finally:
            ox.settings.cache_folder = previous_cache
            ox.settings.use_cache = previous_use_cache
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
    def under_europe(region_id):
        # Follow parent regions upwards, e.g. Denmark -> Europe.
        while region_id:
            if region_id == "europe":
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
        if not pbf_url or not under_europe(region_id):
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
    print(f"Waiting {GEOFABRIK_WAIT_SECONDS} seconds before downloading {destination.name}...", flush=True)
    time.sleep(GEOFABRIK_WAIT_SECONDS)
    with requests.get(url, stream=True, timeout=900) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                file.write(chunk)
    print(f"Downloaded: {destination}", flush=True)

def clip_pbf(regional_pbf, polygon_path, output_pbf):
    """Clip a regional PBF to the boundary in polygon_path."""
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
    """Create missing city PBFs and return their paths, status, and size."""
    plan = download_plan.copy()
    plan["status"] = "cached_city_pbf"
    regional_dir = CACHE_ROOT / "geofabrik" / "regional_temp"
    # Group cities so they can share the same regional download.
    for source_url, cities in plan.groupby("source_url", sort=False):
        source_id = cities.iloc[0]["source_id"]
        regional_pbf = regional_dir / f"{safe_filename(source_id)}.osm.pbf"
        for row_index, city in cities.iterrows():
            city_pbf = city["city_pbf_path"]
            if city_pbf.exists():
                continue
            download_pbf(source_url, regional_pbf)
            print(f"Clipping {city['place']}")
            clip_pbf(regional_pbf, city["clip_boundary_path"], city_pbf)
            plan.loc[row_index, "status"] = "downloaded_clipped"
        if DELETE_REGIONAL_PBF:
            regional_pbf.unlink(missing_ok=True)
    # Calculate each city file's size in MiB.
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

    if cache_path.exists() and not FORCE_REBUILD_DERIVED:
        print(f"Loading cached graph for {place}...", flush=True)

        with cache_path.open("rb") as file:
            graph = pickle.load(file)

        print("Cached graph loaded.", flush=True)
        status = "cached"

    else:
        print(f"Building graph for {place}...", flush=True)

        print("Opening city PBF...", flush=True)
        osm = OSM(
            str(city_pbf_path),
            engine="out_of_core",
            workers="auto",
        )
        print("PBF reader ready.", flush=True)

        print("Reading walking-network nodes and edges...", flush=True)
        network_nodes, network_edges = osm.get_network(
            network_type=NETWORK_TYPE,
            nodes=True,
        )
        print(
            f"Network read: {len(network_nodes):,} nodes, "
            f"{len(network_edges):,} edges.",
            flush=True,
        )

        print("Converting the network to a NetworkX graph...", flush=True)
        graph = osm.to_graph(
            network_nodes,
            network_edges,
            graph_type="networkx",
            network_type=NETWORK_TYPE,
            simplify=False,
        )
        graph.graph["crs"] = "EPSG:4326"
        print("NetworkX graph created.", flush=True)

        # Simplify road segments and allow travel in both directions.
        print("Simplifying road segments...", flush=True)
        graph = ox.simplification.simplify_graph(graph)
        print(
            f"Simplification finished: {graph.number_of_nodes():,} nodes, "
            f"{graph.number_of_edges():,} edges.",
            flush=True,
        )

        print("Converting the graph to undirected...", flush=True)
        graph = ox.convert.to_undirected(graph)
        print("Undirected graph ready.", flush=True)

        # Remove edges connecting a node to itself.
        print("Removing self-loop edges...", flush=True)
        graph.remove_edges_from(list(nx.selfloop_edges(graph, keys=True)))
        print("Self-loop edges removed.", flush=True)

        # Keep the largest connected group of nodes.
        print("Finding the largest connected group of nodes...", flush=True)
        largest = max(nx.connected_components(graph), key=len)
        print(
            f"Largest group found: {len(largest):,} nodes. "
            "Copying this part of the graph...",
            flush=True,
        )
        graph = graph.subgraph(largest).copy()
        print("Largest connected graph ready.", flush=True)

        print(f"Saving graph cache: {cache_path}", flush=True)
        with cache_path.open("wb") as file:
            pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)
        print("Graph cache saved.", flush=True)
        status = "built"

    # Convert coordinates to a local projection measured in metres.
    print("Projecting graph coordinates to metres...", flush=True)
    projected_graph = ox.project_graph(graph)
    print("Graph projection finished.", flush=True)

    print("Creating node and edge GeoDataFrames...", flush=True)
    nodes, edges = ox.graph_to_gdfs(projected_graph)
    print("GeoDataFrames ready.", flush=True)

    print(
        f"Graph preparation finished for {place} "
        f"in {time.time() - started:.1f} seconds.",
        flush=True,
    )

    return {
        "graph": graph,
        "projected_graph": projected_graph,
        "nodes": nodes,
        "edges": edges,
        "graph_cache_path": cache_path,
        "graph_status": status,
        "graph_runtime_seconds": time.time() - started,
    }

def extract_pois_from_city_pbf(
    place, city_pbf_path, city_dir, exact_boundary_path, resolved_categories
):
    """Extract configured POIs inside the buffered boundary, or load cached POIs."""
    poi_path = (
        city_dir
        / f"accessibility_pois_buffer_{CLIP_BUFFER_M}m.geojson"
    )
    if poi_path.exists() and not FORCE_REBUILD_DERIVED:
        print(f"Loading cached POIs for {place}")
        features = gpd.read_file(poi_path)
        status = "cached"
    else:
        # Combine the configured tags into one extraction filter.
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
            # Represent each POI with a point, including buildings and areas.
            raw["geometry"] = raw.geometry.representative_point()
            # Keep POIs whose representative point is inside the buffered boundary.
            boundary = (
                gpd.read_file(exact_boundary_path)
                .to_crs("EPSG:4326")
                .geometry.union_all()
            )
            raw = raw[raw.geometry.apply(boundary.covers)]
            # Match POIs to their configured categories.
            for category, mappings in resolved_categories.items():
                for item in mappings:
                    key, tag = item["osm_key"], item["osm_tag"]
                    if key not in raw.columns:
                        continue
                    for _, poi in raw[raw[key] == tag].iterrows():
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
        features = features.drop_duplicates(
            subset=["osm_type", "osm_id", "category", "osm_key", "osm_tag"]
        ).reset_index(drop=True)
        if not features.empty:
            features.to_file(poi_path, driver="GeoJSON")
        status = "extracted"
    # Include every configured category, even those with no POIs.
    category_counts = pd.DataFrame({
        "category": list(resolved_categories),
        "count": [
            (features["category"] == category).sum()
            for category in resolved_categories
        ],
    })
    tag_counts = (
        features.groupby(["category", "osm_key", "osm_tag"])
        .size()
        .reset_index(name="count")
    )
    return {
        "features": features,
        "features_by_category": {
            category: features[features["category"] == category].copy()
            for category in resolved_categories
        },
        "category_counts": category_counts,
        "tag_counts": tag_counts,
        "poi_status": status,
        "poi_cache_path": poi_path,
    }

def calculate_poi_distances(graph_data, pois, categories):
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

def save_distance_map(place, category, graph_data, distances, folder):
    nodes = graph_data["nodes"].copy()
    nodes["distance"] = distances.reindex(nodes.index)
    category_name = category.replace("_", " ")
    folder.mkdir(parents=True, exist_ok=True)
    image_path = folder / f"distance_to_{category}.png"
    with plt.style.context("default"):
        fig, ax = plt.subplots(figsize=(14, 11))
        boundary = (
            gpd.read_file(
                CACHE_ROOT
                / safe_filename(place)
                / "municipality_boundary.geojson"
            )
            .to_crs(graph_data["edges"].crs)
            .geometry.union_all()
        )
        display_edges = gpd.clip(graph_data["edges"], boundary)
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
            f"{place}\n"
            f"Mean distance to up to {NEAREST_POIS} nearest "
            f"{category_name} POIs",
            fontsize=15,
        )
        ax.set_axis_off()
        fig.text(
            0.5,
            0.02,
            (
                f"Mean over up to {NEAREST_POIS} POIs reachable within "
                f"{CLIP_BUFFER_M:,} m; unavailable neighbours are excluded.\n"
                "Routes and POIs may extend outside the municipality. "
                "POI-to-network snapping distance is excluded."
            ),
            ha="center",
            fontsize=10,
        )
        fig.savefig(
            image_path,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
        plt.close(fig)
    return image_path

def prepare_feather_inputs(graph, distances, categories):
    node_ids = list(graph.nodes)
    features = distances.reindex(
        index=node_ids,
        columns=list(categories),
    )
    if np.isinf(features.to_numpy()).any():
        raise ValueError("Distances contain infinite values.")
    if (features < 0).any().any():
        raise ValueError("Distances cannot be negative.")
    # Treat missing amenities and distances beyond the search limit equally.
    features = features.clip(
        upper=CLIP_BUFFER_M
    ).fillna(CLIP_BUFFER_M)
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

def embedding_columns(categories, order, eval_points):
    # Same order as FEATHER: r, component, category, evaluation point.
    return [
        f"r{r}_{category}_eval{point}_{component}"
        for r in range(1, order + 1)
        for component in ("real", "imaginary")
        for category in categories
        for point in range(1, eval_points + 1)
    ]

def save_curves(series, theta, title, destination):
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
        ax.set_title(label)
        ax.set_xlabel("Evaluation point θ")
        ax.set_ylabel("Mean across nodes")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

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
    print(f"Saved interactive map data: {path}", flush=True)

def move_existing_mean_curves():
    # Move old figures once, keeping their city folders.
    root = OUTPUT_ROOT / "FEATHER_curves"
    mean_folder = root / "FEATHER_curves_mean"
    if not root.exists():
        return
    moved = 0
    for path in list(root.rglob("*.png")):
        relative = path.relative_to(root)
        if relative.parts[0] in ("FEATHER_curves_mean", "FEATHER_curves_high_low"):
            continue
        destination = mean_folder / relative
        if destination.exists():
            print(f"Keeping older duplicate in place: {path}", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
        moved += 1
    for folder in list(root.iterdir()):
        if folder.is_dir() and folder.name not in (
            "FEATHER_curves_mean", "FEATHER_curves_high_low"
        ) and not any(folder.iterdir()):
            folder.rmdir()
    if moved:
        print(f"Moved {moved} existing figures to: {mean_folder}", flush=True)



def high_low_summary(values, distances, node_ids, categories, percent):
    # Choose nodes by original distances, before missing values are filled.
    summaries = {}
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
            positions = pd.Index(node_ids).get_indexer(selected)
            means, deviations = [], []
            for r in range(values.shape[1]):
                block = values[positions, r, :, category_index, :]
                means.append(block.mean(axis=0))
                deviations.append(block.std(axis=0, ddof=0))
            summaries[category][group] = {
                "mean": np.asarray(means),
                "std": np.asarray(deviations),
                "count": count,
                "distance_min": float(scores.loc[selected].min()),
                "distance_max": float(scores.loc[selected].max()),
            }
    return summaries



def draw_band(ax, theta, mean, std, label, color):
    ax.plot(theta, mean, color=color, label=label, linewidth=1.5)
    ax.fill_between(theta, mean - std, mean + std, color=color, alpha=0.18)



def finish_band_figure(fig, axes, title, destination):
    for r, row in enumerate(axes):
        for component, ax in enumerate(row):
            prefix = f"r = {r + 1} — " if len(axes) > 1 else ""
            ax.set_title(prefix + ["Real (cosine)", "Imaginary (sine)"][component])
            ax.set_xlabel("Evaluation point θ")
            ax.set_ylabel("Mean across selected nodes")
            ax.axhline(0, color="grey", linewidth=0.6)
            ax.grid(alpha=0.2)
        row[0].legend(fontsize=8)
    # Leave the band uncut; mean ± SD may extend beyond [-1, 1].
    fig.suptitle(title)
    fig.text(
        0.5, 0.01, "Shading: mean ± 1 standard deviation across selected nodes",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def save_high_low_city(place, summaries, theta, percent, folder):
    for category, groups in summaries.items():
        order = groups["low"]["mean"].shape[0]
        fig, axes = plt.subplots(
            order, 2, figsize=(12, 3.5 * order), squeeze=False, sharex=True, sharey=True
        )
        for r in range(order):
            for group, color in [("low", "tab:blue"), ("high", "tab:red")]:
                data = groups[group]
                for component, ax in enumerate(axes[r]):
                    draw_band(
                        ax, theta, data["mean"][r, component], data["std"][r, component],
                        (
                            f"{group.title()} distance (n={data['count']})\n"
                            f"Mean walking distance: "
                            f"{data['distance_min']:.1f}–{data['distance_max']:.1f} m"
                        ), color,
                    )
        finish_band_figure(
            fig, axes,
            f"{place} — {category.replace('_', ' ')}\nLowest and highest {percent:g}% of finite distances",
            folder / f"feather_{category}.png",
        )



def save_high_low_comparisons(city_groups, categories, theta, order, percent, folder):
    # Keep each city's colour the same across all panels.
    colors = {
        place: plt.get_cmap("tab10")(index % 10)
        for index, place in enumerate(city_groups)
    }
    folder.mkdir(parents=True, exist_ok=True)
    for category in categories:
        available = {
            place: summaries[category]
            for place, summaries in city_groups.items()
            if category in summaries
        }
        if not available:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        # High distances on top, low distances on the bottom.
        for row, group in enumerate(("high", "low")):
            for place, groups in available.items():
                data = groups[group]
                for component, ax in enumerate(axes[row]):
                    draw_band(
                        ax, theta, data["mean"][component], data["std"][component],
                        (
                            f"{place} (n={data['count']})\n"
                            f"Mean walking distance: "
                            f"{data['distance_min']:.1f}–{data['distance_max']:.1f} m"
                        ),
                        colors[place],
                    )
            for component, ax in enumerate(axes[row]):
                component_name = ["Real (cosine)", "Imaginary (sine)"][component]
                ax.set_title(f"{group.title()} distance — {component_name}")
                ax.set_xlabel("Evaluation point θ")
                ax.set_ylabel("Mean across selected nodes")
                ax.axhline(0, color="grey", linewidth=0.6)
                ax.grid(alpha=0.2)
            axes[row, 0].legend(fontsize=8)
        fig.suptitle(
            f"{category.replace('_', ' ')} — city comparison, r = {order}\n"
            f"Highest and lowest {percent:g}% of finite distances"
        )
        fig.text(
            0.5, 0.01, "Shading: mean ± 1 standard deviation across selected nodes",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        destination = folder / f"feather_{category}_high_low.png"
        fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved high/low comparison: {destination}", flush=True)


def process_city(city, config):
    # All large city objects stay here and are released after this call.
    place = city["place"]
    categories = config["resolved_categories"]
    order = config["order"]
    eval_points = config["eval_points"]
    theta = np.linspace(0.01, config["theta_max"], eval_points)
    place_name = safe_filename(place)
    print(f"\nProcessing {place}...", flush=True)
    buffered = build_graph_from_pbf(
        place, city["city_pbf_path"], city["city_dir"]
    )
    poi_data = extract_pois_from_city_pbf(
        place, city["city_pbf_path"], city["city_dir"],
        city["clip_boundary_path"], categories,
    )
    municipality = municipality_data(buffered, city["exact_boundary_path"])
    print("Calculating walking distances...", flush=True)
    distances, poi_counts = calculate_poi_distances(
        buffered, poi_data["features"], categories
    )
    distances = distances.reindex(municipality["nodes"].index)
    poi_counts = poi_counts.reindex(municipality["nodes"].index)

    # Keep mean distance beside the number of POIs used.
    distance_summary = pd.DataFrame(index=distances.index)
    distance_summary.index.name = "node_id"
    for category in categories:
        distance_summary[f"{category}_mean_distance_m"] = distances[category]
        distance_summary[f"{category}_poi_count_up_to_{NEAREST_POIS}"] = poi_counts[category]
    summary_folder = OUTPUT_ROOT / "POI_distances" / place_name
    summary_folder.mkdir(parents=True, exist_ok=True)
    summary_path = summary_folder / f"{place_name}_poi_distances.csv"
    distance_summary.to_csv(summary_path, index=True)
    print(f"Saved distances and reachable POI counts: {summary_path}", flush=True)
    del poi_counts, distance_summary
    del buffered, poi_data
    gc.collect()
    print("Saving distance maps at 600 DPI...", flush=True)
    map_folder = OUTPUT_ROOT / "graph_POis" / place_name
    for category in categories:
        save_distance_map(
            place, category, municipality, distances[category], map_folder
        )
    print(f"Saved {len(categories)} distance maps: {map_folder}", flush=True)
    save_interactive_data(
        place, municipality, distances, city["exact_boundary_path"]
    )
    print(f"Calculating FEATHER embeddings through r = {order}...", flush=True)
    inputs = prepare_feather_inputs(municipality["graph"], distances, categories)
    del municipality
    model = FEATHER_new(
        theta_max=config["theta_max"], eval_points=eval_points, order=order
    )
    model.fit(graph=inputs["graph"], X=inputs["features"])
    embedding = model.get_embedding()
    table = pd.DataFrame(
        embedding,
        index=pd.Index(inputs["node_ids"], name="node_id"),
        columns=embedding_columns(categories, order, eval_points),
    )
    folder = OUTPUT_ROOT / "FEATHER_embeddings" / place_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{place_name}_feather_embeddings.csv"
    table.to_csv(path, index=True)
    print(f"Saved embeddings: {path}", flush=True)
    values = embedding.reshape(
        len(embedding), order, 2, len(categories), eval_points
    )
    means = values.mean(axis=0)
    high_low = high_low_summary(
        values, distances, inputs["node_ids"], categories, config["high_low_percent"]
    )
    del distances
    curve_folder = OUTPUT_ROOT / "FEATHER_curves" / "FEATHER_curves_mean" / place_name
    print("Saving one FEATHER figure per amenity...", flush=True)
    for index, category in enumerate(categories):
        series = {
            f"r = {r + 1}": means[r, :, index, :]
            for r in range(order)
        }
        save_curves(
            series, theta, f"{place} — {category.replace('_', ' ')}",
            curve_folder / f"feather_{category}.png",
        )
    print(f"Saved {len(categories)} FEATHER figures: {curve_folder}", flush=True)
    high_low_folder = (
        OUTPUT_ROOT / "FEATHER_curves" / "FEATHER_curves_high_low" / place_name
    )
    print("Saving high/low distance curves with standard-deviation bands...", flush=True)
    save_high_low_city(place, high_low, theta, config["high_low_percent"], high_low_folder)
    print(f"Saved {len(high_low)} high/low figures: {high_low_folder}", flush=True)
    print(f"Finished {place}.", flush=True)
    print("\n" + "*" * 60 + "\n", flush=True)
    # Only the largest order is needed for the cross-city figures.
    last_groups = {
        category: {
            group: {
                "mean": data["mean"][-1].copy(),
                "std": data["std"][-1].copy(),
                "count": data["count"],
                "distance_min": data["distance_min"],
                "distance_max": data["distance_max"],
            }
            for group, data in groups.items()
        }
        for category, groups in high_low.items()
    }
    return means[-1].copy(), last_groups

def run_pipeline():
    global CLIP_BUFFER_M, NEAREST_POIS
    config = load_config(CONFIG_PATH)
    if not 0 < config["high_low_percent"] <= 50:
        raise ValueError("high_low_percent must be greater than 0 and at most 50.")
    CLIP_BUFFER_M = config["CLIP_BUFFER_M"]
    NEAREST_POIS = config["NEAREST_POIS"]
    if min(CLIP_BUFFER_M, NEAREST_POIS, config["order"], config["eval_points"]) <= 0:
        raise ValueError("Buffer, POI count, order and evaluation count must be positive.")
    if not config["places"] or not config["resolved_categories"]:
        raise ValueError("Configure at least one place and amenity category.")
    print("\nFEATHER pipeline", flush=True)
    print(f"Places: {', '.join(config['places'])}", flush=True)
    print(
        f"Order: {config['order']} | Evaluation points: {config['eval_points']} | "
        f"Maximum theta: {config['theta_max']}", flush=True,
    )
    print(
        f"Buffer/search limit: {CLIP_BUFFER_M:,} m | Nearest POIs: {NEAREST_POIS}",
        flush=True,
    )
    print("\n" + "*" * 60 + "\n", flush=True)
    print(f"High/low groups: {config['high_low_percent']:g}% each; missing distances excluded.", flush=True)
    move_existing_mean_curves()
    print("Preparing city files; existing downloads will be reused...", flush=True)
    index = get_geofabrik_index()
    plan = build_download_plan(config["places"], index)
    # Clip all cities in each region before processing them individually.
    prepare_city_pbfs(plan)
    del index
    print(f"City files ready in: {CACHE_ROOT}", flush=True)
    city_means = {}
    city_groups = {}
    for city in plan.to_dict(orient="records"):
        city_means[city["place"]], city_groups[city["place"]] = process_city(city, config)
        gc.collect()
    order = config["order"]
    theta = np.linspace(0.01, config["theta_max"], config["eval_points"])
    folder = OUTPUT_ROOT / "FEATHER_curves" / "FEATHER_curves_mean"
    print(f"Saving cross-city comparisons for r = {order}...", flush=True)
    for index, category in enumerate(config["resolved_categories"]):
        series = {
            place: means[:, index, :]
            for place, means in city_means.items()
        }
        save_curves(
            series, theta,
            f"{category.replace('_', ' ')} — city comparison, r = {order}",
            folder / f"feather_{category}.png",
        )
    print(f"Saved cross-city figures: {folder}", flush=True)
    high_low_folder = OUTPUT_ROOT / "FEATHER_curves" / "FEATHER_curves_high_low"
    print(f"Saving high/low cross-city comparisons for r = {order}...", flush=True)
    save_high_low_comparisons(
        city_groups, config["resolved_categories"], theta, order,
        config["high_low_percent"], high_low_folder,
    )
    print(f"High/low comparisons saved in: {high_low_folder}", flush=True)
    print("Pipeline finished.", flush=True)