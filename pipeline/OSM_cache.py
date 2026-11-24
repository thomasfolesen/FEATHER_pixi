"""Download/cache OSM data and build reusable walking graph caches."""

import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from pyrosm import OSM
from shapely.geometry import shape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline_common import load_config, safe_filename


CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
CACHE_ROOT = PROJECT_ROOT / "cache"
GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
GEOFABRIK_WAIT_SECONDS = 60
NETWORK_TYPE = "walking"
TARGET_CONTINENTS = {"north-america", "europe", "asia"}

print("Starting pipeline/OSM_cache.py")


def get_geofabrik_index():
    """Load the cached Geofabrik index, or download it once."""
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
    """Load or create the municipality and buffered clip boundaries."""
    city_dir = CACHE_ROOT / safe_filename(place)
    city_dir.mkdir(parents=True, exist_ok=True)

    exact_path = city_dir / "municipality_boundary.geojson"
    clip_path = city_dir / f"municipality_clip_boundary_{CLIP_BUFFER_M}m.geojson"

    if exact_path.exists():
        exact = gpd.read_file(exact_path).to_crs("EPSG:4326")
    else:
        print(f"Downloading municipality boundary for {place}...", flush=True)
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
        print(
            f"Creating {CLIP_BUFFER_M:,} m buffered boundary for {place}...",
            flush=True,
        )
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
    """Choose the smallest permitted Geofabrik region covering the city."""
    properties = {
        feature["properties"]["id"]: feature["properties"]
        for feature in index["features"]
    }

    def under_continent(region_id, targets):
        while region_id:
            if region_id in targets:
                return True
            region = properties.get(region_id)
            if region is None:
                return False
            region_id = region.get("parent")
        return False

    city_geometry = clip_boundary.geometry.union_all()
    city_point = city_geometry.representative_point()
    candidates = []

    for feature in index["features"]:
        prop = feature["properties"]
        region_id = prop["id"]
        pbf_url = prop.get("urls", {}).get("pbf")

        if not pbf_url or not under_continent(region_id, TARGET_CONTINENTS):
            continue

        region = shape(feature["geometry"])
        if region.covers(city_point):
            candidates.append(
                {
                    "source_id": region_id,
                    "source_name": prop["name"],
                    "source_url": pbf_url,
                    "covers_full_city": region.covers(city_geometry),
                    "area": region.area,
                }
            )

    if not candidates:
        raise ValueError(
            "No Geofabrik source was found for the buffered city boundary. "
            f"Allowed continents: {sorted(TARGET_CONTINENTS)}."
        )

    full_city = [
        candidate
        for candidate in candidates
        if candidate["covers_full_city"]
    ]
    return min(full_city or candidates, key=lambda candidate: candidate["area"])


def build_download_plan(places, index):
    """Create paths and source information for every configured place."""
    rows = []

    for place in places:
        boundary = city_boundaries(place)
        source = select_source(index, boundary["clip_boundary"])
        city_pbf_path = (
            boundary["city_dir"]
            / f"{safe_filename(place)}_buffer_{CLIP_BUFFER_M}m.osm.pbf"
        )
        graph_cache_path = (
            boundary["city_dir"]
            / (
                f"city_graph_{NETWORK_TYPE}_buffer_{CLIP_BUFFER_M}m_"
                "simplified_undirected.pkl"
            )
        )

        rows.append(
            {
                "place": place,
                "city_dir": boundary["city_dir"],
                "exact_boundary_path": boundary["exact_boundary_path"],
                "clip_boundary_path": boundary["clip_boundary_path"],
                **source,
                "city_pbf_path": city_pbf_path,
                "graph_cache_path": graph_cache_path,
            }
        )

    return pd.DataFrame(rows)


def download_pbf(url, destination):
    """Download a regional PBF unless it is already cached."""
    if destination.exists():
        print(f"Using cached file: {destination}", flush=True)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Waiting {GEOFABRIK_WAIT_SECONDS} seconds before downloading "
        f"{destination.name}...",
        flush=True,
    )
    time.sleep(GEOFABRIK_WAIT_SECONDS)

    with requests.get(url, stream=True, timeout=900) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8_000_000):
                if chunk:
                    file.write(chunk)

    print(f"Downloaded: {destination}", flush=True)


def clip_pbf(regional_pbf, polygon_path, output_pbf):
    """Clip a regional PBF with osmium using the buffered boundary."""
    output_pbf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "osmium",
            "extract",
            "--polygon",
            str(polygon_path),
            "--strategy=complete_ways",
            "--set-bounds",
            "--output",
            str(output_pbf),
            "--overwrite",
            str(regional_pbf),
        ],
        check=True,
    )


def prepare_city_pbfs(download_plan):
    """Create any missing clipped city PBF files."""
    plan = download_plan.copy()
    regional_dir = CACHE_ROOT / "geofabrik" / "regional_temp"
    plan["pbf_status"] = "cached"

    for row_index, city in plan.iterrows():
        city_pbf = city["city_pbf_path"]
        if city_pbf.exists():
            print(f"Using cached city PBF for {city['place']}.", flush=True)
            continue

        regional_pbf = regional_dir / f"{safe_filename(city['source_id'])}.osm.pbf"
        download_pbf(city["source_url"], regional_pbf)
        print(f"Clipping PBF for {city['place']}...", flush=True)
        clip_pbf(regional_pbf, city["clip_boundary_path"], city_pbf)
        plan.loc[row_index, "pbf_status"] = "created"

    plan["city_pbf_mb"] = plan["city_pbf_path"].map(
        lambda path: path.stat().st_size / 1024**2
    )
    return plan


def build_graph_cache(place, city_pbf_path, graph_cache_path):
    """Build and cache a simplified, undirected walking graph."""
    if graph_cache_path.exists():
        print(f"Using cached walking graph for {place}.", flush=True)
        return "cached"

    started = time.time()
    print(f"Building walking graph for {place}...", flush=True)

    osm = OSM(
        str(city_pbf_path),
        engine="out_of_core",
        workers="auto",
    )
    network_nodes, network_edges = osm.get_network(
        network_type=NETWORK_TYPE,
        nodes=True,
    )

    if (
        network_nodes is None
        or network_edges is None
        or network_nodes.empty
        or network_edges.empty
    ):
        raise ValueError(f"No {NETWORK_TYPE} network was found for {place}.")

    graph = osm.to_graph(
        network_nodes,
        network_edges,
        graph_type="networkx",
        network_type=NETWORK_TYPE,
        simplify=False,
    )
    graph.graph["crs"] = "EPSG:4326"

    graph = ox.simplification.simplify_graph(graph)
    graph = ox.convert.to_undirected(graph)

    if graph.is_multigraph():
        self_loops = list(nx.selfloop_edges(graph, keys=True))
    else:
        self_loops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(self_loops)

    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise ValueError(f"The walking graph for {place} is empty after cleaning.")

    largest = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest).copy()

    graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with graph_cache_path.open("wb") as file:
        pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)

    runtime = time.time() - started
    print(
        f"Saved walking graph for {place}: {graph_cache_path} "
        f"({graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges, {runtime:.1f} seconds).",
        flush=True,
    )
    return "built"


def run_pipeline():
    global CLIP_BUFFER_M

    config = load_config(CONFIG_PATH)
    CLIP_BUFFER_M = config["CLIP_BUFFER_M"]

    if CLIP_BUFFER_M <= 0:
        raise ValueError("CLIP_BUFFER_M must be positive.")
    if not config["places"]:
        raise ValueError("Configure at least one place in input/config.txt.")

    print("\nOSM cache pipeline", flush=True)
    print(
        "Preparing boundaries, clipped PBF files and walking graph caches...",
        flush=True,
    )

    index = get_geofabrik_index()
    plan = build_download_plan(config["places"], index)
    plan = prepare_city_pbfs(plan)
    del index

    for row_index, city in plan.iterrows():
        status = build_graph_cache(
            city["place"],
            city["city_pbf_path"],
            city["graph_cache_path"],
        )
        plan.loc[row_index, "graph_status"] = status

    print("\nOSM cache pipeline finished.", flush=True)
    for _, city in plan.iterrows():
        print(
            f"- {city['place']}: PBF={city['pbf_status']}, "
            f"graph={city['graph_status']}, "
            f"city PBF={city['city_pbf_mb']:.1f} MB",
            flush=True,
        )
    print(
        "Next run: pixi run python pipeline/FEATHER_embeddings.py",
        flush=True,
    )


if __name__ == "__main__":
    run_pipeline()
