import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """

    Relative paths start from the directory where you launch Marimo, regardless of where the Python file is located.
    pixi run marimo edit pipeline/geofabrik_final.py
    """


    #importing lots of thing just to be sure
    import marimo as mo
    import hashlib
    import json
    import pickle
    import re
    import shutil
    import subprocess
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path
    from pyrosm import OSM
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import osmnx as ox
    import pandas as pd
    import requests
    from shapely.geometry import shape
    import pandana as pdna


    """
    current file: pipeline/geofabrik_final.py
    feather placement:  FEATHER/feather.py
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from FEATHER.feather import FEATHER_new


    return (
        FEATHER_new,
        OSM,
        Path,
        gpd,
        json,
        mo,
        np,
        nx,
        ox,
        pd,
        pdna,
        pickle,
        plt,
        re,
        requests,
        shape,
        subprocess,
        time,
    )


@app.cell
def _(Path):
    CONFIG_PATH = Path(__file__).resolve().parent.parent / "input" / "config.txt"
    CACHE_ROOT = Path("cache")
    place_placeHolder = " "
    FEATHER_OUTPUT_DIR = Path("../output") / place_placeHolder / "FEATHER_embeddings_raw_meters"

    # OSM / Geofabrik settings
    GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
    NETWORK_TYPE = "walking"

    # Cache controls
    FORCE_REFRESH_CITY_PBF = False #reuse cached .pbf
    FORCE_REBUILD_DERIVED = False  #reuse files for city data
    DELETE_REGIONAL_PBF = True #delete large regional .pbf after creating smaller city file
    return (
        CACHE_ROOT,
        CONFIG_PATH,
        DELETE_REGIONAL_PBF,
        FORCE_REBUILD_DERIVED,
        FORCE_REFRESH_CITY_PBF,
        GEOFABRIK_INDEX_URL,
        NETWORK_TYPE,
    )


@app.cell
def _(re):
    """
    function to read config file with the following structure
    place =... list of lists of strings [[string]]
    theta_max = ... float
    eval_points = ... int
    order = ... int
    outdoor_activities = ... list of lists of strings [[string]]
    learning = ... list of lists of strings [[string]]
    supplies = ... list of lists of strings [[string]]
    eating = ... list of lists of strings [[string]]
    moving = ... list of lists of strings [[string]]
    cultural_activities = ... list of lists of strings [[string]]
    physical_exercise = ... list of lists of strings [[string]]
    services = ... list of lists of strings [[string]]
    healthcare = ... list of lists of strings [[string]]
    financial = ... list of lists of strings [[string]]

    """
    def parse_config_list(value):
        """Read items inside [[...], [...]], preserving commas within each item. using regex"""
        return [
            item.strip()
            for item in re.findall(r"\[([^\[\]]+)\]", value)
        ]


    def load_config(path):
        config = {"resolved_categories": {}}

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if key == "place":
                config["places"] = parse_config_list(value)

            elif key == "theta_max":
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

    return (load_config,)


@app.cell
def _(CONFIG_PATH, load_config, pd):
    config = load_config(CONFIG_PATH)

    places = config["places"]
    theta_max = config["theta_max"]
    eval_points = config["eval_points"]
    order = config["order"]
    CLIP_BUFFER_M = config["CLIP_BUFFER_M"]
    NEAREST_POIS = config["NEAREST_POIS"]
    resolved_categories = config["resolved_categories"]

    if CLIP_BUFFER_M <= 0:
        raise ValueError("CLIP_BUFFER_M must be greater than zero.")

    config_settings = pd.DataFrame({
        "place": places,
        "theta_max": theta_max,
        "eval_points": eval_points,
        "order": order,
        "CLIP_BUFFER_M": CLIP_BUFFER_M,
    })

    config_settings
    return (
        CLIP_BUFFER_M,
        NEAREST_POIS,
        eval_points,
        order,
        places,
        resolved_categories,
        theta_max,
    )


@app.cell
def _(re):
    def safe_filename(place):
        return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()

    return (safe_filename,)


@app.cell
def _(CACHE_ROOT, GEOFABRIK_INDEX_URL, json, requests):
    #function to support downloading graph and POIs for place
    def get_geofabrik_index():
        """
        Get geofabrik index
        put into: cache/geofabrik/index-v1.json
        """
        index_path = CACHE_ROOT / "geofabrik" / "index-v1.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if not index_path.exists():
            response = requests.get(GEOFABRIK_INDEX_URL, timeout=60)
            response.raise_for_status()
            index_path.write_bytes(response.content)#write to disk

        return json.loads(index_path.read_text(encoding="utf-8"))

    return (get_geofabrik_index,)


@app.cell
def _(CACHE_ROOT, CLIP_BUFFER_M, gpd, ox, safe_filename):
    #function to support downloading graph and POIs for place
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


    return (city_boundaries,)


@app.cell
def _(shape):
    #function to support downloading graph and POIs for place
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


    return (select_source,)


@app.cell
def _(CLIP_BUFFER_M, city_boundaries, pd, safe_filename, select_source):
    #function to support downloading graph and POIs for place
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

    return (build_download_plan,)


@app.cell
def _(requests):
    #function to support downloading graph and POIs for place
    def download_pbf(url, destination):
        if destination.exists():
            return

        destination.parent.mkdir(parents=True, exist_ok=True)

        with requests.get(url, stream=True, timeout=900) as response:
            response.raise_for_status()
            with destination.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    file.write(chunk)

    return (download_pbf,)


@app.cell
def _(subprocess):
    #function to support downloading graph and POIs for place
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

    return (clip_pbf,)


@app.cell
def _(
    CACHE_ROOT,
    DELETE_REGIONAL_PBF,
    FORCE_REFRESH_CITY_PBF,
    clip_pbf,
    download_pbf,
    safe_filename,
):
    #function to support downloading graph and POIs for place
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

                if city_pbf.exists() and not FORCE_REFRESH_CITY_PBF:
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

    return (prepare_city_pbfs,)


@app.cell
def _(
    CLIP_BUFFER_M,
    FORCE_REBUILD_DERIVED,
    NETWORK_TYPE,
    OSM,
    nx,
    ox,
    pickle,
    time,
):
    def build_graph_from_pbf(place, city_pbf_path, city_dir):
        """Build or load a simplified, undirected walking graph."""
        started = time.time()
        cache_path = (
            city_dir
            / f"city_graph_{NETWORK_TYPE}_buffer_{CLIP_BUFFER_M}m_simplified_undirected.pkl"
        )

        if cache_path.exists() and not FORCE_REBUILD_DERIVED:
            print(f"Loading cached graph for {place}")
            with cache_path.open("rb") as file:
                graph = pickle.load(file)
            status = "cached"

        else:
            print(f"Building graph for {place}")
            osm = OSM(str(city_pbf_path), engine="out_of_core")

            network_nodes, network_edges = osm.get_network(
                network_type=NETWORK_TYPE,
                nodes=True,
            )

            graph = osm.to_graph(
                network_nodes,
                network_edges,
                graph_type="networkx",
                network_type=NETWORK_TYPE,
                simplify=False,
            )
            graph.graph["crs"] = "EPSG:4326"

            # Simplify road segments and allow travel in both directions.
            graph = ox.simplification.simplify_graph(graph)
            graph = ox.convert.to_undirected(graph)

            # Remove edges connecting a node to itself.
            graph.remove_edges_from(list(nx.selfloop_edges(graph, keys=True)))

            # Keep the largest connected group of nodes.
            largest = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest).copy()

            with cache_path.open("wb") as file:
                pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)
            status = "built"

        # Convert coordinates to a local projection measured in metres.
        projected_graph = ox.project_graph(graph)
        nodes, edges = ox.graph_to_gdfs(projected_graph)

        return {
            "graph": graph,
            "projected_graph": projected_graph,
            "nodes": nodes,
            "edges": edges,
            "graph_cache_path": cache_path,
            "graph_status": status,
            "graph_runtime_seconds": time.time() - started,
        }

    return (build_graph_from_pbf,)


@app.cell
def _(CLIP_BUFFER_M, FORCE_REBUILD_DERIVED, OSM, gpd, pd):
    def extract_pois_from_city_pbf(
        place, city_pbf_path, city_dir, exact_boundary_path, resolved_categories
    ):
        """Extract configured POIs inside the city boundary, or load cached POIs."""
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
            osm = OSM(str(city_pbf_path), engine="out_of_core")
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

                # Keep POIs whose representative point is inside the municipality.
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

    return (extract_pois_from_city_pbf,)


@app.cell
def _(build_graph_from_pbf, extract_pois_from_city_pbf, pd):
    def build_city_datasets(download_plan, resolved_categories):
        """Build or load each city's graph and POIs, and create a summary table."""
        graphs = {}
        poi_results = {}
        summary = []

        for city in download_plan.to_dict(orient="records"):
            place = city["place"]
            city_pbf = city["city_pbf_path"]
            city_dir = city["city_dir"]

            print(f"Building graph and POIs: {place}")

            graph_data = build_graph_from_pbf(place, city_pbf, city_dir)
            poi_data = extract_pois_from_city_pbf(
                place,
                city_pbf,
                city_dir,
                city["clip_boundary_path"],
                resolved_categories,
            )

            graphs[place] = graph_data
            poi_results[place] = poi_data

            graph = graph_data["graph"]
            pois = poi_data["features"]

            summary.append({
                "place": place,
                "geofabrik_source": city["source_name"],
                "city_pbf_mb": city_pbf.stat().st_size / 1024**2, #convert to MiB
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "graph_status": graph_data["graph_status"],
                "graph_runtime_seconds": round(graph_data["graph_runtime_seconds"], 2),
                "poi_records": len(pois),
                "unique_osm_pois": len(
                    pois.drop_duplicates(subset=["osm_type", "osm_id"])
                ),
                "poi_status": poi_data["poi_status"],
            })

        return graphs, poi_results, pd.DataFrame(summary)

    return (build_city_datasets,)


@app.cell
def _(build_download_plan, get_geofabrik_index, places):
    geofabrik_index = get_geofabrik_index()
    download_plan_df = build_download_plan(places, geofabrik_index)
    download_plan_df
    return (download_plan_df,)


@app.cell
def _(
    build_city_datasets,
    download_plan_df,
    prepare_city_pbfs,
    resolved_categories,
):
    pbf_results_df = prepare_city_pbfs(download_plan_df)
    graphs, poi_results, pipeline_metadata_df = build_city_datasets(
        download_plan_df, resolved_categories
    )
    return graphs, poi_results


@app.cell
def _(download_plan_df, gpd, graphs):
    buffered_graphs = graphs
    municipality_graphs = {}

    for _city in download_plan_df.to_dict(orient="records"):
        _place = _city["place"]
        _data = buffered_graphs[_place]

        _boundary = (
            gpd.read_file(_city["exact_boundary_path"])
            .to_crs(_data["nodes"].crs)
            .geometry.union_all()
        )

        _inside = _data["nodes"].geometry.apply(_boundary.covers)
        _node_ids = _data["nodes"].index[_inside]

        _graph = _data["graph"].subgraph(_node_ids).copy()
        _projected_graph = (
            _data["projected_graph"].subgraph(_node_ids).copy()
        )

        _edges = _data["edges"]
        _keep_edges = (
            _edges.index.get_level_values("u").isin(_node_ids)
            & _edges.index.get_level_values("v").isin(_node_ids)
        )

        municipality_graphs[_place] = {
            **_data,
            "graph": _graph,
            "projected_graph": _projected_graph,
            "nodes": _data["nodes"].loc[_node_ids].copy(),
            "edges": _edges.loc[_keep_edges].copy(),
        }
    return buffered_graphs, municipality_graphs


@app.cell
def _(
    CONFIG_PATH,
    graphs,
    mo,
    places,
    plt,
    poi_results,
    resolved_categories,
    safe_filename,
):
    _maps = []
    _colors = dict(zip(resolved_categories, plt.get_cmap("tab10").colors))
    _output = CONFIG_PATH.parent.parent / "output" / "graph_POis"

    for _place in places:
        _edges = graphs[_place]["edges"]
        _pois = poi_results[_place]["features"].to_crs(_edges.crs)

        _fig, _ax = plt.subplots(
            figsize=(14, 11), dpi=150, facecolor="white"
        )
        _ax.set_facecolor("white")

        _edges.plot(ax=_ax, color="#777777", linewidth=0.5, zorder=1)

        for _category, _points in _pois.groupby("category"):
            _points.plot(
                ax=_ax,
                color=_colors[_category],
                markersize=25,
                edgecolor="black",
                linewidth=0.3,
                label=f"{_category.replace('_', ' ')} ({len(_points)})",
                zorder=2,
            )

        _ax.set_title(_place, color="black")
        _ax.set_axis_off()

        if not _pois.empty:
            _ax.legend(
                loc="upper left",
                bbox_to_anchor=(1, 1),
                facecolor="white",
                edgecolor="#cccccc",
                labelcolor="black",
                framealpha=1,
            )

        _folder = _output / safe_filename(_place)
        _folder.mkdir(parents=True, exist_ok=True)

        _fig.savefig(
            _folder / "graph_pois.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )

        _maps.append(mo.as_html(_fig))
        plt.close(_fig)

    mo.vstack(_maps)
    return


@app.cell
def _(CLIP_BUFFER_M, NEAREST_POIS, np, pd, pdna):
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

        return distances

    return (calculate_poi_distances,)


@app.cell
def _(CACHE_ROOT, CLIP_BUFFER_M, NEAREST_POIS, gpd, np, plt, safe_filename):
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
                    markersize=8,
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
                    markersize=8,
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
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                transparent=False,
            )
            plt.close(fig)

        return image_path

    return (save_distance_map,)


@app.cell
def _(
    CONFIG_PATH,
    buffered_graphs,
    calculate_poi_distances,
    municipality_graphs,
    places,
    poi_results,
    resolved_categories,
    safe_filename,
    save_distance_map,
):
    distance_tables = {}
    _output = CONFIG_PATH.parent.parent / "output" / "graph_POis"

    for _place in places:
        print(f"Calculating walking distances for {_place}...", flush=True)

        _buffered_distances = calculate_poi_distances(
            buffered_graphs[_place],
            poi_results[_place]["features"],
            resolved_categories,
        )
    
        distance_tables[_place] = _buffered_distances.reindex(
            municipality_graphs[_place]["nodes"].index
        )

        _folder = _output / safe_filename(_place)
        _folder.mkdir(parents=True, exist_ok=True)

        for _category in resolved_categories:
            _image_path = save_distance_map(
                _place,
                _category,
                municipality_graphs[_place],
                distance_tables[_place][_category],
                _folder,
            )

            print(f"Saved {_image_path.name}", flush=True)

        print(f"Finished: {_folder}", flush=True)
    return (distance_tables,)


@app.cell
def _(CLIP_BUFFER_M, np, nx):
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

        # Treat missing amenities and distances beyond 2 km equally.
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

    return (prepare_feather_inputs,)


@app.cell
def _(
    distance_tables,
    municipality_graphs,
    pd,
    places,
    prepare_feather_inputs,
    resolved_categories,
):
    feather_inputs = {}

    for _place in places:
        feather_inputs[_place] = prepare_feather_inputs(
            graph=municipality_graphs[_place]["graph"],
            distances=distance_tables[_place],
            categories=resolved_categories,
        )

    pd.DataFrame([
        {
            "place": _place,
            "nodes": _data["features"].shape[0],
            "categories": _data["features"].shape[1],
            "feature_unit": "kilometres",
        }
        for _place, _data in feather_inputs.items()
    ])
    return (feather_inputs,)


@app.cell
def _(FEATHER_new, eval_points, feather_inputs, order, pd, theta_max):
    feather_embeddings = {}

    for _place, _data in feather_inputs.items():
        _model = FEATHER_new(
            theta_max=theta_max,
            eval_points=eval_points,
            order=order,
        )

        _model.fit(
            graph=_data["graph"],
            X=_data["features"],
        )

        feather_embeddings[_place] = pd.DataFrame(
            _model.get_embedding(),
            index=pd.Index(_data["node_ids"], name="node_id"),
        )

    pd.DataFrame([
        {
            "place": _place,
            "nodes": len(_embedding),
            "embedding_columns": _embedding.shape[1],
        }
        for _place, _embedding in feather_embeddings.items()
    ])
    return (feather_embeddings,)


@app.cell
def _(
    CONFIG_PATH,
    eval_points,
    feather_embeddings,
    feather_inputs,
    mo,
    np,
    order,
    plt,
    theta_max,
):
    _feather_maps = []
    _output = CONFIG_PATH.parent.parent / "output" / "FEATHER_curves"
    _output.mkdir(parents=True, exist_ok=True)

    # Match the evaluation points used inside FEATHER_new.
    _theta = np.linspace(0.01, theta_max, eval_points)
    _categories = next(iter(feather_inputs.values()))["categories"]

    _city_curves = {}

    for _place, _embedding in feather_embeddings.items():
        if feather_inputs[_place]["categories"] != _categories:
            raise ValueError("Cities must use the same categories in the same order.")

        # Separate the embedding columns into their original groups.
        _values = _embedding.to_numpy().reshape(
            len(_embedding),
            order,
            2,
            len(_categories),
            eval_points,
        )

        # Average over nodes to obtain city-level curves.
        _city_curves[_place] = _values.mean(axis=0)

    for _category_index, _category in enumerate(_categories):
        with plt.style.context("default"):
            _fig, _axes = plt.subplots(
                order,
                2,
                figsize=(12, 4 * order),
                squeeze=False,
                sharex=True,
                sharey=True,
            )

            for _step in range(order):
                for _component, _label in enumerate(["Real (cosine)", "Imaginary (sine)"]):
                    _ax = _axes[_step, _component]

                    for _city_index, (_place, _curves) in enumerate(_city_curves.items()):
                        _ax.plot(
                            _theta,
                            _curves[_step, _component, _category_index],
                            color=plt.get_cmap("tab10")(_city_index % 10),
                            marker="o",
                            label=_place,
                        )

                    _ax.set_title(f"Order {_step + 1} — {_label}")
                    _ax.set_xlabel("Evaluation point θ")
                    _ax.set_ylabel("Mean across nodes")
                    _ax.set_ylim(-1.05, 1.05)
                    _ax.axhline(0, color="grey", linewidth=0.6)
                    _ax.grid(alpha=0.2)

            _axes[0, 0].legend(fontsize=8)
            _fig.suptitle(
                f"FEATHER city comparison: {_category.replace('_', ' ')}",
                fontsize=15,
            )
            _fig.tight_layout()

            _fig.savefig(
                _output / f"feather_{_category}.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )

            _feather_maps.append(mo.as_html(_fig))
            plt.close(_fig)

    mo.vstack(_feather_maps)
    return


@app.cell
def _(CONFIG_PATH, feather_embeddings, safe_filename):
    _embedding_folder = (
        CONFIG_PATH.parent.parent / "output" / "FEATHER_embeddings"
    )

    for _place, _embedding in feather_embeddings.items():
        _place_name = safe_filename(_place)
        _place_folder = _embedding_folder / _place_name
        _place_folder.mkdir(parents=True, exist_ok=True)

        _file = _place_folder / f"{_place_name}_feather_embeddings.csv"
        _embedding.to_csv(_file, index=True)

        print(f"{_place} saved in: {_file.resolve()}")
    return


@app.cell
def _(
    CONFIG_PATH,
    eval_points,
    feather_embeddings,
    feather_inputs,
    mo,
    np,
    order,
    plt,
    safe_filename,
    theta_max,
):
    _order_figures = []
    _order_output = CONFIG_PATH.parent.parent / "output" / "FEATHER_curves"

    # Match the evaluation points used inside FEATHER_new.
    _theta = np.linspace(0.01, theta_max, eval_points)

    for _place, _embedding in feather_embeddings.items():
        _categories = feather_inputs[_place]["categories"]

        _values = _embedding.to_numpy().reshape(
            len(_embedding),
            order,
            2,
            len(_categories),
            eval_points,
        )

        _means = _values.mean(axis=0)

        _fig, _axes = plt.subplots(
            len(_categories),
            2,
            figsize=(16, 4 * len(_categories)),
            squeeze=False,
        )

        for _category_index, _category in enumerate(_categories):
            _category_label = _category.replace("_", " ").title()

            for _component, _component_name in enumerate(
                ["Real (cosine)", "Imaginary (sine)"]
            ):
                _ax = _axes[_category_index, _component]

                for _step in range(order):
                    _ax.plot(
                        _theta,
                        _means[_step, _component, _category_index],
                        color=plt.get_cmap("tab10")(_step % 10),
                        linestyle=["-", "--", "-.", ":"][_step % 4],
                        marker=["o", "s", "^", "D", "x"][_step % 5],
                        markerfacecolor="none",
                        markersize=5,
                        linewidth=1.5,
                        label=f"r = {_step + 1}",
                    )

                _ax.set_title(
                    f"{_category_label} — {_component_name}",
                    fontsize=13,
                )
                _ax.set_xlabel("Evaluation point θ")
                _ax.set_ylabel("Mean across nodes")
                _ax.set_ylim(-1.05, 1.05)
                _ax.axhline(0, color="grey", linewidth=0.6)
                _ax.grid(alpha=0.25)
                _ax.legend(title="Order", fontsize=8)

        _fig.suptitle(
            f"{_place}\nFEATHER means by amenity category and order",
            fontsize=16,
        )
        _fig.tight_layout(rect=(0, 0, 1, 0.97))

        _folder = _order_output / safe_filename(_place)
        _folder.mkdir(parents=True, exist_ok=True)

        _file = _folder / "feather_all_orders.png"

        _fig.savefig(
            _file,
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )

        _order_figures.append(mo.as_html(_fig))
        plt.close(_fig)

        print(f"Saved: {_file.resolve()}")

    mo.vstack(_order_figures)
    return


if __name__ == "__main__":
    app.run()
