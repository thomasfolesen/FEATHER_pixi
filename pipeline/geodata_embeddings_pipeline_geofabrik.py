import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # UrbanFEATHER local-PBF pipeline

    This notebook has four stages:

    1. **Config** - read places, FEATHER parameters, and OSM amenity categories.
    2. **Local OSM data** - choose a Geofabrik extract, clip a city PBF, build the walking graph, and extract POIs.
    3. **FEATHER features** - calculate each node's network distance to the nearest POI in every category and pass those raw metre values directly into the sine/cosine transform.
    4. **Output** - run `FEATHER_new`, save one embedding CSV per city, and inspect the results on the street graph.

    The workbook runs automatically when opened. Expensive reusable OSM/graph results are cached under `../cache/`; raw-metre FEATHER CSV files are written to `../output/FEATHER_embeddings_raw_meters/`.
    """)
    return


@app.cell
def _():
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

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import osmnx as ox
    import pandas as pd
    import requests
    from shapely.geometry import shape

    try:
        from pyrosm import OSM
        pyrosm_import_error = None
    except Exception as _error:
        OSM = None
        pyrosm_import_error = str(_error)

    # Find the repository root by looking for FEATHER/feather.py.
    # This keeps the notebook independent of the directory it is launched from.
    _repo_root = Path(__file__).resolve().parent
    while (
        _repo_root != _repo_root.parent
        and not (_repo_root / "FEATHER" / "feather.py").exists()
    ):
        _repo_root = _repo_root.parent

    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    try:
        from FEATHER.feather import FEATHER_new
        feather_import_error = None
    except Exception as _error:
        FEATHER_new = None
        feather_import_error = str(_error)
    return (
        FEATHER_new,
        OSM,
        Path,
        datetime,
        feather_import_error,
        gpd,
        hashlib,
        json,
        np,
        nx,
        ox,
        pd,
        pickle,
        plt,
        pyrosm_import_error,
        re,
        requests,
        shape,
        shutil,
        subprocess,
        time,
        timezone,
    )


@app.cell
def _(feather_import_error, mo, pyrosm_import_error, shutil):
    _problems = []

    if pyrosm_import_error:
        _problems.append("`pyrosm` is not importable. Run `pixi add pyrosm`.")
    if feather_import_error:
        _problems.append(
            "Could not import `FEATHER_new` from `FEATHER/feather.py`: "
            + feather_import_error
        )
    if shutil.which("osmium") is None:
        _problems.append("The `osmium` command is missing. Run `pixi add osmium-tool`.")

    dependency_messages = _problems
    mo.callout(
        mo.md("\n\n".join(_problems) if _problems else "Dependencies are available."),
        kind="warn" if _problems else "success",
    )
    return


@app.cell
def _(Path):
    # Paths
    place_placeHolder = ""
    CONFIG_PATH = Path("../input/config.txt")
    CACHE_ROOT = Path("../cache")
    FEATHER_OUTPUT_DIR = Path("../output/FEATHER_embeddings_raw_meters")

    # OSM / Geofabrik settings
    GEOFABRIK_INDEX_URL = "https://download.geofabrik.de/index-v1.json"
    GEOFABRIK_INDEX_MAX_AGE_HOURS = 24
    CLIP_BUFFER_M = 100
    NETWORK_TYPE = "walking"

    # Cache controls
    FORCE_REFRESH_CITY_PBF = False #reuse cached .pbf
    FORCE_REBUILD_DERIVED = False  #reuse files for city data
    DELETE_REGIONAL_PBF = True #delete large regional .pbf after creating smaller city file

    # A bare config value such as [hospital] uses the category's default OSM key.
    DEFAULT_OSM_KEY = {
        "outdoor_activities": "leisure",
        "learning": "amenity",
        "supplies": "shop",
        "eating": "amenity",
        "moving": "public_transport",
        "cultural_activities": "amenity",
        "physical_exercise": "leisure",
        "services": "amenity",
        "healthcare": "amenity",
        "financial": "amenity",
    }
    return (
        CACHE_ROOT,
        CLIP_BUFFER_M,
        CONFIG_PATH,
        DEFAULT_OSM_KEY,
        DELETE_REGIONAL_PBF,
        FEATHER_OUTPUT_DIR,
        FORCE_REBUILD_DERIVED,
        FORCE_REFRESH_CITY_PBF,
        GEOFABRIK_INDEX_MAX_AGE_HOURS,
        GEOFABRIK_INDEX_URL,
        NETWORK_TYPE,
    )


@app.cell
def _(pd):
    def load_config(path, default_osm_key):
        """Read config.txt and resolve each accessibility entry to OSM key/value pairs."""

        def parse_list(value):
            value = value.strip()
            if value == "[]":
                return []
            if not value.startswith("[[") or not value.endswith("]]" ):
                raise ValueError(f"Expected [[value], [value]] syntax, got: {value}")
            inner = value[2:-2]
            return [] if not inner else [item.strip() for item in inner.split("], [")]

        values = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

        places = parse_list(values["place"])
        theta_max = float(values["theta_max"])
        eval_points = int(values["eval_points"])
        order = int(values["order"])
        # distance_scale_m is kept as a reserved legacy key so older config files
        # still load, but it is intentionally not used. FEATHER now receives raw
        # shortest-path distances in metres.
        reserved = {"place", "theta_max", "eval_points", "order", "distance_scale_m"}
        categories = {
            key: parse_list(value)
            for key, value in values.items()
            if key not in reserved
        }

        resolved = {}
        for category, entries in categories.items():
            resolved[category] = []
            for entry in entries:
                if "=" in entry:
                    osm_key, osm_tag = entry.split("=", 1)
                else:
                    if category not in default_osm_key:
                        raise ValueError(
                            f"'{category}' uses bare value '{entry}', but has no default OSM key."
                        )
                    osm_key, osm_tag = default_osm_key[category], entry

                resolved[category].append(
                    {"osm_key": osm_key.strip(), "osm_tag": osm_tag.strip()}
                )

        metadata = pd.DataFrame(
            {
                "place": places,
                "theta_max": theta_max,
                "eval_points": eval_points,
                "order": order,
            }
        )

        return {
            "text": path.read_text(encoding="utf-8"),
            "places": places,
            "theta_max": theta_max,
            "eval_points": eval_points,
            "order": order,
            "categories": categories,
            "resolved_categories": resolved,
            "metadata": metadata,
        }

    return (load_config,)


@app.cell
def _(CONFIG_PATH, DEFAULT_OSM_KEY, load_config):
    config = load_config(CONFIG_PATH, DEFAULT_OSM_KEY)

    places = config["places"]
    theta_max = config["theta_max"]
    eval_points = config["eval_points"]
    order = config["order"]
    accessibility_categories = config["categories"]
    resolved_categories = config["resolved_categories"]
    return (
        accessibility_categories,
        config,
        eval_points,
        order,
        places,
        resolved_categories,
        theta_max,
    )


@app.cell
def _(config, mo, pd, resolved_categories):
    _rows = [
        {"category": category, **item}
        for category, items in resolved_categories.items()
        for item in items
    ]

    resolved_config_df = pd.DataFrame(_rows)
    mo.vstack(
        [
            mo.md("## Parsed config"),
            config["metadata"],
            mo.md("## Resolved OSM tags"),
            resolved_config_df,
        ]
    )
    return


@app.cell
def _(OSM, re, shutil):
    def slugify(value):
        return re.sub(r"[^a-z0-9]+", "_", value.lower().strip()).strip("_")

    def safe_filename(value):
        return re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_")

    def require_dependencies():
        missing = []
        if OSM is None:
            missing.append("pyrosm")
        if shutil.which("osmium") is None:
            missing.append("osmium-tool")
        if missing:
            raise RuntimeError(
                "Missing dependencies: " + ", ".join(missing) + ". "
                "Run: pixi add pyrosm osmium-tool"
            )

    return require_dependencies, safe_filename, slugify


@app.cell
def _(
    CACHE_ROOT,
    CLIP_BUFFER_M,
    GEOFABRIK_INDEX_MAX_AGE_HOURS,
    GEOFABRIK_INDEX_URL,
    gpd,
    json,
    ox,
    pd,
    requests,
    shape,
    slugify,
    time,
):
    def get_geofabrik_index():
        """Download the small Geofabrik region index only when its cache is stale."""
        cache_dir = CACHE_ROOT / "_geofabrik"
        cache_dir.mkdir(parents=True, exist_ok=True)
        index_path = cache_dir / "index-v1.json"

        stale = True
        if index_path.exists():
            age_hours = (time.time() - index_path.stat().st_mtime) / 3600
            stale = age_hours > GEOFABRIK_INDEX_MAX_AGE_HOURS

        if stale:
            try:
                print("Refreshing Geofabrik index...")
                response = requests.get(GEOFABRIK_INDEX_URL, timeout=60)
                response.raise_for_status()
                index_path.write_bytes(response.content)
            except Exception:
                if not index_path.exists():
                    raise
                print("Geofabrik refresh failed; using cached index.")

        return json.loads(index_path.read_text(encoding="utf-8"))

    def city_boundaries(place):
        """Return the exact municipality boundary and a slightly buffered clip boundary."""
        city_dir = CACHE_ROOT / slugify(place)
        city_dir.mkdir(parents=True, exist_ok=True)
        exact_path = city_dir / "municipality_boundary.geojson"
        clip_path = city_dir / f"municipality_clip_boundary_{CLIP_BUFFER_M}m.geojson"

        if exact_path.exists():
            exact = gpd.read_file(exact_path).to_crs("EPSG:4326")
            boundary_changed = False
        else:
            print(f"Geocoding boundary for {place}...")
            geocoded = ox.geocoder.geocode_to_gdf(place).to_crs("EPSG:4326")
            if geocoded.empty:
                raise ValueError(f"No geocoding result for '{place}'.")
            exact = gpd.GeoDataFrame(
                {"place": [place]},
                geometry=[geocoded.geometry.union_all()],
                crs="EPSG:4326",
            )
            exact.to_file(exact_path, driver="GeoJSON")
            boundary_changed = True

        if boundary_changed or not clip_path.exists():
            metric_crs = exact.estimate_utm_crs() or "EPSG:3857"
            buffered = exact.to_crs(metric_crs).geometry.iloc[0].buffer(CLIP_BUFFER_M)
            clip = gpd.GeoDataFrame(
                {"place": [place]}, geometry=[buffered], crs=metric_crs
            ).to_crs("EPSG:4326")
            clip.to_file(clip_path, driver="GeoJSON")
        else:
            clip = gpd.read_file(clip_path).to_crs("EPSG:4326")

        return {
            "city_dir": city_dir,
            "exact_boundary_path": exact_path,
            "clip_boundary": clip,
            "clip_boundary_path": clip_path,
        }

    def select_source(index, clip_boundary):
        """Choose the smallest European Geofabrik region that covers the city."""
        features = index["features"]
        properties = {
            feature["properties"]["id"]: feature["properties"]
            for feature in features
            if feature.get("properties", {}).get("id")
        }

        def under_europe(region_id):
            current, seen = region_id, set()
            while current and current not in seen:
                if current == "europe":
                    return True
                seen.add(current)
                current = properties.get(current, {}).get("parent")
            return False

        city_geometry = clip_boundary.geometry.union_all()
        city_point = city_geometry.representative_point()
        candidates = []

        for feature in features:
            prop = feature.get("properties", {})
            region_id = prop.get("id")
            pbf_url = prop.get("urls", {}).get("pbf")
            geometry_json = feature.get("geometry")

            if not region_id or not pbf_url or geometry_json is None:
                continue
            if not under_europe(region_id):
                continue

            region = shape(geometry_json)
            if not region.is_valid:
                region = region.buffer(0)
            if not region.covers(city_point):
                continue

            candidates.append(
                {
                    "source_id": region_id,
                    "source_name": prop.get("name", region_id),
                    "source_url": pbf_url,
                    "covers_full_city": region.covers(city_geometry),
                    "area": region.area,
                }
            )

        if not candidates:
            raise ValueError("No European Geofabrik extract covers the municipality.")

        full = [candidate for candidate in candidates if candidate["covers_full_city"]]
        return min(full or candidates, key=lambda candidate: candidate["area"])

    def build_download_plan(places, index):
        rows = []
        for place in places:
            boundary = city_boundaries(place)
            source = select_source(index, boundary["clip_boundary"])
            city_pbf_path = boundary["city_dir"] / f"{slugify(place)}.osm.pbf"

            rows.append(
                {
                    "place": place,
                    "city_dir": boundary["city_dir"],
                    "exact_boundary_path": boundary["exact_boundary_path"],
                    "clip_boundary_path": boundary["clip_boundary_path"],
                    **source,
                    "city_pbf_path": city_pbf_path,
                    "city_pbf_exists": city_pbf_path.exists(),
                }
            )

        return pd.DataFrame(rows)

    return build_download_plan, get_geofabrik_index


@app.cell
def _(
    CACHE_ROOT,
    CLIP_BUFFER_M,
    DELETE_REGIONAL_PBF,
    FORCE_REFRESH_CITY_PBF,
    datetime,
    json,
    pd,
    requests,
    require_dependencies,
    slugify,
    subprocess,
    timezone,
):
    def download_pbf(url, destination):
        """Stream a large file to *.part, then rename only after a complete download."""
        if destination.exists():
            print(f"Reusing temporary regional PBF: {destination}")
            return {}

        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_name(destination.name + ".part")
        part.unlink(missing_ok=True)

        print(f"Downloading {url}")
        with requests.get(url, stream=True, timeout=(30, 900)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0) or 0)
            downloaded = 0
            next_report = 0.1

            with part.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded / total >= next_report:
                        print(f"  {100 * downloaded / total:.0f}%")
                        next_report += 0.1

            part.replace(destination)
            return {
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            }

    def clip_pbf(regional_pbf, polygon_path, output_pbf):
        output_pbf.unlink(missing_ok=True)
        result = subprocess.run(
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
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("osmium extract failed:\n" + result.stderr)
        if not output_pbf.exists():
            raise RuntimeError(f"osmium did not create {output_pbf}")

    def osmium_fileinfo(pbf_path):
        try:
            result = subprocess.run(
                ["osmium", "fileinfo", "--json", str(pbf_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except Exception:
            return None

    def prepare_city_pbfs(download_plan_df):
        """Create persistent city PBFs; download each shared regional PBF only once."""
        require_dependencies()
        regional_dir = CACHE_ROOT / "_geofabrik" / "regional_temp"
        regional_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        missing = []

        for row in download_plan_df.to_dict(orient="records"):
            city_pbf = row["city_pbf_path"]
            if city_pbf.exists() and not FORCE_REFRESH_CITY_PBF:
                results[row["place"]] = {
                    "place": row["place"],
                    "status": "cached_city_pbf",
                    "source_name": row["source_name"],
                    "source_url": row["source_url"],
                    "city_pbf_path": city_pbf,
                    "city_pbf_mb": city_pbf.stat().st_size / 1024**2,
                }
            else:
                missing.append(row)

        if missing:
            missing_df = pd.DataFrame(missing)
            for source_url, source_rows in missing_df.groupby("source_url", sort=False):
                first = source_rows.iloc[0]
                regional_pbf = regional_dir / f"{slugify(first['source_id'])}-latest.osm.pbf"
                completed = False

                try:
                    download_metadata = download_pbf(source_url, regional_pbf)
                    regional_size = regional_pbf.stat().st_size

                    for row in source_rows.to_dict(orient="records"):
                        place = row["place"]
                        print(f"Clipping city PBF: {place}")
                        clip_pbf(regional_pbf, row["clip_boundary_path"], row["city_pbf_path"])

                        provenance = {
                            "place": place,
                            "source_id": row["source_id"],
                            "source_name": row["source_name"],
                            "source_url": row["source_url"],
                            "clip_buffer_m": CLIP_BUFFER_M,
                            "regional_file_size_bytes": regional_size,
                            "city_file_size_bytes": row["city_pbf_path"].stat().st_size,
                            "download": download_metadata,
                            "city_pbf_fileinfo": osmium_fileinfo(row["city_pbf_path"]),
                        }
                        (row["city_dir"] / "geofabrik_source.json").write_text(
                            json.dumps(provenance, indent=2), encoding="utf-8"
                        )

                        results[place] = {
                            "place": place,
                            "status": "downloaded_clipped",
                            "source_name": row["source_name"],
                            "source_url": row["source_url"],
                            "city_pbf_path": row["city_pbf_path"],
                            "city_pbf_mb": row["city_pbf_path"].stat().st_size / 1024**2,
                        }

                    completed = True
                finally:
                    if DELETE_REGIONAL_PBF and completed and regional_pbf.exists():
                        print(f"Deleting temporary regional PBF: {regional_pbf}")
                        regional_pbf.unlink()
                    elif not completed and regional_pbf.exists():
                        print(f"Keeping regional PBF after failure: {regional_pbf}")

        ordered = [results[place] for place in download_plan_df["place"].tolist()]
        return pd.DataFrame(ordered)

    return (prepare_city_pbfs,)


@app.cell
def _(
    FORCE_REBUILD_DERIVED,
    NETWORK_TYPE,
    OSM,
    nx,
    ox,
    pickle,
    require_dependencies,
    time,
):
    def build_graph_from_pbf(place, city_pbf_path, city_dir):
        """Build and cache a simplified, undirected, largest-component walking graph."""
        require_dependencies()
        cache_path = city_dir / f"city_graph_{NETWORK_TYPE}_simplified_undirected.pkl"
        cache_is_current = (
            not FORCE_REBUILD_DERIVED
            and cache_path.exists()
            and cache_path.stat().st_mtime >= city_pbf_path.stat().st_mtime
        )
        started = time.time()

        if cache_is_current:
            print(f"Loading cached graph for {place}")
            with cache_path.open("rb") as file:
                graph = pickle.load(file)
            status = "cached"
        else:
            print(f"Building walking graph for {place}")
            osm = OSM(str(city_pbf_path), engine="out_of_core")
            network_nodes, network_edges = osm.get_network(
                network_type=NETWORK_TYPE,
                nodes=True,
            )
            if network_nodes is None or network_edges is None or network_nodes.empty or network_edges.empty:
                raise ValueError(f"No {NETWORK_TYPE} network found for {place}.")

            graph = osm.to_graph(
                network_nodes,
                network_edges,
                graph_type="networkx",
                network_type=NETWORK_TYPE,
                simplify=False,
            )
            if graph is None:
                raise ValueError(f"Could not create graph for {place}.")
            graph.graph.setdefault("crs", "EPSG:4326")

            graph = ox.simplification.simplify_graph(graph)
            try:
                graph = ox.convert.to_undirected(graph)
            except Exception:
                graph = nx.MultiGraph(graph)
                graph.graph["crs"] = "EPSG:4326"

            if graph.is_multigraph():
                self_loops = list(nx.selfloop_edges(graph, keys=True))
            else:
                self_loops = list(nx.selfloop_edges(graph))
            graph.remove_edges_from(self_loops)
            if graph.number_of_nodes() == 0:
                raise ValueError(f"Graph for {place} is empty after simplification.")

            if not nx.is_connected(graph):
                largest = max(nx.connected_components(graph), key=len)
                graph = graph.subgraph(largest).copy()

            with cache_path.open("wb") as file:
                pickle.dump(graph, file, protocol=pickle.HIGHEST_PROTOCOL)
            status = "built"

        projected_graph = ox.project_graph(graph)
        nodes, edges = ox.graph_to_gdfs(projected_graph, nodes=True, edges=True)

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
def _(
    FORCE_REBUILD_DERIVED,
    OSM,
    gpd,
    hashlib,
    json,
    pd,
    require_dependencies,
):
    def extract_pois_from_city_pbf(
        place,
        city_pbf_path,
        city_dir,
        exact_boundary_path,
        resolved_categories,
    ):
        """Extract configured POIs from the same city PBF used to build the graph."""
        require_dependencies()

        tags_by_key = {}
        pair_to_categories = {}
        for category, mappings in resolved_categories.items():
            for item in mappings:
                key, tag = item["osm_key"], item["osm_tag"]
                tags_by_key.setdefault(key, [])
                if tag not in tags_by_key[key]:
                    tags_by_key[key].append(tag)
                pair_to_categories.setdefault((key, tag), []).append(category)

        poi_path = city_dir / "accessibility_pois.geojson"
        signature_path = city_dir / "accessibility_pois_signature.txt"
        signature = hashlib.sha256(
            json.dumps(resolved_categories, sort_keys=True).encode("utf-8")
        ).hexdigest()

        cache_is_current = (
            not FORCE_REBUILD_DERIVED
            and poi_path.exists()
            and signature_path.exists()
            and poi_path.stat().st_mtime >= city_pbf_path.stat().st_mtime
            and signature_path.read_text(encoding="utf-8").strip() == signature
        )

        if cache_is_current:
            print(f"Loading cached POIs for {place}")
            features = gpd.read_file(poi_path)
            status = "cached"
        else:
            print(f"Extracting POIs for {place}")
            osm = OSM(str(city_pbf_path), engine="out_of_core")
            columns = list(tags_by_key) + (["name"] if "name" not in tags_by_key else [])
            raw = osm.get_data_by_custom_criteria(
                custom_filter=tags_by_key,
                osm_keys_to_keep=list(tags_by_key),
                tags_as_columns=columns,
                keep_nodes=True,
                keep_ways=True,
                keep_relations=True,
                keep_other_tags=False,
            )

            rows = []
            if raw is not None and not raw.empty:
                if raw.crs is None:
                    raw = raw.set_crs("EPSG:4326")
                raw = raw.to_crs("EPSG:4326")
                raw = raw[raw.geometry.notna()].copy()
                raw["geometry"] = raw.geometry.apply(
                    lambda geometry: geometry
                    if geometry.geom_type == "Point"
                    else geometry.representative_point()
                )

                exact_boundary = (
                    gpd.read_file(exact_boundary_path)
                    .to_crs("EPSG:4326")
                    .geometry.union_all()
                )
                raw = raw[raw.geometry.apply(exact_boundary.covers)].copy()

                for _, source_row in raw.iterrows():
                    for key, wanted_tags in tags_by_key.items():
                        value = source_row.get(key)
                        if pd.isna(value) or str(value) not in wanted_tags:
                            continue
                        tag = str(value)
                        for category in pair_to_categories.get((key, tag), []):
                            rows.append(
                                {
                                    "osm_type": source_row.get("osm_type"),
                                    "osm_id": source_row.get("id"),
                                    "category": category,
                                    "osm_key": key,
                                    "osm_tag": tag,
                                    "name": source_row.get("name"),
                                    "geometry": source_row["geometry"],
                                }
                            )

            features = gpd.GeoDataFrame(
                rows,
                columns=["osm_type", "osm_id", "category", "osm_key", "osm_tag", "name", "geometry"],
                geometry="geometry",
                crs="EPSG:4326",
            )
            if not features.empty:
                features = features.drop_duplicates(
                    subset=["osm_type", "osm_id", "category", "osm_key", "osm_tag"]
                ).reset_index(drop=True)
                features.to_file(poi_path, driver="GeoJSON")
                signature_path.write_text(signature, encoding="utf-8")
            status = "extracted"

        category_counts = pd.DataFrame(
            {
                "category": list(resolved_categories),
                "count": [
                    int((features["category"] == category).sum()) if not features.empty else 0
                    for category in resolved_categories
                ],
            }
        )

        if features.empty:
            tag_counts = pd.DataFrame(columns=["category", "osm_key", "osm_tag", "count"])
        else:
            tag_counts = (
                features.groupby(["category", "osm_key", "osm_tag"])
                .size()
                .reset_index(name="count")
                .sort_values(["category", "osm_key", "osm_tag"])
                .reset_index(drop=True)
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
    def build_city_datasets(download_plan_df, resolved_categories):
        """Build/load graph and POI data for every configured city."""
        graphs, poi_results, metadata = {}, {}, []

        for row in download_plan_df.to_dict(orient="records"):
            place = row["place"]
            city_pbf = row["city_pbf_path"]
            if not city_pbf.exists():
                raise FileNotFoundError(f"City PBF is missing for {place}: {city_pbf}")

            print(f"\nBuilding graph and POIs: {place}")
            graph_data = build_graph_from_pbf(place, city_pbf, row["city_dir"])
            poi_data = extract_pois_from_city_pbf(
                place,
                city_pbf,
                row["city_dir"],
                row["exact_boundary_path"],
                resolved_categories,
            )

            graphs[place] = graph_data
            poi_results[place] = poi_data
            unique_pois = (
                len(poi_data["features"].drop_duplicates(subset=["osm_type", "osm_id"]))
                if not poi_data["features"].empty
                else 0
            )
            metadata.append(
                {
                    "place": place,
                    "geofabrik_source": row["source_name"],
                    "city_pbf_mb": city_pbf.stat().st_size / 1024**2,
                    "graph_nodes": graph_data["graph"].number_of_nodes(),
                    "graph_edges": graph_data["graph"].number_of_edges(),
                    "graph_status": graph_data["graph_status"],
                    "graph_runtime_seconds": round(graph_data["graph_runtime_seconds"], 2),
                    "poi_records": len(poi_data["features"]),
                    "unique_osm_pois": unique_pois,
                    "poi_status": poi_data["poi_status"],
                }
            )

        return graphs, poi_results, pd.DataFrame(metadata)

    return (build_city_datasets,)


@app.cell
def _(build_download_plan, get_geofabrik_index, places):
    geofabrik_index = get_geofabrik_index()
    download_plan_df = build_download_plan(places, geofabrik_index)
    return (download_plan_df,)


@app.cell
def _(download_plan_df, mo):
    mo.vstack(
        [
            mo.md("## Geofabrik download plan"),
            download_plan_df[
                ["place", "source_name", "source_url", "city_pbf_exists", "city_pbf_path"]
            ],
        ]
    )
    return


@app.cell
def _(
    build_city_datasets,
    download_plan_df,
    prepare_city_pbfs,
    resolved_categories,
):
    # Run automatically when the workbook opens. Existing PBF/graph caches are
    # reused by the helper functions, so cached cities do not need to be rebuilt.
    pbf_results_df = prepare_city_pbfs(download_plan_df)
    graphs, poi_results, pipeline_metadata_df = build_city_datasets(
        download_plan_df, resolved_categories
    )
    return graphs, pbf_results_df, pipeline_metadata_df, poi_results


@app.cell
def _(mo, pbf_results_df, pipeline_metadata_df):
    mo.vstack(
        [
            mo.md("## Local PBFs"),
            pbf_results_df,
            mo.md("## Graph and POI summary"),
            pipeline_metadata_df,
        ]
    )
    return


@app.cell
def _(mo, places):
    selected_city = mo.ui.dropdown(
        options=places,
        value=places[0] if places else None,
        label="Inspect city",
    )
    selected_city
    return (selected_city,)


@app.cell
def _(mo, poi_results, selected_city):
    _place = selected_city.value
    if _place in poi_results:
        _display = mo.vstack(
            [
                mo.md(f"## POI counts - {_place}"),
                poi_results[_place]["category_counts"],
                mo.md("### OSM key/tag counts"),
                poi_results[_place]["tag_counts"],
            ]
        )
    else:
        _display = mo.md("POI data is not available for the selected city.")
    _display
    return


@app.cell
def _(ox):
    def plot_poi_map(place, graph_data, features, color_by_type):
        figure, axis = ox.plot_graph(
            graph_data["projected_graph"],
            node_size=0,
            edge_color="lightgray",
            edge_linewidth=0.5,
            bgcolor="white",
            show=False,
            close=False,
            figsize=(12, 12),
        )

        if not features.empty:
            projected = features.to_crs(graph_data["nodes"].crs)
            for (osm_key, osm_tag), group in projected.groupby(["osm_key", "osm_tag"]):
                label = f"{osm_key}={osm_tag}"
                group.plot(
                    ax=axis,
                    color=color_by_type[label],
                    markersize=25,
                    alpha=0.9,
                    label=label,
                    zorder=3,
                )
            axis.legend(
                title="POI type",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                fontsize=8,
                title_fontsize=9,
            )

        axis.set_title(f"Street graph and POIs - {place}", fontsize=14)
        figure.tight_layout()
        return figure

    return (plot_poi_map,)


@app.cell
def _(graphs, mo, places, plot_poi_map, plt, poi_results):
    if not graphs:
        _display = mo.md("POI map data is not available.")
    else:
        _types = sorted(
            {
                f"{key}={tag}"
                for place in places
                if place in poi_results
                for key, tag in zip(
                    poi_results[place]["features"]["osm_key"],
                    poi_results[place]["features"]["osm_tag"],
                )
            }
        )
        _cmap = plt.get_cmap("gist_ncar", max(len(_types), 1))
        _colors = {name: _cmap(i) for i, name in enumerate(_types)}

        _items = []
        for _place in places:
            if _place not in graphs:
                continue
            _items.extend(
                [
                    mo.md(f"## {_place}"),
                    plot_poi_map(
                        _place,
                        graphs[_place],
                        poi_results[_place]["features"],
                        _colors,
                    ),
                ]
            )
        _display = mo.vstack(_items, gap=2)

    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    # FEATHER_new

    For each graph node and accessibility category, the notebook computes the
    **shortest walking-network distance to the nearest POI**. These distances are
    passed to FEATHER_new directly in **metres**. No 20 km scaling or other
    normalization is applied.

    Therefore the characteristic-function phase is `theta * distance_m`.
    This can oscillate very rapidly for large distances; that behaviour is
    intentional in this raw-metres experiment.
    """)
    return


@app.cell
def _(np, nx, ox):
    def build_distance_feature_matrix(graph_data, poi_data, feature_names):
        """Return nearest-POI walking-network distances in metres."""
        graph = graph_data["graph"]
        original_nodes = list(graph.nodes())
        X_distance = np.full((len(original_nodes), len(feature_names)), np.nan, dtype=float)
        poi_node_counts = {name: 0 for name in feature_names}
        features = poi_data["features"]

        if features.empty:
            return original_nodes, X_distance, poi_node_counts

        projected = features.to_crs(graph_data["nodes"].crs).copy()
        projected["_point"] = projected.geometry.apply(
            lambda geometry: geometry
            if geometry.geom_type == "Point"
            else geometry.representative_point()
        )
        projected["_nearest_node"] = np.atleast_1d(
            ox.distance.nearest_nodes(
                graph_data["projected_graph"],
                X=projected["_point"].x.to_numpy(),
                Y=projected["_point"].y.to_numpy(),
            )
        )

        for feature_index, feature_name in enumerate(feature_names):
            source_nodes = (
                projected.loc[projected["category"] == feature_name, "_nearest_node"]
                .dropna()
                .drop_duplicates()
                .tolist()
            )
            source_nodes = [node for node in source_nodes if node in graph]
            poi_node_counts[feature_name] = len(source_nodes)

            if not source_nodes:
                continue

            distances = nx.multi_source_dijkstra_path_length(
                graph, source_nodes, weight="length"
            )
            X_distance[:, feature_index] = [
                float(distances[node]) if node in distances else np.nan
                for node in original_nodes
            ]

        return original_nodes, X_distance, poi_node_counts

    return (build_distance_feature_matrix,)


@app.cell
def _(
    FEATHER_new,
    accessibility_categories,
    build_distance_feature_matrix,
    eval_points,
    np,
    nx,
    order,
    pd,
    theta_max,
):
    def embedding_column_names(feature_names):
        columns = []
        for r in range(1, order + 1):
            for part in ("real", "imag"):
                for feature in feature_names:
                    columns.extend(
                        f"{feature}__r{r}__theta{i}__{part}"
                        for i in range(eval_points)
                    )
        return columns

    def build_feather_city_embedding(place, graph_data, poi_data):
        """Create raw distance features, run FEATHER_new, and build the CSV table."""
        if FEATHER_new is None:
            raise ImportError("FEATHER_new could not be imported from FEATHER/feather.py.")

        graph = graph_data["graph"]
        feature_names = list(accessibility_categories)
        original_nodes, X_distance, poi_node_counts = build_distance_feature_matrix(
            graph_data, poi_data, feature_names
        )

        missing_features = [
            feature_names[i]
            for i in range(len(feature_names))
            if not np.isfinite(X_distance[:, i]).all()
        ]
        if missing_features:
            raise ValueError(
                f"{place}: missing finite nearest-POI distances for: "
                + ", ".join(missing_features)
            )

        # FEATHER_new expects node IDs 0..N-1 because it constructs matrices with range(N).
        node_mapping = {old: new for new, old in enumerate(original_nodes)}
        feather_graph = nx.relabel_nodes(graph, node_mapping, copy=True)

        # Raw-metres experiment: FEATHER receives the shortest-path distances
        # exactly as calculated above. No normalization is applied.
        X_feather = X_distance

        model = FEATHER_new(theta_max=theta_max, eval_points=eval_points, order=order)
        model.fit(feather_graph, X_feather)
        embedding = model.get_embedding()

        embedding_columns = embedding_column_names(feature_names)
        expected = 2 * eval_points * len(feature_names) * order
        if embedding.shape != (len(original_nodes), expected):
            raise ValueError(
                f"Unexpected FEATHER shape {embedding.shape}; expected "
                f"({len(original_nodes)}, {expected})."
            )

        metadata_df = pd.DataFrame(
            {
                "place": place,
                "osmid": original_nodes,
                "longitude": [graph.nodes[node].get("x", np.nan) for node in original_nodes],
                "latitude": [graph.nodes[node].get("y", np.nan) for node in original_nodes],
            }
        )

        feature_columns = {
            f"distance__{feature}_m": X_distance[:, i]
            for i, feature in enumerate(feature_names)
        }

        result_df = pd.concat(
            [
                metadata_df,
                pd.DataFrame(feature_columns),
                pd.DataFrame(embedding, columns=embedding_columns),
            ],
            axis=1,
        )

        feature_stats = pd.DataFrame(
            {
                "category": feature_names,
                "poi_nodes": [poi_node_counts[name] for name in feature_names],
                "min_distance_m": X_distance.min(axis=0),
                "mean_distance_m": X_distance.mean(axis=0),
                "max_distance_m": X_distance.max(axis=0),
            }
        )

        print(
            f"{place}: {len(original_nodes)} nodes, {len(feature_names)} features, "
            f"embedding {embedding.shape}"
        )

        return {
            "dataframe": result_df,
            "feature_stats": feature_stats,
            "embedding_shape": embedding.shape,
            "feature_names": feature_names,
        }

    return (build_feather_city_embedding,)


@app.cell
def _(
    FEATHER_OUTPUT_DIR,
    build_feather_city_embedding,
    graphs,
    pd,
    places,
    poi_results,
    safe_filename,
):
    FEATHER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feather_new_results = {}
    _summary = []
    _preview = []

    for _place in places:
        if _place not in graphs or _place not in poi_results:
            _summary.append({"place": _place, "status": "graph or POIs not loaded"})
            continue

        try:
            _result = build_feather_city_embedding(
                _place, graphs[_place], poi_results[_place]
            )
            _csv_path = FEATHER_OUTPUT_DIR / f"{safe_filename(_place)}.csv"
            _result["dataframe"].to_csv(_csv_path, index=False)

            feather_new_results[_place] = {
                "csv_path": _csv_path,
                "feature_names": _result["feature_names"],
                "feature_stats": _result["feature_stats"],
                "embedding_shape": _result["embedding_shape"],
            }
            _summary.append(
                {
                    "place": _place,
                    "status": "ok",
                    "nodes": len(_result["dataframe"]),
                    "embedding_columns": _result["embedding_shape"][1],
                    "csv_path": str(_csv_path),
                }
            )
            _preview.append(_result["dataframe"].head(2))
        except Exception as _error:
            _summary.append({"place": _place, "status": str(_error)})
            print(f"FEATHER skipped {_place}: {_error}")

    feather_new_summary_df = pd.DataFrame(_summary)
    feather_new_preview_df = (
        pd.concat(_preview, ignore_index=True) if _preview else pd.DataFrame()
    )
    return feather_new_preview_df, feather_new_results, feather_new_summary_df


@app.cell
def _(FEATHER_OUTPUT_DIR, feather_new_preview_df, feather_new_summary_df, mo):
    mo.vstack(
        [
            mo.md("## FEATHER output"),
            feather_new_summary_df,
            mo.md(f"Files are stored in `{FEATHER_OUTPUT_DIR}`"),
            mo.md("### Two-row preview per successful city"),
            feather_new_preview_df,
        ]
    )
    return


@app.cell
def _(accessibility_categories, feather_new_results, mo, order):
    if feather_new_results:
        feather_map_place = mo.ui.dropdown(
            options=list(feather_new_results),
            value=next(iter(feather_new_results)),
            label="Place",
        )
        feather_map_category = mo.ui.dropdown(
            options=list(accessibility_categories),
            value=next(iter(accessibility_categories)),
            label="Category",
        )
        feather_map_order = mo.ui.slider(
            start=1,
            stop=order,
            step=1,
            value=order,
            label="Random-walk order r",
        )
        feather_map_value = mo.ui.dropdown(
            options=[
                "nearest_distance_m",
                "magnitude",
                "real",
                "imaginary",
            ],
            value="nearest_distance_m",
            label="Map value",
        )
        _controls = mo.hstack(
            [feather_map_place, feather_map_category, feather_map_order, feather_map_value],
            justify="start",
            gap=2,
        )
    else:
        feather_map_place = None
        feather_map_category = None
        feather_map_order = None
        feather_map_value = None
        _controls = mo.md("FEATHER output is not available for any city.")

    _controls
    return (
        feather_map_category,
        feather_map_order,
        feather_map_place,
        feather_map_value,
    )


@app.cell
def _(
    eval_points,
    feather_map_category,
    feather_map_order,
    feather_map_place,
    feather_map_value,
    feather_new_results,
    graphs,
    mo,
    np,
    ox,
    pd,
):
    if feather_map_place is None or feather_map_place.value not in feather_new_results:
        _display = mo.md("FEATHER output is not available for the selected city.")
    else:
        _place = feather_map_place.value
        _category = feather_map_category.value
        _r = int(feather_map_order.value)
        _value_type = feather_map_value.value
        _df = pd.read_csv(feather_new_results[_place]["csv_path"])

        _real_cols = [f"{_category}__r{_r}__theta{i}__real" for i in range(eval_points)]
        _imag_cols = [f"{_category}__r{_r}__theta{i}__imag" for i in range(eval_points)]
        _real = _df[_real_cols].to_numpy(float)
        _imag = _df[_imag_cols].to_numpy(float)

        if _value_type == "nearest_distance_m":
            _scores = _df[f"distance__{_category}_m"].to_numpy(float)
            _label = f"Nearest {_category} POI distance (m)"
            _limits = (0.0, max(float(np.nanmax(_scores)), 1.0))
            _title = f"Nearest {_category} POI distance - {_place}"
        elif _value_type == "magnitude":
            _scores = np.sqrt(_real**2 + _imag**2).sum(axis=1)
            _label = f"Magnitude sum - {_category}, r={_r}"
            _limits = (0.0, float(eval_points))
            _title = f"FEATHER_new - {_place}\n{_category}, r={_r}"
        elif _value_type == "real":
            _scores = _real.mean(axis=1)
            _label = f"Mean real value - {_category}, r={_r}"
            _limits = (-1.0, 1.0)
            _title = f"FEATHER_new - {_place}\n{_category}, r={_r}"
        else:
            _scores = _imag.mean(axis=1)
            _label = f"Mean imaginary value - {_category}, r={_r}"
            _limits = (-1.0, 1.0)
            _title = f"FEATHER_new - {_place}\n{_category}, r={_r}"

        _lookup = pd.Series(_scores, index=_df["osmid"].astype(str))
        _nodes = graphs[_place]["nodes"].copy()
        _nodes["_score"] = _nodes.index.astype(str).map(_lookup)
        _nodes = _nodes.dropna(subset=["_score"])

        _figure, _axis = ox.plot_graph(
            graphs[_place]["projected_graph"],
            node_size=0,
            edge_color="lightgray",
            edge_linewidth=0.25,
            bgcolor="black",
            show=False,
            close=False,
            figsize=(12, 12),
        )
        _nodes.plot(
            ax=_axis,
            column="_score",
            cmap="plasma",
            markersize=2.5,
            alpha=0.9,
            legend=True,
            zorder=3,
            vmin=_limits[0],
            vmax=_limits[1],
            legend_kwds={"shrink": 0.6, "label": _label, "orientation": "vertical"},
        )
        _axis.set_title(_title, fontsize=14)
        _figure.tight_layout()
        _display = _figure

    _display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Characteristic-function explorer

    The original FEATHER `charfun.jpg` plots characteristic-function values against the evaluation point $\theta$.
    Its two lines are class-wise means and the shaded areas are one standard deviation around those means.

    This urban version uses the same idea. Because the street-network data has no "popular/unpopular" class label,
    it compares two groups for the selected amenity category:

    - **closest nodes**: the selected fraction of nodes with the shortest network distance to the amenity;
    - **farthest nodes**: the same fraction with the largest network distance.

    In this workbook FEATHER receives raw nearest-amenity distances in metres, so the sine/cosine phase is `theta * distance_m` with no 20 km scaling.

    The plot reads only the required columns from an existing FEATHER CSV.
    """)
    return


@app.cell
def _(FEATHER_OUTPUT_DIR, accessibility_categories, mo, order):
    _csv_files = sorted(FEATHER_OUTPUT_DIR.glob("*.csv"))

    if _csv_files:
        charfun_csv_file = mo.ui.dropdown(
            options=[path.name for path in _csv_files],
            value=_csv_files[0].name,
            label="FEATHER CSV",
        )
        charfun_category = mo.ui.dropdown(
            options=list(accessibility_categories),
            value=next(iter(accessibility_categories)),
            label="Amenity category",
        )
        charfun_order = mo.ui.slider(
            start=1,
            stop=order,
            step=1,
            value=order,
            label="Random-walk order r",
        )
        charfun_component = mo.ui.dropdown(
            options=["real", "imaginary"],
            value="real",
            label="Characteristic-function part",
        )
        charfun_tail_fraction = mo.ui.slider(
            start=0.05,
            stop=0.45,
            step=0.05,
            value=0.20,
            label="Closest/farthest fraction",
        )
        _charfun_controls = mo.hstack(
            [
                charfun_csv_file,
                charfun_category,
                charfun_order,
                charfun_component,
                charfun_tail_fraction,
            ],
            justify="start",
            gap=2,
        )
    else:
        charfun_csv_file = None
        charfun_category = None
        charfun_order = None
        charfun_component = None
        charfun_tail_fraction = None
        _charfun_controls = mo.md(
            f"No FEATHER CSV files found in `{FEATHER_OUTPUT_DIR}`. FEATHER output is not available yet."
        )

    _charfun_controls
    return (
        charfun_category,
        charfun_component,
        charfun_csv_file,
        charfun_order,
        charfun_tail_fraction,
    )


@app.cell
def _(
    FEATHER_OUTPUT_DIR,
    charfun_category,
    charfun_component,
    charfun_csv_file,
    charfun_order,
    charfun_tail_fraction,
    eval_points,
    mo,
    np,
    pd,
    plt,
    theta_max,
):
    if charfun_csv_file is None:
        _charfun_display = mo.md("Create a FEATHER CSV first to enable the explorer.")
    else:
        _csv_path = FEATHER_OUTPUT_DIR / charfun_csv_file.value
        _category = charfun_category.value
        _r = int(charfun_order.value)
        _part = "real" if charfun_component.value == "real" else "imag"
        _fraction = float(charfun_tail_fraction.value)

        _distance_column = f"distance__{_category}_m"

        # Read only the CSV header first.  The explorer must use the evaluation
        # points that are actually present in this file, because an older CSV may
        # have been created with a different eval_points value than config.txt.
        _csv_columns = pd.read_csv(_csv_path, nrows=0).columns.tolist()
        _prefix = f"{_category}__r{_r}__theta"
        _suffix = f"__{_part}"

        _theta_columns = []
        for _column in _csv_columns:
            if not (_column.startswith(_prefix) and _column.endswith(_suffix)):
                continue

            _theta_text = _column[len(_prefix) : -len(_suffix)]
            if _theta_text.isdigit():
                _theta_columns.append((int(_theta_text), _column))

        _theta_columns.sort(key=lambda item: item[0])
        _theta_indices = np.array([item[0] for item in _theta_columns], dtype=int)
        _value_columns = [item[1] for item in _theta_columns]

        if _distance_column not in _csv_columns:
            _charfun_display = mo.md(
                f"The CSV does not contain `{_distance_column}`. "
                "Regenerate this city with the current distance-feature workbook."
            )
        elif len(_value_columns) == 0:
            _charfun_display = mo.md(
                f"No FEATHER columns were found for `{_category}`, r={_r}, {_part}."
            )
        else:
            # FEATHER_new always generates theta indices starting at zero.  Using
            # max(index)+1 lets this plot older CSVs without assuming that the
            # current config's eval_points is identical to the file's setting.
            _csv_eval_points = int(_theta_indices.max()) + 1
            _theta_all = np.linspace(0.01, theta_max, _csv_eval_points)
            _theta = _theta_all[_theta_indices]

            _config_warning = None
            if _csv_eval_points != eval_points:
                _config_warning = mo.md(
                    f"**Config mismatch:** this CSV contains {_csv_eval_points} "
                    f"evaluation points, while the current config says {eval_points}. "
                    "The plot uses the columns that actually exist in the CSV. "
                    "If `theta_max` also changed since this CSV was created, regenerate "
                    "the embeddings so the x-axis theta values are exact."
                )

            # Read only the columns needed for this figure, not the full embedding.
            _plot_df = pd.read_csv(
                _csv_path,
                usecols=[_distance_column, *_value_columns],
            )

            _distances = _plot_df[_distance_column].to_numpy(float)
            _values = _plot_df[_value_columns].to_numpy(float)
            _valid = np.isfinite(_distances) & np.isfinite(_values).all(axis=1)
            _distances = _distances[_valid]
            _values = _values[_valid]

            if len(_distances) == 0:
                _charfun_display = mo.md("No finite values were available for this selection.")
            else:
                _near_cutoff = float(np.quantile(_distances, _fraction))
                _far_cutoff = float(np.quantile(_distances, 1.0 - _fraction))

                _near_values = _values[_distances <= _near_cutoff]
                _far_values = _values[_distances >= _far_cutoff]

                if len(_near_values) == 0 or len(_far_values) == 0:
                    _charfun_display = mo.md("The selected groups contain no nodes.")
                else:
                    _near_mean = _near_values.mean(axis=0)
                    _near_std = _near_values.std(axis=0)
                    _far_mean = _far_values.mean(axis=0)
                    _far_std = _far_values.std(axis=0)

                    _fig, _ax = plt.subplots(figsize=(9, 6))

                    _ax.plot(
                        _theta,
                        _near_mean,
                        linewidth=2.0,
                        color="tab:red",
                        label=(
                            f"Closest {int(_fraction * 100)}% "
                            f"(<= {_near_cutoff:.0f} m)"
                        ),
                    )
                    _ax.fill_between(
                        _theta,
                        _near_mean - _near_std,
                        _near_mean + _near_std,
                        color="tab:red",
                        alpha=0.18,
                    )

                    _ax.plot(
                        _theta,
                        _far_mean,
                        linewidth=2.0,
                        color="tab:blue",
                        label=(
                            f"Farthest {int(_fraction * 100)}% "
                            f"(>= {_far_cutoff:.0f} m)"
                        ),
                    )
                    _ax.fill_between(
                        _theta,
                        _far_mean - _far_std,
                        _far_mean + _far_std,
                        color="tab:blue",
                        alpha=0.18,
                    )

                    for _y in (-0.5, 0.0, 0.5):
                        _ax.axhline(
                            _y,
                            color="0.75",
                            linestyle="--",
                            linewidth=1.0,
                            zorder=0,
                        )

                    _part_name = "Real / cosine" if _part == "real" else "Imaginary / sine"
                    _ax.set_xlabel("Evaluation point theta")
                    _ax.set_ylabel("Characteristic function value")
                    _ax.set_xlim(0.0, theta_max)
                    _ax.set_ylim(-1.05, 1.05)
                    _ax.set_title(
                        f"{_category} - {_part_name}, r={_r}\n{_csv_path.stem}"
                    )
                    _ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.14),
                        ncol=2,
                        frameon=True,
                    )
                    _fig.subplots_adjust(bottom=0.22)

                    _display_items = [_fig]
                    if _config_warning is not None:
                        _display_items.append(_config_warning)
                    _display_items.append(
                        mo.md(
                            f"**Nodes plotted:** {len(_near_values):,} closest and "
                            f"{len(_far_values):,} farthest. The line is the mean across "
                            "nodes; the shaded band is +/- one standard deviation."
                        )
                    )
                    _charfun_display = mo.vstack(_display_items)

    _charfun_display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cache layout

    `../cache/<city>/` keeps the city boundary, clipped city PBF, graph pickle, POI GeoJSON, and source metadata. These are the expensive reusable inputs.

    The much larger regional Geofabrik file is temporary. It is downloaded once for all cities that use the same source, clipped into city PBFs, and deleted only after every clip from that source succeeds.

    - Set `FORCE_REFRESH_CITY_PBF = True` to fetch a fresh OSM snapshot.
    - Set `FORCE_REBUILD_DERIVED = True` to rebuild graph and POI caches from the existing city PBF.
    """)
    return


if __name__ == "__main__":
    app.run()
