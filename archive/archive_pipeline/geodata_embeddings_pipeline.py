import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Pipeline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import libraries
    """)
    return


@app.cell
def _():
    import marimo as mo
    import osmnx as ox
    import networkx as nx
    import pandas as pd
    import pandana as pdna
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import sys
    import time
    import requests
    import re
    from shapely.geometry import Point

    sys.path.insert(0, "../")
    from FEATHER.feather import FEATHER, FEATHER_new



    return Point, gpd, mo, nx, os, ox, pd, re, requests, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load config file
    """)
    return


@app.cell
def _(pd):
    config = open("../input/config.txt", "r").read()

    config_values = {}

    for line in config.splitlines():

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)

        config_values[
            key.strip()
        ] = value.strip()


    def parse_config_list(value):

        value = value.strip()

        if value == "[]":
            return []

        value = value[2:-2]

        if not value:
            return []

        return [
            item.strip()
            for item in value.split("], [")
        ]


    # ==========================================================
    # Normal config values
    # ==========================================================

    places = parse_config_list(
        config_values["place"]
    )

    theta_max = float(
        config_values["theta_max"]
    )

    eval_points = int(
        config_values["eval_points"]
    )

    order = int(
        config_values["order"]
    )


    # ==========================================================
    # These are NOT accessibility categories
    # ==========================================================

    reserved_keys = [
        "place",
        "theta_max",
        "eval_points",
        "order",
    ]


    # ==========================================================
    # Accessibility categories
    #
    # Example:
    #
    # healthcare = [[clinic], [hospital]]
    #
    # becomes:
    #
    # {
    #     "healthcare": [
    #         "clinic",
    #         "hospital"
    #     ]
    # }
    # ==========================================================

    accessibility_categories = {}

    for key, value in config_values.items():

        if key in reserved_keys:
            continue

        accessibility_categories[
            key
        ] = parse_config_list(
            value
        )


    # ==========================================================
    # Default OSM key for tags that do NOT explicitly specify one
    #
    # Explicit key=value entries in config.txt always override this.
    # ==========================================================

    default_osm_key = {
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


    # ==========================================================
    # Country
    # ==========================================================

    countries = [
        place.split(",")[-1].strip()
        for place in places
    ]


    # ==========================================================
    # Metadata DataFrame
    # ==========================================================

    city_metadata_df = pd.DataFrame({
        "place": places,
        "country": countries,
        "theta_max": theta_max,
        "eval_points": eval_points,
        "order": order,
        "accessibility_categories": [
            {
                category: values.copy()
                for category, values
                in accessibility_categories.items()
            }
            for place in places
        ],
    })


    city_metadata_df
    return city_metadata_df, default_osm_key


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Load OSMNX graph
    """)
    return


@app.cell
def _(ox):
    #OSMNX settings
    ox.settings.use_cache = True
    ox.settings.log_console = True
    overpass_urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Get graph for place function
    """)
    return


@app.cell
def _(gpd, nx, os, ox, time):
    def get_city_graph(
        place,
        retry_wait=120,
        boundary_tolerance=100,
    ):
        start_time = time.perf_counter()

        # ==========================================================
        # City-specific cache folder
        # ==========================================================

        city_folder_name = (
            place
            .replace(",", "")
            .replace(" ", "_")
            .lower()
        )

        cache_dir = os.path.join(
            "..",
            "cache",
            city_folder_name,
        )

        os.makedirs(
            cache_dir,
            exist_ok=True,
        )

        ox.settings.cache_folder = cache_dir


        # ==========================================================
        # Path to saved graph
        # ==========================================================

        graph_path = os.path.join(
            cache_dir,
            "city_graph.graphml",
        )


        # ==========================================================
        # Load cached graph if it exists
        # ==========================================================

        if os.path.exists(graph_path):

            print(
                f"Loading cached graph for {place}"
            )

            city_graph = ox.load_graphml(
                graph_path
            )


        # ==========================================================
        # Otherwise download graph
        # ==========================================================

        else:

            print(
                f"Finding boundary for {place} ..."
            )

            place_gdf = ox.geocoder.geocode_to_gdf(
                place
            )


            # Project boundary so tolerance is in metres
            projected_place = place_gdf.to_crs(
                place_gdf.estimate_utm_crs()
            )


            # Simplify city boundary
            simplified_polygon = (
                projected_place
                .geometry
                .iloc[0]
                .simplify(
                    tolerance=boundary_tolerance,
                    preserve_topology=True,
                )
            )


            # Convert boundary back to lat/lon
            simplified_polygon = gpd.GeoSeries(
                [simplified_polygon],
                crs=projected_place.crs,
            ).to_crs(
                "EPSG:4326"
            ).iloc[0]


            # Download graph with retries
            attempt = 1

            while True:

                try:

                    print()
                    print(
                        f"Downloading graph for {place} ..."
                    )

                    print(
                        f"Boundary tolerance: "
                        f"{boundary_tolerance} m"
                    )

                    print(
                        f"Attempt {attempt}"
                    )


                    city_graph = ox.graph.graph_from_polygon(
                        simplified_polygon,
                        network_type="walk",
                        simplify=True,
                        retain_all=False,
                    )


                    # Save successful download
                    ox.save_graphml(
                        city_graph,
                        graph_path,
                    )


                    print(
                        f"Graph saved to {graph_path}"
                    )

                    break


                except Exception as error:

                    print()
                    print(
                        f"Graph download attempt "
                        f"{attempt} failed for {place}"
                    )

                    print(error)


                    wait_time = min(
                        retry_wait * (2 ** (attempt - 1)),
                        1800,
                    )


                    print(
                        f"Waiting {wait_time} "
                        f"seconds before retrying..."
                    )


                    time.sleep(
                        wait_time
                    )

                    attempt += 1


        # ==========================================================
        # Clean graph
        # ==========================================================

        city_graph = ox.convert.to_undirected(
            city_graph
        )

        city_graph.remove_edges_from(
            nx.selfloop_edges(
                city_graph
            )
        )


        # ==========================================================
        # Project graph
        # ==========================================================

        city_graph_projected = ox.project_graph(
            city_graph
        )


        # ==========================================================
        # Convert to GeoDataFrames
        # ==========================================================

        city_nodes, city_edges = ox.graph_to_gdfs(
            city_graph_projected
        )

        city_edges = city_edges.reset_index()


        # ==========================================================
        # Runtime
        # ==========================================================

        graph_time_s = round(
            time.perf_counter() - start_time,
            2,
        )


        return (
            city_graph,
            city_graph_projected,
            city_nodes,
            city_edges,
            graph_time_s,
        )

    return (get_city_graph,)


@app.cell
def _(get_city_graph):
    def insert_graph_metadata(
        city_metadata_df,
    ):
        graphs = {}

        for index, row in city_metadata_df.iterrows():

            place = row["place"]

            print()
            print("=" * 60)
            print(f"Processing graph for {place}")
            print("=" * 60)


            (
                city_graph,
                city_graph_projected,
                city_nodes,
                city_edges,
                graph_time_s,
            ) = get_city_graph(
                place
            )


            city_metadata_df.loc[
                index,
                "nodes"
            ] = city_graph.number_of_nodes()


            city_metadata_df.loc[
                index,
                "edges"
            ] = city_graph.number_of_edges()


            city_metadata_df.loc[
                index,
                "graph_time_s"
            ] = graph_time_s


            graphs[place] = {
                "graph": city_graph,
                "projected_graph": city_graph_projected,
                "nodes": city_nodes,
                "edges": city_edges,
            }


        city_metadata_df["nodes"] = (
            city_metadata_df["nodes"]
            .astype(int)
        )

        city_metadata_df["edges"] = (
            city_metadata_df["edges"]
            .astype(int)
        )


        return (
            city_metadata_df,
            graphs,
        )

    return (insert_graph_metadata,)


@app.cell
def _(city_metadata_df, insert_graph_metadata):
    city_metadata_with_graphs_df, graphs = (
        insert_graph_metadata(
            city_metadata_df
        )
    )

    city_metadata_with_graphs_df
    return city_metadata_with_graphs_df, graphs


@app.cell
def _(Point, gpd, os, ox, pd, re, requests, time):
    def get_city_pois(
        row,
        graph_data,
        default_osm_key,
        retry_wait=120,
        max_attempts=5,
    ):
        place = row["place"]

        accessibility_categories = row[
            "accessibility_categories"
        ]

        city_graph_projected = graph_data[
            "projected_graph"
        ]

        city_nodes = graph_data[
            "nodes"
        ]


        # ==========================================================
        # 1. Resolve config categories into OSM key/tag pairs
        #
        # Examples:
        #
        # healthcare -> hospital
        # becomes:
        # amenity=hospital
        #
        # tourism=museum
        # stays:
        # tourism=museum
        # ==========================================================

        resolved_categories = {}


        for category, values in accessibility_categories.items():

            if category not in default_osm_key:

                raise ValueError(
                    f"No default OSM key defined for "
                    f"category '{category}'"
                )


            resolved_categories[
                category
            ] = []


            for value in values:

                if "=" in value:

                    osm_key, osm_tag = value.split(
                        "=",
                        1,
                    )

                else:

                    osm_key = default_osm_key[
                        category
                    ]

                    osm_tag = value


                resolved_categories[
                    category
                ].append({
                    "osm_key": osm_key,
                    "osm_tag": osm_tag,
                })


        # ==========================================================
        # 2. Group tags by actual OSM key
        #
        # Example:
        #
        # amenity:
        #   hospital
        #   cafe
        #   bank
        #
        # shop:
        #   supermarket
        #   bakery
        #
        # tourism:
        #   museum
        #   zoo
        # ==========================================================

        tags_by_osm_key = {}


        for category, mappings in resolved_categories.items():

            for mapping in mappings:

                osm_key = mapping[
                    "osm_key"
                ]

                osm_tag = mapping[
                    "osm_tag"
                ]


                if osm_key not in tags_by_osm_key:

                    tags_by_osm_key[
                        osm_key
                    ] = []


                if osm_tag not in tags_by_osm_key[
                    osm_key
                ]:

                    tags_by_osm_key[
                        osm_key
                    ].append(
                        osm_tag
                    )


        # ==========================================================
        # 3. Map OSM key/tag pairs back to project categories
        # ==========================================================

        pair_to_categories = {}


        for category, mappings in resolved_categories.items():

            for mapping in mappings:

                pair = (
                    mapping["osm_key"],
                    mapping["osm_tag"],
                )


                if pair not in pair_to_categories:

                    pair_to_categories[
                        pair
                    ] = []


                if category not in pair_to_categories[
                    pair
                ]:

                    pair_to_categories[
                        pair
                    ].append(
                        category
                    )


        # ==========================================================
        # 4. City-specific cache
        # ==========================================================

        city_folder_name = (
            place
            .replace(",", "")
            .replace(" ", "_")
            .lower()
        )


        cache_dir = os.path.join(
            "..",
            "cache",
            city_folder_name,
        )


        os.makedirs(
            cache_dir,
            exist_ok=True,
        )


        ox.settings.cache_folder = cache_dir


        poi_path = os.path.join(
            cache_dir,
            "accessibility_pois.geojson",
        )


        poi_config_path = os.path.join(
            cache_dir,
            "accessibility_pois_config.txt",
        )


        # ==========================================================
        # 5. Create cache signature
        #
        # If config.txt changes, this signature changes.
        # That causes the POIs to be downloaded again.
        # ==========================================================

        config_lines = []


        for category, mappings in resolved_categories.items():

            for mapping in mappings:

                config_lines.append(
                    f"{category}|"
                    f"{mapping['osm_key']}|"
                    f"{mapping['osm_tag']}"
                )


        poi_config = "\n".join(
            sorted(config_lines)
        )


        cache_is_valid = (
            os.path.exists(poi_path)
            and
            os.path.exists(poi_config_path)
        )


        if cache_is_valid:

            with open(
                poi_config_path,
                "r",
            ) as file:

                cached_config = file.read()


            cache_is_valid = (
                cached_config
                == poi_config
            )


        # ==========================================================
        # 6. Load cached POIs
        # ==========================================================

        if cache_is_valid:

            print(
                f"Loading cached accessibility POIs "
                f"for {place}"
            )


            features = gpd.read_file(
                poi_path
            )


            download_status = "cached"


        # ==========================================================
        # 7. Otherwise download POIs
        # ==========================================================

        else:

            print(
                f"Finding OSM administrative area "
                f"for {place} ..."
            )


            place_gdf = ox.geocoder.geocode_to_gdf(
                place
            )


            osm_id = int(
                place_gdf.iloc[0][
                    "osm_id"
                ]
            )


            osm_type = place_gdf.iloc[0][
                "osm_type"
            ]


            # ======================================================
            # Convert OSM ID into Overpass area ID
            # ======================================================

            if osm_type == "relation":

                area_id = (
                    3_600_000_000
                    + osm_id
                )

            elif osm_type == "way":

                area_id = (
                    2_400_000_000
                    + osm_id
                )

            else:

                raise ValueError(
                    f"{place} resolved to OSM type "
                    f"'{osm_type}'. "
                    f"Expected relation or way."
                )


            print(
                f"OSM type: {osm_type}"
            )

            print(
                f"OSM ID: {osm_id}"
            )

            print(
                f"Overpass area ID: {area_id}"
            )


            # ======================================================
            # 8. Build Overpass query
            #
            # One query section per actual OSM key.
            # ======================================================

            query_parts = []


            for osm_key, tags in tags_by_osm_key.items():

                tag_regex = "|".join(
                    re.escape(tag)
                    for tag in tags
                )


                query_parts.append(
                    f'''
                    nwr["{osm_key}"~"^({tag_regex})$"]
                    (area.searchArea);
                    '''
                )


            query_body = "\n".join(
                query_parts
            )


            query = f'''
            [out:json][timeout:240];

            area({area_id})->.searchArea;

            (
                {query_body}
            );

            out center tags;
            '''


            # ======================================================
            # 9. Overpass servers
            #
            # If one returns 429 or another request error,
            # try the next server.
            # ======================================================

            overpass_urls = [
                "https://overpass-api.de/api/interpreter",
                "https://overpass.private.coffee/api/interpreter",
            ]


            data = None


            # ======================================================
            # 10. Download with server fallback
            # ======================================================

            for attempt in range(
                1,
                max_attempts + 1,
            ):

                print()

                print(
                    f"Download attempt "
                    f"{attempt}/{max_attempts}"
                )


                for overpass_url in overpass_urls:

                    print()

                    print(
                        f"Trying:"
                    )

                    print(
                        overpass_url
                    )


                    try:

                        response = requests.post(
                            overpass_url,
                            data={
                                "data": query
                            },
                            timeout=300,
                        )


                        # ==========================================
                        # 429 = server temporarily refusing request
                        # ==========================================

                        if response.status_code == 429:

                            print(
                                "429 Too Many Requests"
                            )

                            print(
                                "Trying next Overpass server..."
                            )

                            continue


                        # ==========================================
                        # Other HTTP error
                        # ==========================================

                        response.raise_for_status()


                        # ==========================================
                        # Successful response
                        # ==========================================

                        data = response.json()


                        print(
                            f"Success."
                        )

                        print(
                            f"Received "
                            f"{len(data['elements'])} "
                            f"OSM elements."
                        )


                        break


                    except requests.RequestException as error:

                        print(
                            f"Request failed:"
                        )

                        print(
                            error
                        )

                        print(
                            "Trying next Overpass server..."
                        )


                    except ValueError as error:

                        print(
                            "Could not parse response as JSON:"
                        )

                        print(
                            error
                        )


                # ==================================================
                # One server succeeded
                # ==================================================

                if data is not None:

                    break


                # ==================================================
                # Every server failed
                # ==================================================

                if attempt < max_attempts:

                    wait_time = min(
                        retry_wait * attempt,
                        600,
                    )


                    print()

                    print(
                        "All Overpass servers failed."
                    )

                    print(
                        f"Waiting {wait_time} seconds "
                        f"before attempt {attempt + 1}..."
                    )


                    time.sleep(
                        wait_time
                    )


            # ======================================================
            # 11. Stop if all attempts failed
            # ======================================================

            if data is None:

                raise RuntimeError(
                    f"Could not download accessibility POIs "
                    f"for {place} after "
                    f"{max_attempts} attempts."
                )


            # ======================================================
            # 12. Convert downloaded OSM elements into rows
            # ======================================================

            rows = []


            for element in data[
                "elements"
            ]:

                element_type = element[
                    "type"
                ]


                element_tags = element.get(
                    "tags",
                    {},
                )


                # --------------------------------------------------
                # Coordinates
                # --------------------------------------------------

                if element_type == "node":

                    lat = element.get(
                        "lat"
                    )

                    lon = element.get(
                        "lon"
                    )


                else:

                    center = element.get(
                        "center"
                    )


                    if center is None:

                        continue


                    lat = center.get(
                        "lat"
                    )

                    lon = center.get(
                        "lon"
                    )


                if lat is None or lon is None:

                    continue


                # --------------------------------------------------
                # Match downloaded feature to configured key/tag
                # --------------------------------------------------

                for osm_key, wanted_tags in tags_by_osm_key.items():

                    osm_tag = element_tags.get(
                        osm_key
                    )


                    if osm_tag not in wanted_tags:

                        continue


                    pair = (
                        osm_key,
                        osm_tag,
                    )


                    categories = pair_to_categories.get(
                        pair,
                        [],
                    )


                    # ----------------------------------------------
                    # One OSM feature can belong to more than one
                    # project category if configured that way.
                    # ----------------------------------------------

                    for category in categories:

                        rows.append({
                            "osm_type": element_type,
                            "osm_id": element[
                                "id"
                            ],
                            "category": category,
                            "osm_key": osm_key,
                            "osm_tag": osm_tag,
                            "name": element_tags.get(
                                "name"
                            ),
                            "geometry": Point(
                                lon,
                                lat,
                            ),
                        })


            # ======================================================
            # 13. Create GeoDataFrame
            # ======================================================

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


            # ======================================================
            # 14. Remove duplicate records
            # ======================================================

            if len(features) > 0:

                features = (
                    features
                    .drop_duplicates(
                        subset=[
                            "osm_type",
                            "osm_id",
                            "category",
                            "osm_key",
                            "osm_tag",
                        ]
                    )
                    .reset_index(
                        drop=True
                    )
                )


            print()

            print(
                f"Created "
                f"{len(features)} accessibility POI records."
            )


            # ======================================================
            # 15. Save cache
            # ======================================================

            if len(features) > 0:

                features.to_file(
                    poi_path,
                    driver="GeoJSON",
                )


                with open(
                    poi_config_path,
                    "w",
                ) as file:

                    file.write(
                        poi_config
                    )


            download_status = "downloaded"


        # ==========================================================
        # 16. Count POIs per project category
        # ==========================================================

        if len(features) > 0:

            category_counts = (
                features
                .groupby(
                    "category"
                )
                .size()
                .reset_index(
                    name="count"
                )
                .sort_values(
                    "category"
                )
                .reset_index(
                    drop=True
                )
            )


        else:

            category_counts = pd.DataFrame(
                columns=[
                    "category",
                    "count",
                ]
            )


        # ==========================================================
        # 17. Count individual OSM tags
        # ==========================================================

        if len(features) > 0:

            tag_counts = (
                features
                .groupby(
                    [
                        "category",
                        "osm_key",
                        "osm_tag",
                    ]
                )
                .size()
                .reset_index(
                    name="count"
                )
                .sort_values(
                    [
                        "category",
                        "osm_key",
                        "osm_tag",
                    ]
                )
                .reset_index(
                    drop=True
                )
            )


        else:

            tag_counts = pd.DataFrame(
                columns=[
                    "category",
                    "osm_key",
                    "osm_tag",
                    "count",
                ]
            )


        # ==========================================================
        # 18. Split POIs by project category
        #
        # Example:
        #
        # features_by_category["healthcare"]
        # features_by_category["learning"]
        # ==========================================================

        features_by_category = {}


        for category in accessibility_categories:

            features_by_category[
                category
            ] = features[
                features["category"]
                == category
            ].copy()


        # ==========================================================
        # 19. Plot
        # ==========================================================

        poi_fig, poi_ax = ox.plot_graph(
            city_graph_projected,
            node_size=0,
            edge_color="lightgray",
            edge_linewidth=0.5,
            bgcolor="white",
            show=False,
            close=False,
        )


        if len(features) > 0:

            features_projected = (
                features
                .to_crs(
                    city_nodes.crs
                )
            )


            features_projected.plot(
                ax=poi_ax,
                column="category",
                categorical=True,
                legend=True,
                markersize=20,
                zorder=3,
            )


        poi_ax.set_title(
            f"Accessibility POIs in {place}",
            fontsize=14,
        )


        # ==========================================================
        # 20. Return
        # ==========================================================

        return {
            "features": features,
            "features_by_category": features_by_category,
            "category_counts": category_counts,
            "tag_counts": tag_counts,
            "resolved_categories": resolved_categories,
            "download_status": download_status,
            "poi_fig": poi_fig,
        }

    return (get_city_pois,)


@app.cell
def _(
    city_metadata_with_graphs_df,
    default_osm_key,
    get_city_pois,
    graphs,
    time,
):
    poi_results = {}


    for index, row in city_metadata_with_graphs_df.iterrows():

        place = row["place"]


        print()

        print("=" * 60)

        print(
            f"Processing accessibility POIs for {place}"
        )

        print("=" * 60)


        result = get_city_pois(
            row,
            graphs[place],
            default_osm_key,
            retry_wait=120,
        )


        poi_results[
            place
        ] = result


        city_metadata_with_graphs_df.loc[
            index,
            "poi_count"
        ] = len(
            result["features"]
        )


        city_metadata_with_graphs_df.loc[
            index,
            "poi_status"
        ] = result[
            "download_status"
        ]


        if result["download_status"] == "downloaded":

            time.sleep(30)


    city_metadata_with_graphs_df[
        "poi_count"
    ] = (
        city_metadata_with_graphs_df[
            "poi_count"
        ].astype(int)
    )


    city_metadata_with_graphs_df
    return (poi_results,)


@app.cell
def _(copenhagen_place, poi_results):
    poi_results[
        copenhagen_place
    ]["poi_fig"]
    return


if __name__ == "__main__":
    app.run()
