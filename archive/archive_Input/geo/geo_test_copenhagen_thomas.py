import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    # Make sure the project root is available so this works:
    # from src.feather import FEATHER, FEATHER_new
    project_root = Path.cwd()

    if not (project_root / "src").exists():
        project_root = Path(__file__).resolve().parents[2]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import marimo as mo
    import osmnx as ox
    import networkx as nx
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from src.feather import FEATHER, FEATHER_new

    ox.settings.use_cache = True
    ox.settings.log_console = True
    return FEATHER, FEATHER_new, mo, np, nx, ox, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Copenhagen FEATHER hospital experiment

    This notebook uses the full Copenhagen walking graph.

    Settings:

    - Place: **Copenhagen Municipality, Denmark**
    - Amenity: **hospital**
    - `theta_max = 2.5`
    - `eval_points = 2`
    - `order = 1`
    - Visualization column: `cos_r1_theta0`

    FEATHER and FEATHER_new use exactly the same hospital-distance
    node feature.

    The difference is the transition matrix:

    - **FEATHER:** unweighted topology
    - **FEATHER_new:** inverse edge length weighting
    """)
    return


@app.cell
def _():
    place = "Copenhagen Municipality, Denmark"

    hospital_tags = {
        "amenity": ["hospital"],
    }

    theta_max = 2.5
    eval_points = 2
    order = 1

    visual_column = "cos_r1_theta0"
    return eval_points, hospital_tags, order, place, theta_max, visual_column


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load Copenhagen walking graph
    """)
    return


@app.cell
def _(nx, ox, place):
    # ==============================================================
    # Download Copenhagen walking graph
    # ==============================================================

    copenhagen_graph = ox.graph.graph_from_place(
        place,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )

    # Use an undirected graph for FEATHER.
    copenhagen_graph = ox.convert.to_undirected(
        copenhagen_graph
    )

    copenhagen_graph.remove_edges_from(
        nx.selfloop_edges(copenhagen_graph)
    )

    # Project only for plotting / spatial coordinates.
    # OSMnx edge "length" is already stored in metres.
    copenhagen_graph_projected = ox.project_graph(
        copenhagen_graph
    )

    copenhagen_nodes, copenhagen_edges = ox.graph_to_gdfs(
        copenhagen_graph_projected
    )

    print(
        "Nodes:",
        copenhagen_graph.number_of_nodes(),
    )

    print(
        "Edges:",
        copenhagen_graph.number_of_edges(),
    )
    return copenhagen_graph, copenhagen_graph_projected, copenhagen_nodes


@app.cell
def _(copenhagen_graph_projected, ox):
    # ==============================================================
    # Plot Copenhagen street network
    # ==============================================================

    copenhagen_fig, _ = ox.plot_graph(
        copenhagen_graph_projected,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.1,
        bgcolor="black",
        show=False,
        close=False,
    )

    copenhagen_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Load hospitals

    Only OSM features with `amenity=hospital` are used.
    """)
    return


@app.cell
def _(copenhagen_graph_projected, copenhagen_nodes, hospital_tags, ox, place):
    # ==============================================================
    # Get hospital locations
    # ==============================================================

    hospital_features = ox.features.features_from_place(
        place,
        hospital_tags,
    )

    hospital_points = hospital_features.to_crs(
        copenhagen_nodes.crs
    ).copy()

    # Hospitals can be polygons.
    # Convert every hospital geometry into one representative point.
    hospital_points["geometry"] = (
        hospital_points.geometry.representative_point()
    )

    # Find the closest street-network node to each hospital.
    hospital_node_ids = ox.distance.nearest_nodes(
        copenhagen_graph_projected,
        X=hospital_points.geometry.x.to_numpy(),
        Y=hospital_points.geometry.y.to_numpy(),
    )

    # Multiple hospital geometries can snap to the same graph node.
    hospital_node_ids = list(set(hospital_node_ids))

    print(
        "Hospital features:",
        len(hospital_features),
    )

    print(
        "Unique hospital network nodes:",
        len(hospital_node_ids),
    )
    return hospital_node_ids, hospital_points


@app.cell
def _(copenhagen_graph_projected, hospital_points, ox):
    # ==============================================================
    # Plot hospitals
    # ==============================================================

    hospital_fig, hospital_ax = ox.plot_graph(
        copenhagen_graph_projected,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.1,
        bgcolor="black",
        show=False,
        close=False,
    )

    hospital_points.plot(
        ax=hospital_ax,
        marker="*",
        markersize=45,
        color="red",
        zorder=3,
    )

    hospital_ax.set_title(
        "Hospitals in Copenhagen"
    )

    hospital_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Walking distance to nearest hospital

    The shortest-path calculation uses the original OSMnx graph.

    OSMnx's `length` edge attribute is already in metres, so the
    resulting feature is walking-network distance in metres.
    """)
    return


@app.cell
def _(copenhagen_graph, hospital_node_ids, np, nx, pd):
    # ==============================================================
    # Hospital distance feature
    # ==============================================================

    if len(hospital_node_ids) == 0:
        raise ValueError(
            "No hospitals were mapped to the Copenhagen street network."
        )

    # Start Dijkstra from every hospital at the same time.
    # Each node receives distance to its nearest hospital.
    hospital_distances = nx.multi_source_dijkstra_path_length(
        copenhagen_graph,
        hospital_node_ids,
        weight="length",
    )

    original_nodes = list(
        copenhagen_graph.nodes()
    )

    # retain_all=False should normally make every node reachable.
    # This fallback prevents KeyError if a node is unexpectedly absent.
    maximum_hospital_distance = max(
        hospital_distances.values()
    )

    hospital_X = np.array(
        [
            hospital_distances.get(
                node,
                maximum_hospital_distance,
            )
            for node in original_nodes
        ],
        dtype=float,
    )

    hospital_distance_df = pd.DataFrame(
        {
            "osmid": original_nodes,
            "nearest_hospital_m": hospital_X,
        }
    )

    print(
        "Minimum hospital distance:",
        round(float(hospital_X.min()), 3),
    )

    print(
        "Maximum hospital distance:",
        round(float(hospital_X.max()), 3),
    )

    print(
        "Mean hospital distance:",
        round(float(hospital_X.mean()), 3),
    )
    return hospital_X, hospital_distance_df, original_nodes


@app.cell
def _(hospital_distance_df):
    hospital_distance_df.head(20).round(3)
    return


@app.cell
def _(
    copenhagen_graph_projected,
    copenhagen_nodes,
    hospital_X,
    original_nodes,
    ox,
):
    # ==============================================================
    # Plot hospital distance feature
    # ==============================================================

    distance_nodes = copenhagen_nodes.loc[
        original_nodes
    ].copy()

    distance_nodes["nearest_hospital_m"] = hospital_X

    distance_fig, distance_ax = ox.plot_graph(
        copenhagen_graph_projected,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.05,
        bgcolor="black",
        show=False,
        close=False,
    )

    distance_nodes.plot(
        ax=distance_ax,
        column="nearest_hospital_m",
        cmap="plasma",
        markersize=0.5,
        alpha=0.9,
        legend=True,
        zorder=3,
        legend_kwds={
            "shrink": 0.5,
            "label": "Walking distance to nearest hospital (m)",
            "orientation": "vertical",
        },
    )

    distance_ax.set_title(
        "Copenhagen - distance to nearest hospital"
    )

    distance_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Relabel graph for FEATHER

    OSMnx uses large OpenStreetMap node IDs.

    The current FEATHER implementation expects:

    `0, 1, 2, ..., N-1`
    """)
    return


@app.cell
def _(copenhagen_graph, nx, original_nodes):
    # ==============================================================
    # Relabel OSM node IDs to 0..N-1
    # ==============================================================

    node_mapping = {
        osm_id: new_id
        for new_id, osm_id in enumerate(original_nodes)
    }

    feather_graph = nx.relabel_nodes(
        copenhagen_graph,
        node_mapping,
        copy=True,
    )

    feather_new_graph = nx.relabel_nodes(
        copenhagen_graph,
        node_mapping,
        copy=True,
    )
    return feather_graph, feather_new_graph


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Run FEATHER

    Original FEATHER uses unweighted topology.

    With `eval_points = 2` and `order = 1`,
    each node gets four output values:

    - `cos_r1_theta0`
    - `cos_r1_theta1`
    - `sin_r1_theta0`
    - `sin_r1_theta1`
    """)
    return


@app.cell
def _(
    FEATHER,
    eval_points,
    feather_graph,
    hospital_X,
    order,
    original_nodes,
    pd,
    theta_max,
):
    # ==============================================================
    # Run FEATHER
    # ==============================================================

    feather_model = FEATHER(
        theta_max=theta_max,
        eval_points=eval_points,
        order=order,
    )

    feather_model.fit(
        feather_graph,
        hospital_X,
    )

    feather_embedding = (
        feather_model.get_embedding()
    )

    embedding_columns = [
        "cos_r1_theta0",
        "cos_r1_theta1",
        "sin_r1_theta0",
        "sin_r1_theta1",
    ]

    feather_df = pd.DataFrame(
        feather_embedding,
        columns=embedding_columns,
    )

    feather_df.insert(
        0,
        "osmid",
        original_nodes,
    )

    feather_df.insert(
        1,
        "hospital_distance_m",
        hospital_X,
    )
    return embedding_columns, feather_df


@app.cell
def _(feather_df):
    feather_df.head(20).round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Run FEATHER_new

    FEATHER_new converts edge length internally to:

    `weight = 1 / length`

    so shorter streets receive greater transition probability.
    """)
    return


@app.cell
def _(
    FEATHER_new,
    embedding_columns,
    eval_points,
    feather_new_graph,
    hospital_X,
    order,
    original_nodes,
    pd,
    theta_max,
):
    # ==============================================================
    # Run FEATHER_new
    # ==============================================================

    feather_new_model = FEATHER_new(
        theta_max=theta_max,
        eval_points=eval_points,
        order=order,
    )

    feather_new_model.fit(
        feather_new_graph,
        hospital_X,
    )

    feather_new_embedding = (
        feather_new_model.get_embedding()
    )

    feather_new_df = pd.DataFrame(
        feather_new_embedding,
        columns=embedding_columns,
    )

    feather_new_df.insert(
        0,
        "osmid",
        original_nodes,
    )

    feather_new_df.insert(
        1,
        "hospital_distance_m",
        hospital_X,
    )
    return (feather_new_df,)


@app.cell
def _(feather_new_df):
    feather_new_df.head(20).round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Compare FEATHER and FEATHER_new
    """)
    return


@app.cell
def _(feather_df, feather_new_df, original_nodes, pd, visual_column):
    # ==============================================================
    # Compare selected output coordinate
    # ==============================================================

    comparison_df = pd.DataFrame(
        {
            "osmid": original_nodes,
            "FEATHER": feather_df[
                visual_column
            ].to_numpy(),
            "FEATHER_new": feather_new_df[
                visual_column
            ].to_numpy(),
        }
    )

    comparison_df["difference"] = (
        comparison_df["FEATHER_new"]
        - comparison_df["FEATHER"]
    )
    return (comparison_df,)


@app.cell
def _(comparison_df):
    comparison_df.head(20).round(6)
    return


@app.cell
def _(comparison_df):
    print(
        "Mean absolute difference:",
        round(
            float(
                comparison_df[
                    "difference"
                ].abs().mean()
            ),
            6,
        ),
    )

    print(
        "Maximum absolute difference:",
        round(
            float(
                comparison_df[
                    "difference"
                ].abs().max()
            ),
            6,
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FEATHER maps

    FEATHER and FEATHER_new use the same color range `[-1, 1]`.

    The difference map uses a symmetric range around zero.
    """)
    return


@app.cell
def _(
    copenhagen_graph_projected,
    copenhagen_nodes,
    feather_df,
    original_nodes,
    ox,
    visual_column,
):
    # ==============================================================
    # FEATHER map
    # ==============================================================

    feather_map_nodes = copenhagen_nodes.loc[
        original_nodes
    ].copy()

    feather_map_nodes[visual_column] = feather_df[
        visual_column
    ].to_numpy()

    feather_fig, feather_ax = ox.plot_graph(
        copenhagen_graph_projected,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.05,
        bgcolor="black",
        show=False,
        close=False,
    )

    feather_map_nodes.plot(
        ax=feather_ax,
        column=visual_column,
        cmap="plasma",
        markersize=0.5,
        alpha=0.9,
        legend=True,
        zorder=3,
        vmin=-1,
        vmax=1,
        legend_kwds={
            "shrink": 0.5,
            "label": f"FEATHER {visual_column}",
            "orientation": "vertical",
        },
    )

    feather_ax.set_title(
        f"Copenhagen - FEATHER - {visual_column}"
    )
    return (feather_fig,)


@app.cell
def _(feather_fig):
    feather_fig
    return


@app.cell
def _(
    copenhagen_graph_projected,
    copenhagen_nodes,
    feather_new_df,
    original_nodes,
    ox,
    visual_column,
):
    # ==============================================================
    # FEATHER_new map
    # ==============================================================

    feather_new_map_nodes = copenhagen_nodes.loc[
        original_nodes
    ].copy()

    feather_new_map_nodes[visual_column] = (
        feather_new_df[
            visual_column
        ].to_numpy()
    )

    feather_new_fig, feather_new_ax = ox.plot_graph(
        copenhagen_graph_projected,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.05,
        bgcolor="black",
        show=False,
        close=False,
    )

    feather_new_map_nodes.plot(
        ax=feather_new_ax,
        column=visual_column,
        cmap="plasma",
        markersize=0.5,
        alpha=0.9,
        legend=True,
        zorder=3,
        vmin=-1,
        vmax=1,
        legend_kwds={
            "shrink": 0.5,
            "label": f"FEATHER_new {visual_column}",
            "orientation": "vertical",
        },
    )

    feather_new_ax.set_title(
        f"Copenhagen - FEATHER_new - {visual_column}"
    )
    return (feather_new_fig,)


@app.cell
def _(feather_new_fig):
    feather_new_fig
    return


@app.cell
def _(
    comparison_df,
    copenhagen_graph_projected,
    copenhagen_nodes,
    np,
    original_nodes,
    ox,
    plt,
):
    # ==============================================================
    # High-resolution difference map: FEATHER_new - FEATHER
    # ==============================================================

    # Create a GeoDataFrame with the same node order used by FEATHER
    difference_map_nodes = copenhagen_nodes.loc[
        original_nodes
    ].copy()

    # Add the FEATHER_new - FEATHER difference
    difference_map_nodes["difference"] = (
        comparison_df["difference"].to_numpy()
    )

    # ==============================================================
    # Use the 99th percentile for the color scale
    # ==============================================================

    # This makes smaller differences easier to see by preventing
    # a few extreme values from controlling the whole color scale.
    difference_limit = np.percentile(
        np.abs(comparison_df["difference"]),
        99,
    )

    # Prevent problems if all values happen to be zero
    if difference_limit == 0:
        difference_limit = 1e-9


    # ==============================================================
    # Create a detailed figure that Marimo can still display
    # ==============================================================

    difference_fig, difference_ax = plt.subplots(
        figsize=(18, 18),
        dpi=120,
    )


    # ==============================================================
    # Plot the Copenhagen street network
    # ==============================================================

    ox.plot_graph(
        copenhagen_graph_projected,
        ax=difference_ax,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.1,
        edge_alpha=0.35,
        bgcolor="black",
        show=False,
        close=False,
    )


    # ==============================================================
    # Plot node differences
    # ==============================================================

    difference_map_nodes.plot(
        ax=difference_ax,
        column="difference",
        cmap="coolwarm",

        # Increase this if you want nodes to be more visible
        markersize=2,

        alpha=0.95,
        legend=True,
        zorder=3,

        # Symmetric color range around zero
        vmin=-difference_limit,
        vmax=difference_limit,

        legend_kwds={
            "shrink": 0.5,
            "label": "FEATHER_new - FEATHER",
            "orientation": "vertical",
        },
    )


    # ==============================================================
    # Title and layout
    # ==============================================================

    difference_ax.set_title(
        "Copenhagen - FEATHER_new minus FEATHER",
        fontsize=20,
    )

    difference_fig.tight_layout()


    # ==============================================================
    # Optional: save a much higher-resolution version to disk
    # ==============================================================

    difference_fig.savefig(
        "copenhagen_feather_difference_highres.png",
        dpi=300,
        bbox_inches="tight",
    )


    # ==============================================================
    # Display in Marimo
    # ==============================================================

    difference_fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
