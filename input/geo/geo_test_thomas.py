import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import osmnx as ox
    import networkx as nx
    import pandas as pd
    import pandana as pdna
    import matplotlib.pyplot as plt
    import numpy as np

    from src.feather import FEATHER, FEATHER_new



    #https://github.com/gboeing/osmnx-examples/blob/main/notebooks/03-graph-place-queries.ipynb
    return FEATHER, FEATHER_new, np, nx, ox, pd, pdna


@app.cell
def _(ox):
    ox.settings.log_console = True
    ox.__version__
    return


@app.cell
def _(ox):
    #place = "Copenhagen Municipality, Denmark"
    #place = "Varde, Denmark"
    place = "Aarhus Municipality, Denmark"

    copenhagen = ox.geocoder.geocode_to_gdf(place)

    copenhagen_G = ox.graph.graph_from_place(
        place,
        network_type="walk",
        simplify=True, # simplify for less nodes and edges
        retain_all=False, # discard disconnected vertice 
    )

    copenhagen_G = ox.convert.to_undirected(copenhagen_G)

    """
    Since when calculating distance to ammenities we also want to includes amenities which are not on the graph on the relevant city, should we use the buffer parameter to get the city + 500m fx.?

    # or get the walking network within a 500 meter buffer of piedmont
    gdf = ox.geocoder.geocode_to_gdf("Piedmont, CA, USA")
    polygon = ox.utils_geo.buffer_geometry(gdf.iloc[0]["geometry"], 500)
    G = ox.graph.graph_from_polygon(polygon, network_type="walk")
    """
    return copenhagen_G, place


@app.cell
def _(copenhagen_G, ox):
    ox.plot_graph(
        copenhagen_G,
        node_size=0.75,
        edge_linewidth=0.5,
        bgcolor="white",
    )
    return


@app.cell
def _(copenhagen_G, ox, place):
    tags_hospital = {
        "amenity": ["hospital"],
    }

    hospital_features = ox.features.features_from_place(
        place,
        tags_hospital,
    )

    hospital_fig, hospital_ax = ox.plot.plot_graph(
        copenhagen_G,
        show=False,
        close=False,
        bgcolor="white",
        edge_linewidth=0.5,
        node_size=0.75,
    )

    hospital_features.plot(
        column="amenity",
        categorical=True,
        legend=True,
        ax=hospital_ax,
        cmap="Set3",
    )

    hospital_fig
    return (hospital_features,)


@app.cell
def _(copenhagen_G, hospital_features, ox, pdna):
    # Project Copenhagen graph to a CRS that uses meters
    hospital_projected_graph = ox.project_graph(copenhagen_G)

    # Convert OSMnx graph to GeoDataFrames
    hospital_nodes, hospital_edges = ox.graph_to_gdfs(
        hospital_projected_graph
    )

    # Pandana needs u and v as normal columns
    hospital_edges = hospital_edges.reset_index()

    # Create Pandana network
    hospital_network = pdna.Network(
        hospital_nodes.geometry.x,
        hospital_nodes.geometry.y,
        hospital_edges["u"],
        hospital_edges["v"],
        hospital_edges[["length"]],
    )

    # Use the hospital features downloaded in the previous cell
    hospital_pois = hospital_features.to_crs(hospital_nodes.crs)

    # Convert all hospital geometries to points
    hospital_pois = hospital_pois.copy()
    hospital_pois["geometry"] = hospital_pois.geometry.centroid

    # Maximum search distance in meters
    hospital_search_distance = 10000

    # Register hospitals as POIs in Pandana
    hospital_network.set_pois(
        category="hospital",
        maxdist=hospital_search_distance,
        maxitems=100,
        x_col=hospital_pois.geometry.x,
        y_col=hospital_pois.geometry.y,
    )

    # Calculate distance from every network node to the nearest hospital
    nearest_hospitals = hospital_network.nearest_pois(
        distance=hospital_search_distance,
        category="hospital",
        num_pois=1,
    )

    # Add the nearest-hospital distance to the node GeoDataFrame
    hospital_nodes = hospital_nodes.copy()
    hospital_nodes["nearest_hospital"] = nearest_hospitals.iloc[:, 0]

    # Plot the street network
    hospital_distance_fig, hospital_distance_ax = ox.plot_graph(
        hospital_projected_graph,
        node_size=0,
        edge_color="#afdffe",
        edge_linewidth=0.5,
        bgcolor="#1a1a1a",
        show=False,
        close=False,
    )

    # Plot nodes coloured by distance to nearest hospital
    hospital_nodes.plot(
        ax=hospital_distance_ax,
        column="nearest_hospital",
        cmap="plasma",
        markersize=0.5,
        alpha=0.8,
        legend=True,
        legend_kwds={
            "shrink": 0.5,
            "label": "Distance to closest hospital (m)",
            "orientation": "vertical",
        },
    )

    hospital_distance_fig
    return hospital_nodes, hospital_projected_graph, hospital_search_distance


@app.cell
def _(
    FEATHER,
    copenhagen_G,
    hospital_nodes,
    hospital_projected_graph,
    hospital_search_distance,
    np,
    nx,
    ox,
    pd,
):
    # ==========================================
    # FEATHER - hospital accessibility embedding
    # ==========================================

    # Work on a copy of the Copenhagen graph
    feather_graph = copenhagen_G.copy()

    # OSMnx uses OpenStreetMap node IDs.
    # FEATHER expects nodes numbered 0, 1, 2, ..., N-1.
    original_nodes = list(feather_graph.nodes())

    node_mapping = {
        old_node: new_node
        for new_node, old_node in enumerate(original_nodes)
    }

    feather_graph = nx.relabel_nodes(
        feather_graph,
        node_mapping,
        copy=True,
    )

    # ==========================================
    # Create hospital feature vector
    # ==========================================

    # Each value is the network distance from that node
    # to the nearest hospital.
    hospital_X_feather = hospital_nodes.loc[
        original_nodes,
        "nearest_hospital"
    ].to_numpy(dtype=float)

    # Replace NaN values with the maximum Pandana search distance.
    hospital_X_feather = np.nan_to_num(
        hospital_X_feather,
        nan=hospital_search_distance,
    )

    # ==========================================
    # Run original FEATHER
    # ==========================================

    feather_model = FEATHER(
        theta_max=2.5,
        eval_points=25,
        order=5,
    )

    feather_model.fit(
        feather_graph,
        hospital_X_feather,
    )

    feather_embedding = feather_model.get_embedding()

    # ==========================================
    # Create meaningful DataFrame column names
    # ==========================================

    feather_columns = []

    for feather_r in range(1, feather_model.order + 1):

        # Cosine values
        for theta_index in range(feather_model.eval_points):
            feather_columns.append(
                f"cos_r{feather_r}_theta{theta_index}"
            )

        # Sine values
        for theta_index in range(feather_model.eval_points):
            feather_columns.append(
                f"sin_r{feather_r}_theta{theta_index}"
            )

    # Store full FEATHER embedding
    feather_df = pd.DataFrame(
        feather_embedding,
        columns=feather_columns,
    )

    # Keep original OpenStreetMap node IDs
    feather_df["osmid"] = original_nodes

    # ==========================================
    # Choose random-walk order to visualize
    # ==========================================

    feather_map_r = 5

    feather_cos_columns = [
        f"cos_r{feather_map_r}_theta{i}"
        for i in range(feather_model.eval_points)
    ]

    feather_sin_columns = [
        f"sin_r{feather_map_r}_theta{i}"
        for i in range(feather_model.eval_points)
    ]

    # ==========================================
    # Create one visualization value per node
    # ==========================================

    # Mean cosine value across all theta evaluation points
    feather_df[
        f"cos_mean_r{feather_map_r}"
    ] = feather_df[
        feather_cos_columns
    ].mean(axis=1)

    # Mean sine value across all theta evaluation points
    feather_df[
        f"sin_mean_r{feather_map_r}"
    ] = feather_df[
        feather_sin_columns
    ].mean(axis=1)

    # ==========================================
    # Join FEATHER results back onto map nodes
    # ==========================================

    feather_map_nodes = hospital_nodes.loc[
        original_nodes
    ].copy()

    feather_map_nodes[
        f"cos_mean_r{feather_map_r}"
    ] = feather_df[
        f"cos_mean_r{feather_map_r}"
    ].to_numpy()

    feather_map_nodes[
        f"sin_mean_r{feather_map_r}"
    ] = feather_df[
        f"sin_mean_r{feather_map_r}"
    ].to_numpy()

    # ==========================================
    # Plot FEATHER cosine values
    # ==========================================

    # IMPORTANT:
    # hospital_nodes came from hospital_projected_graph,
    # so use the projected graph here too.
    feather_fig, feather_ax = ox.plot_graph(
        hospital_projected_graph,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.05,
        bgcolor="black",
        show=False,
        close=False,
    )

    # Plot FEATHER values as visible coloured nodes
    feather_map_nodes.plot(
        ax=feather_ax,
        column=f"cos_mean_r{feather_map_r}",
        cmap="plasma",
        markersize=0.5,
        alpha=0.9,
        legend=True,
        zorder=3,

        # Cosine values always lie between -1 and 1.
        # Using a fixed range also makes later FEATHER vs FEATHER_new
        # comparisons visually fair.
        vmin=-1,
        vmax=1,

        legend_kwds={
            "shrink": 0.5,
            "label": f"FEATHER hospital cosine mean, r={feather_map_r}",
            "orientation": "vertical",
        },
    )

    # ==========================================
    # Display in Marimo
    # ==========================================

    feather_df
    return feather_df, feather_fig, original_nodes


@app.cell
def _(feather_fig):
    feather_fig
    return


@app.cell
def _(
    FEATHER_new,
    copenhagen_G,
    hospital_nodes,
    hospital_projected_graph,
    hospital_search_distance,
    np,
    nx,
    ox,
    pd,
):
    # ==============================================
    # FEATHER_new - weighted hospital accessibility
    # ==============================================

    # Work on a copy of the Copenhagen graph
    feather_new_graph = copenhagen_G.copy()

    # OSMnx uses OpenStreetMap node IDs.
    # FEATHER_new expects nodes numbered 0, 1, 2, ..., N-1.
    original_nodes_new = list(feather_new_graph.nodes())

    node_mapping_new = {
        old_node: new_node
        for new_node, old_node in enumerate(original_nodes_new)
    }

    feather_new_graph = nx.relabel_nodes(
        feather_new_graph,
        node_mapping_new,
        copy=True,
    )

    # ==============================================
    # Use OSMnx edge length as FEATHER_new weight
    # ==============================================

    # OSMnx stores physical edge length in the "length" attribute.
    # FEATHER_new looks for an edge attribute called "weight".
    for _, _, edge_data in feather_new_graph.edges(data=True):
        edge_data["weight"] = edge_data.get("length", 1.0)

    # ==============================================
    # Create hospital feature vector
    # ==============================================

    # Each value is the network distance from that node
    # to the nearest hospital.
    hospital_X_feather_new = hospital_nodes.loc[
        original_nodes_new,
        "nearest_hospital"
    ].to_numpy(dtype=float)

    # Replace NaN values with the maximum Pandana search distance.
    hospital_X_feather_new = np.nan_to_num(
        hospital_X_feather_new,
        nan=hospital_search_distance,
    )

    # ==============================================
    # Run weighted FEATHER_new
    # ==============================================

    feather_new_model = FEATHER_new(
        theta_max=2.5,
        eval_points=25,
        order=5,
    )

    feather_new_model.fit(
        feather_new_graph,
        hospital_X_feather_new,
    )

    feather_new_embedding = feather_new_model.get_embedding()

    # ==============================================
    # Create meaningful DataFrame column names
    # ==============================================

    feather_new_columns = []

    for feather_new_r in range(1, feather_new_model.order + 1):

        # Cosine values
        for theta_index_new in range(feather_new_model.eval_points):
            feather_new_columns.append(
                f"cos_r{feather_new_r}_theta{theta_index_new}"
            )

        # Sine values
        for theta_index_new in range(feather_new_model.eval_points):
            feather_new_columns.append(
                f"sin_r{feather_new_r}_theta{theta_index_new}"
            )

    # Store full FEATHER_new embedding
    feather_new_df = pd.DataFrame(
        feather_new_embedding,
        columns=feather_new_columns,
    )

    # Keep original OpenStreetMap node IDs
    feather_new_df["osmid"] = original_nodes_new

    # ==============================================
    # Choose random-walk order to visualize
    # ==============================================

    feather_new_map_r = 5

    feather_new_cos_columns = [
        f"cos_r{feather_new_map_r}_theta{i}"
        for i in range(feather_new_model.eval_points)
    ]

    feather_new_sin_columns = [
        f"sin_r{feather_new_map_r}_theta{i}"
        for i in range(feather_new_model.eval_points)
    ]

    # ==============================================
    # Create one visualization value per node
    # ==============================================

    # Mean cosine value across all theta evaluation points
    feather_new_df[
        f"cos_mean_r{feather_new_map_r}"
    ] = feather_new_df[
        feather_new_cos_columns
    ].mean(axis=1)

    # Mean sine value across all theta evaluation points
    feather_new_df[
        f"sin_mean_r{feather_new_map_r}"
    ] = feather_new_df[
        feather_new_sin_columns
    ].mean(axis=1)

    # ==============================================
    # Join FEATHER_new results back onto map nodes
    # ==============================================

    feather_new_map_nodes = hospital_nodes.loc[
        original_nodes_new
    ].copy()

    feather_new_map_nodes[
        f"cos_mean_r{feather_new_map_r}"
    ] = feather_new_df[
        f"cos_mean_r{feather_new_map_r}"
    ].to_numpy()

    feather_new_map_nodes[
        f"sin_mean_r{feather_new_map_r}"
    ] = feather_new_df[
        f"sin_mean_r{feather_new_map_r}"
    ].to_numpy()

    # ==============================================
    # Plot FEATHER_new cosine values
    # ==============================================

    # IMPORTANT:
    # hospital_nodes came from hospital_projected_graph,
    # so use the projected graph here too.
    feather_new_fig, feather_new_ax = ox.plot_graph(
        hospital_projected_graph,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.05,
        bgcolor="black",
        show=False,
        close=False,
    )

    # Plot FEATHER_new values as visible coloured nodes
    feather_new_map_nodes.plot(
        ax=feather_new_ax,
        column=f"cos_mean_r{feather_new_map_r}",
        cmap="plasma",
        markersize=0.5,
        alpha=0.9,
        legend=True,
        zorder=3,

        # Cosine values always lie between -1 and 1.
        # Using the same fixed range as FEATHER makes
        # the two maps directly comparable.
        vmin=-1,
        vmax=1,

        legend_kwds={
            "shrink": 0.5,
            "label": f"FEATHER_new hospital cosine mean, r={feather_new_map_r}",
            "orientation": "vertical",
        },
    )

    # ==============================================
    # Display in Marimo
    # ==============================================

    feather_new_df
    return feather_new_df, feather_new_fig


@app.cell
def _(feather_new_fig):
    feather_new_fig
    return


@app.cell
def _(
    feather_df,
    feather_new_df,
    hospital_nodes,
    hospital_projected_graph,
    original_nodes,
    ox,
):
    # ==============================================
    # Difference map: FEATHER_new - FEATHER
    # ==============================================

    # Use the same node order as the FEATHER results
    comparison_nodes = hospital_nodes.loc[
        original_nodes
    ].copy()

    # Calculate the difference between FEATHER_new and FEATHER
    comparison_nodes["FEATHER_difference"] = (
        feather_new_df["cos_mean_r5"].to_numpy()
        - feather_df["cos_mean_r5"].to_numpy()
    )

    # ==============================================
    # Plot difference values
    # ==============================================

    difference_fig, difference_ax = ox.plot_graph(
        hospital_projected_graph,
        node_size=0,
        edge_color="lightgray",
        edge_linewidth=0.05,
        bgcolor="black",
        show=False,
        close=False,
    )

    comparison_nodes.plot(
        ax=difference_ax,
        column="FEATHER_difference",
        cmap="coolwarm",
        markersize=0.5,
        alpha=0.9,
        legend=True,
        zorder=3,
        legend_kwds={
            "shrink": 0.5,
            "label": "FEATHER_new - FEATHER",
            "orientation": "vertical",
        },
    )

    # ==============================================
    # Display in Marimo
    # ==============================================

    difference_fig
    return


if __name__ == "__main__":
    app.run()
