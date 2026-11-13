import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
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
    # Copenhagen 5-node FEATHER demo

    This notebook intentionally keeps the example tiny:

    - Place: **Copenhagen Municipality, Denmark**
    - Amenity: **hospital**
    - Hospitals used: **1**
    - Street nodes used by FEATHER: **5 connected nodes**
    - `theta_max = 2.5`
    - `eval_points = 2`
    - `order = 1`

    The goal is to make the difference between **FEATHER** and **FEATHER_new** easy to inspect.
    """)
    return


@app.cell
def _():
    place = "Copenhagen Municipality, Denmark"
    hospital_tags = {"amenity": ["hospital"]}

    theta_max = 2.5
    eval_points = 2
    order = 1

    use_five_node_demo = True  # Change to False to run the full Copenhagen graph.

    # We visualize the first cosine coordinate.
    # With eval_points=2, theta values are [0.01, 2.5].
    visual_column = "cos_r1_theta0"
    return (
        eval_points,
        hospital_tags,
        order,
        place,
        theta_max,
        use_five_node_demo,
        visual_column,
    )


@app.cell
def _(hospital_tags, nx, ox, pd, place, plt, use_five_node_demo):
    # ==============================================================
    # 1. Download Copenhagen walking graph and hospital locations
    # ==============================================================

    city_graph = ox.graph.graph_from_place(
        place,
        network_type="walk",
        simplify=True,
        retain_all=False,
    )
    city_graph = ox.convert.to_undirected(city_graph)
    city_graph.remove_edges_from(nx.selfloop_edges(city_graph))

    city_graph_projected = ox.project_graph(city_graph)
    city_nodes_projected, _city_edges_projected = ox.graph_to_gdfs(
        city_graph_projected
    )

    hospital_features = ox.features.features_from_place(
        place,
        hospital_tags,
    )

    # Keep exactly ONE real hospital.
    hospital_points = hospital_features.to_crs(city_nodes_projected.crs).copy()
    hospital_points["geometry"] = hospital_points.geometry.representative_point()
    one_hospital = hospital_points.iloc[[0]].copy()

    hospital_x = float(one_hospital.geometry.x.iloc[0])
    hospital_y = float(one_hospital.geometry.y.iloc[0])

    if "name" in one_hospital.columns and pd.notna(one_hospital["name"].iloc[0]):
        selected_hospital_name = str(one_hospital["name"].iloc[0])
    else:
        selected_hospital_name = "Selected hospital"

    # Find the street-network node closest to that one hospital.
    hospital_osm_node = ox.distance.nearest_nodes(
        city_graph_projected,
        X=hospital_x,
        Y=hospital_y,
    )

    # ==============================================================
    # 2. ONLY 5-NODE FILTER
    # ==============================================================
    # This is the part to remove/change later if you want the whole graph.
    # BFS guarantees that the selected nodes are connected.

    if use_five_node_demo:
        selected_osm_nodes = list(
            nx.bfs_tree(city_graph, hospital_osm_node).nodes()
        )[:5]

        # For the tiny teaching graph, collapse parallel OSM edges so the
        # picture is easy to read.
        demo_graph_osm = nx.Graph(
            city_graph.subgraph(selected_osm_nodes).copy()
        )
    else:
        # Full graph mode: keep the original OSMnx graph structure.
        selected_osm_nodes = list(city_graph.nodes())
        demo_graph_osm = city_graph.copy()

    # ==============================================================
    # 3. Relabel the 5 real OSM nodes as 0,1,2,3,4 for FEATHER
    # ==============================================================

    osm_to_demo = {
        osm_id: demo_id
        for demo_id, osm_id in enumerate(selected_osm_nodes)
    }

    demo_graph = nx.relabel_nodes(
        demo_graph_osm,
        osm_to_demo,
        copy=True,
    )

    demo_to_osm = {
        demo_id: osm_id
        for osm_id, demo_id in osm_to_demo.items()
    }

    # Real projected coordinates, but with simple 0..4 node labels.
    demo_pos = {
        osm_to_demo[osm_id]: (
            float(city_nodes_projected.loc[osm_id].geometry.x),
            float(city_nodes_projected.loc[osm_id].geometry.y),
        )
        for osm_id in selected_osm_nodes
    }

    # ==============================================================
    # 4. Hospital distance feature
    # ==============================================================
    # Distances are calculated on the FULL Copenhagen graph, even though
    # FEATHER itself is only run on the 5-node demonstration graph.

    # Calculate walking distance from the selected hospital node
    # to all reachable nodes in the full Copenhagen graph.
    full_hospital_distances = nx.single_source_dijkstra_path_length(
        city_graph,
        hospital_osm_node,
        weight="length",
    )

    # Extract the hospital distance for our 5 selected nodes.
    demo_hospital_distance_m = {
        osm_to_demo[osm_id]: float(full_hospital_distances[osm_id])
        for osm_id in selected_osm_nodes
    }

    demo_node_count = demo_graph.number_of_nodes()

    demo_node_df = pd.DataFrame(
        {
            "node": range(demo_node_count),
            "osmid": [demo_to_osm[i] for i in range(demo_node_count)],
            "hospital_distance_m": [
                demo_hospital_distance_m[i] for i in range(demo_node_count)
            ],
        }
    )

    # ==============================================================
    # 5. Plot the raw 5-node graph
    # ==============================================================

    demo_fig, demo_ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_edges(
        demo_graph,
        demo_pos,
        ax=demo_ax,
        width=2,
        edge_color="gray",
    )
    nx.draw_networkx_nodes(
        demo_graph,
        demo_pos,
        ax=demo_ax,
        node_size=900,
        node_color="lightblue",
        edgecolors="black",
    )

    if demo_node_count <= 20:
        raw_node_labels = {
            node: f"{node}\n{demo_hospital_distance_m[node]:.1f} m"
            for node in demo_graph.nodes()
        }
        nx.draw_networkx_labels(
            demo_graph,
            demo_pos,
            labels=raw_node_labels,
            ax=demo_ax,
            font_size=9,
        )

        raw_edge_labels = {
            (u, v): f"{float(data.get('length', 1.0)):.1f} m"
            for u, v, data in demo_graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            demo_graph,
            demo_pos,
            edge_labels=raw_edge_labels,
            ax=demo_ax,
            font_size=8,
        )

    demo_ax.scatter(
        [hospital_x],
        [hospital_y],
        marker="*",
        s=350,
        c="red",
        edgecolors="black",
        zorder=5,
        label=selected_hospital_name,
    )
    demo_ax.legend(loc="best")
    demo_ax.set_title(
        f"{demo_node_count} connected Copenhagen nodes\n"
        "node label includes distance to the selected hospital"
    )
    demo_ax.set_axis_off()
    return (
        demo_fig,
        demo_graph,
        demo_hospital_distance_m,
        demo_node_count,
        demo_node_df,
        demo_pos,
        hospital_x,
        hospital_y,
        selected_hospital_name,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The 5-node teaching graph

    Each node label shows:

    `node number`
    `distance to the single selected hospital`

    Each edge label shows the real OSMnx edge length in metres.
    """)
    return


@app.cell
def _(demo_fig):
    demo_fig
    return


@app.cell
def _(demo_node_df):
    demo_node_df.round(3)
    return


@app.cell
def _(
    FEATHER,
    FEATHER_new,
    demo_graph,
    demo_hospital_distance_m,
    demo_node_count,
    eval_points,
    np,
    nx,
    order,
    pd,
    theta_max,
):
    # ==============================================================
    # 6. Prepare the SAME 5-node feature vector for both algorithms
    # ==============================================================

    X_hospital = np.array(
        [demo_hospital_distance_m[node] for node in range(demo_node_count)],
        dtype=float,
    )

    theta_values = np.linspace(0.01, theta_max, eval_points)

    embedding_columns = []
    for r in range(1, order + 1):
        for theta_index in range(eval_points):
            embedding_columns.append(f"cos_r{r}_theta{theta_index}")
        for theta_index in range(eval_points):
            embedding_columns.append(f"sin_r{r}_theta{theta_index}")

    # ==============================================================
    # 7. FEATHER - unweighted topology
    # ==============================================================

    feather_graph = demo_graph.copy()

    # Make sure original FEATHER does NOT see a weight attribute.
    for _, _, feather_edge_data in feather_graph.edges(data=True):
        feather_edge_data.pop("weight", None)

    feather_model = FEATHER(
        theta_max=theta_max,
        eval_points=eval_points,
        order=order,
    )

    feather_A = nx.adjacency_matrix(
        feather_graph,
        nodelist=range(demo_node_count),
        weight=None,
    )
    feather_D_inverse = feather_model._create_D_inverse(feather_graph)
    feather_A_tilde = feather_model._create_A_tilde(feather_graph)

    feather_model.fit(feather_graph, X_hospital)
    feather_embedding = feather_model.get_embedding()

    feather_df = pd.DataFrame(
        feather_embedding,
        columns=embedding_columns,
    )
    feather_df.insert(0, "node", range(demo_node_count))
    feather_df.insert(1, "hospital_distance_m", X_hospital)

    # ==============================================================
    # 8. FEATHER_new - inverse edge length weighting
    # ==============================================================

    feather_new_graph = demo_graph.copy()

    # IMPORTANT:
    # Your current FEATHER_new creates the weights on an internal graph copy G,
    # but then builds A from `graph` instead of `G`.
    # Setting the weights here makes the current class behave as intended.
    for _, _, weighted_edge_data in feather_new_graph.edges(data=True):
        edge_length = max(float(weighted_edge_data.get("length", 1.0)), 1.0)
        weighted_edge_data["weight"] = 1.0 / edge_length

    feather_new_model = FEATHER_new(
        theta_max=theta_max,
        eval_points=eval_points,
        order=order,
    )

    feather_new_A = nx.adjacency_matrix(
        feather_new_graph,
        nodelist=range(demo_node_count),
        weight="weight",
    )
    feather_new_D_inverse = feather_new_model._create_D_inverse(
        feather_new_graph
    )
    feather_new_A_tilde = feather_new_model._create_A_tilde(
        feather_new_graph
    )

    feather_new_model.fit(feather_new_graph, X_hospital)
    feather_new_embedding = feather_new_model.get_embedding()

    feather_new_df = pd.DataFrame(
        feather_new_embedding,
        columns=embedding_columns,
    )
    feather_new_df.insert(0, "node", range(demo_node_count))
    feather_new_df.insert(1, "hospital_distance_m", X_hospital)

    # ==============================================================
    # 9. One compact comparison table
    # ==============================================================

    comparison_df = pd.DataFrame(
        {
            "node": range(demo_node_count),
            "hospital_distance_m": X_hospital,
        }
    )

    for output_column in embedding_columns:
        comparison_df[f"FEATHER_{output_column}"] = feather_df[output_column]
        comparison_df[f"FEATHER_new_{output_column}"] = feather_new_df[output_column]
        comparison_df[f"difference_{output_column}"] = (
            feather_new_df[output_column] - feather_df[output_column]
        )

    # Easy-to-read matrix DataFrames.
    matrix_labels = [f"node_{i}" for i in range(demo_node_count)]

    feather_A_df = pd.DataFrame(
        feather_A.toarray(),
        index=matrix_labels,
        columns=matrix_labels,
    )
    feather_A_tilde_df = pd.DataFrame(
        feather_A_tilde.toarray(),
        index=matrix_labels,
        columns=matrix_labels,
    )
    feather_new_A_df = pd.DataFrame(
        feather_new_A.toarray(),
        index=matrix_labels,
        columns=matrix_labels,
    )
    feather_new_A_tilde_df = pd.DataFrame(
        feather_new_A_tilde.toarray(),
        index=matrix_labels,
        columns=matrix_labels,
    )
    return (
        comparison_df,
        feather_A_df,
        feather_A_tilde_df,
        feather_df,
        feather_new_A_df,
        feather_new_A_tilde_df,
        feather_new_df,
        theta_values,
    )


@app.cell(hide_code=True)
def _(mo, theta_values):
    mo.md(f"""
    ## FEATHER settings

    `theta = {theta_values}`  
    `order = 1`

    With `eval_points = 2` and `order = 1`, each node receives only four FEATHER outputs:

    - `cos_r1_theta0`
    - `cos_r1_theta1`
    - `sin_r1_theta0`
    - `sin_r1_theta1`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Transition matrices

    The first pair is ordinary FEATHER.
    The second pair is FEATHER_new using `weight = 1 / edge_length`.
    """)
    return


@app.cell
def _(feather_A_df):
    print("FEATHER A")
    feather_A_df.round(4)
    return


@app.cell
def _(feather_A_tilde_df):
    print("FEATHER A_tilde = D^-1 A")
    feather_A_tilde_df.round(4)
    return


@app.cell
def _(feather_new_A_df):
    print("FEATHER_new weighted A")
    feather_new_A_df.round(6)
    return


@app.cell
def _(feather_new_A_tilde_df):
    print("FEATHER_new A_tilde = D^-1 A")
    feather_new_A_tilde_df.round(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding outputs
    """)
    return


@app.cell
def _(feather_df):
    feather_df.round(4)
    return


@app.cell
def _(feather_new_df):
    feather_new_df.round(4)
    return


@app.cell
def _(comparison_df):
    comparison_df.round(4)
    return


@app.cell
def _(
    demo_graph,
    demo_node_count,
    demo_pos,
    feather_df,
    feather_new_df,
    hospital_x,
    hospital_y,
    nx,
    plt,
    selected_hospital_name,
    visual_column,
):
    # ==============================================================
    # 10. Three directly comparable 5-node visualizations
    # ==============================================================

    feather_values = feather_df.set_index("node")[visual_column]
    feather_new_values = feather_new_df.set_index("node")[visual_column]
    difference_values = feather_new_values - feather_values

    edge_length_labels = {
        (u, v): f"{float(data.get('length', 1.0)):.1f}m"
        for u, v, data in demo_graph.edges(data=True)
    }

    def draw_result_graph(values, title, colorbar_label, cmap, vmin, vmax):
        fig, ax = plt.subplots(figsize=(8, 6))

        nx.draw_networkx_edges(
            demo_graph,
            demo_pos,
            ax=ax,
            width=2,
            edge_color="gray",
        )

        node_artist = nx.draw_networkx_nodes(
            demo_graph,
            demo_pos,
            ax=ax,
            node_size=1100,
            node_color=[values[node] for node in demo_graph.nodes()],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="black",
        )

        if demo_node_count <= 20:
            value_labels = {
                node: f"{node}\n{values[node]:.4f}"
                for node in demo_graph.nodes()
            }
            nx.draw_networkx_labels(
                demo_graph,
                demo_pos,
                labels=value_labels,
                ax=ax,
                font_size=9,
            )

            nx.draw_networkx_edge_labels(
                demo_graph,
                demo_pos,
                edge_labels=edge_length_labels,
                ax=ax,
                font_size=8,
            )

        ax.scatter(
            [hospital_x],
            [hospital_y],
            marker="*",
            s=350,
            c="red",
            edgecolors="black",
            zorder=5,
            label=selected_hospital_name,
        )

        colorbar = fig.colorbar(node_artist, ax=ax, shrink=0.75)
        colorbar.set_label(colorbar_label)

        ax.legend(loc="best")
        ax.set_title(title)
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    feather_result_fig = draw_result_graph(
        feather_values,
        f"FEATHER - {visual_column}",
        visual_column,
        plt.cm.plasma,
        -1,
        1,
    )

    feather_new_result_fig = draw_result_graph(
        feather_new_values,
        f"FEATHER_new - {visual_column}",
        visual_column,
        plt.cm.plasma,
        -1,
        1,
    )

    difference_limit = max(
        float(np_abs)
        for np_abs in abs(difference_values).to_numpy()
    )
    if difference_limit == 0:
        difference_limit = 1e-9

    difference_result_fig = draw_result_graph(
        difference_values,
        f"FEATHER_new - FEATHER - {visual_column}",
        "difference",
        plt.cm.coolwarm,
        -difference_limit,
        difference_limit,
    )
    return difference_result_fig, feather_new_result_fig, feather_result_fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FEATHER visualization

    Node labels contain the FEATHER output value being visualized.
    """)
    return


@app.cell
def _(feather_result_fig):
    feather_result_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FEATHER_new visualization

    Same nodes, same hospital feature, same theta and order.
    Only the random-walk transition probabilities are changed by inverse edge-length weighting.
    """)
    return


@app.cell
def _(feather_new_result_fig):
    feather_new_result_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Difference: FEATHER_new - FEATHER

    A value of zero means both methods produced the same value for that node.
    Non-zero values show exactly where weighting the edges changed the output.
    """)
    return


@app.cell
def _(difference_result_fig):
    difference_result_fig
    return


if __name__ == "__main__":
    app.run()
