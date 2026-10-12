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
    import os

    from src.feather import FEATHER, FEATHER_new
    from src.config_loader import load_config, load_amenities

    return (
        FEATHER,
        FEATHER_new,
        load_amenities,
        load_config,
        mo,
        np,
        nx,
        os,
        ox,
        pd,
        pdna,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load config and amenities (.txt files)
    """)
    return


@app.cell
def _(load_amenities, load_config):
    config = load_config("input/geo/config.txt")
    amenities = load_amenities("input/geo/amenities.txt")

    print(config)
    print(amenities)
    return amenities, config


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dropdown menu of amenities
    """)
    return


@app.cell
def _(amenities, mo):
    amenity_dropdown = mo.ui.dropdown(
        options=amenities,
        value=amenities[0],
        label="Select amenity",
    )
    amenity_dropdown
    return (amenity_dropdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download street network (cached for each city)
    """)
    return


@app.cell
def _(config, os, ox):
    place = config["place"]

    # City folder - cached
    city_folder_name = place.replace(",", "").replace(" ", "_").lower()
    cache_dir = os.path.join("cache", city_folder_name)
    os.makedirs(cache_dir, exist_ok=True)

    graph_path = os.path.join(cache_dir, "city_graph.graphml")

    if os.path.exists(graph_path):
        print(f"Loading cached graph from {graph_path}")
        city_graph = ox.load_graphml(graph_path)
        city_graph = ox.convert.to_undirected(city_graph)
    else:
        print(f"Downloading graph for {place} ...")
        city_graph = ox.graph.graph_from_place(
            place, network_type="walk", simplify=True, retain_all=False
        )
        city_graph = ox.convert.to_undirected(city_graph)
        ox.save_graphml(city_graph, graph_path)
        print(f"Graph saved to {graph_path}")

    #project to CRS in order to use metrics (meters)
    city_graph_projected = ox.project_graph(city_graph)

    #define nodes and edges from the OSMNx graph to GeoDataFrames
    city_nodes, city_edges = ox.graph_to_gdfs(city_graph_projected) 
    city_edges = city_edges.reset_index() #we reset u and v due to Pandana needing normal columns

    #plot raw street network
    network_fig, _ = ox.plot_graph(
        city_graph_projected,
        node_size=0.75,
        edge_linewidth=0.5,
        bgcolor="white",
        show=False,
        close=False,
    )

    network_fig
    return (
        cache_dir,
        city_edges,
        city_graph,
        city_graph_projected,
        city_nodes,
        place,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Pipeline for one amenity
    #### Returns 5 matplotlib figures
    """)
    return


@app.cell
def _(FEATHER, FEATHER_new, np, nx, ox, pd, pdna):

    def amenity_pipeline(
        amenity,
        config,
        city_graph,
        city_graph_projected,
        city_nodes,
        city_edges,
        place,
        cache_dir,
        search_distance=50_000,
    ):
        """
        Run FEATHER and FEATHER_new for one OSM amenity tag

        This will return:
        - poi_fig
        - distance _fig
        - feather_fig
        - feather_new_fig
        - difference_fig

        and matrices!
        """
        order = config["order"]
        theta_max = config["theta_max"]
        eval_pts = config["eval_points"]

        #------------  1  -------------------
        #Get the amenity POIs from OSM
        tags = {"amenity": [amenity]}
        features = ox.features.features_from_place(place, tags)

        #------------  2  -------------------
        #plot graph of road network (POI map)
        poi_fig, poi_ax = ox.plot.plot_graph(
            city_graph,
            show=False,
            close=False,
            bgcolor="white",
            edge_linewidth=0.5,
            node_size=0.75,
        )
        features.plot(
            column="amenity",
            categorical=True,
            legend=True,
            ax=poi_ax,
            cmap="Set3",
        )
        poi_ax.set_title(f"{amenity.capitalize()} locations", fontsize=14)

        #------------  3  -------------------
        #Create the pandana network and distances
        network = pdna.Network(
            city_nodes.geometry.x,
            city_nodes.geometry.y,
            city_edges["u"],
            city_edges["v"],
            city_edges[["length"]],
        )

        pois = features.to_crs(city_nodes.crs).copy()
        pois["geometry"] = pois.geometry.centroid

        network.set_pois(
            category=amenity,
            maxdist=search_distance,
            maxitems=100,
            x_col=pois.geometry.x,
            y_col=pois.geometry.y,
        )

        nearest = network.nearest_pois(
            distance=search_distance,
            category=amenity,
            num_pois=1,
        )

        amenity_nodes = city_nodes.copy()
        amenity_nodes[f"nearest_{amenity}"] = nearest.iloc[:, 0]

        #------------  4  -------------------
        distance_fig, distance_ax = ox.plot_graph(
            city_graph_projected,
            node_size=0,
            edge_color="#afdffe",
            edge_linewidth=0.5,
            bgcolor="#1a1a1a",
            show=False,
            close=False,
        )
        amenity_nodes.plot(
            ax=distance_ax,
            column=f"nearest_{amenity}",
            cmap="plasma",
            markersize=0.5,
            alpha=0.8,
            legend=True,
            legend_kwds={
                "shrink": 0.5,
                "label": f"Distance to closest {amenity} (m)",
                "orientation": "vertical",
            },
        )
        distance_ax.set_title(f"Walking distance to nearest {amenity}", fontsize=14)

        #------------  5  -------------------
        original_nodes = list(city_graph.nodes())
        node_mapping = {}
        for new, old in enumerate(original_nodes):
            node_mapping[old] = new

        X_raw = amenity_nodes.loc[original_nodes, f"nearest_{amenity}"].to_numpy(dtype=float)
        X_raw = np.nan_to_num(X_raw, nan=search_distance)

        def _make_columns(order, eval_pts):
            cols = []
            for r in range(1, order + 1):
                for t in range(eval_pts):
                    cols.append(f"cos_r{r}_theta{t}")
                for t in range(eval_pts):
                    cols.append(f"sin_{r}_theta{t}")
            return cols

        vis_r = order

        #------------  6  -------------------
        feather_graph = nx.relabel_nodes(city_graph.copy(), node_mapping, copy=True)

        feather_model = FEATHER(
            theta_max=theta_max,
            eval_points=eval_pts,
            order=order,
        )

        #Get the intermediate matrices
        A_mat = nx.adjacency_matrix(feather_graph, nodelist=range(feather_graph.number_of_nodes()))
        D_inv = feather_model._create_D_inverse(feather_graph)
        A_tilde = feather_model._create_A_tilde(feather_graph)


        feather_model.fit(feather_graph, X_raw)
        feather_embedding = feather_model.get_embedding()

        feather_df = pd.DataFrame(
            feather_embedding,
            columns=_make_columns(order, eval_pts),
        )
        feather_df["osmid"] = original_nodes

        cos_cols = [f"cos_r{vis_r}_theta{t}" for t in range(eval_pts)]
        feather_df[f"cos_mean_r{vis_r}"] = feather_df[cos_cols].mean(axis=1)

        feather_map_nodes = amenity_nodes.loc[original_nodes].copy()
        feather_map_nodes[f"cos_mean_r{vis_r}"] = feather_df[f"cos_mean_r{vis_r}"].to_numpy()

        feather_fig, feather_ax = ox.plot_graph(
            city_graph_projected,
            node_size=0,
            edge_color="lightgray",
            edge_linewidth=0.05,
            bgcolor="black",
            show=False,
            close=False,
        )
        feather_map_nodes.plot(
            ax=feather_ax,
            column=f"cos_mean_r{vis_r}",
            cmap="plasma",
            markersize=0.5,
            alpha=0.9,
            legend=True,
            zorder=3,
            vmin=-1,
            vmax=1,
            legend_kwds={
                "shrink": 0.5,
                "label": f"FEATHER {amenity} cosine mean, r={vis_r}",
                "orientation": "vertical",
            },
        )
        feather_ax.set_title(f"FEATHER — {amenity}", fontsize=14)

        #------------  7  -------------------
        feather_new_graph = nx.relabel_nodes(city_graph.copy(), node_mapping, copy=True)
        for _, _, edge_data in feather_new_graph.edges(data=True):
            edge_data["weight"] = edge_data.get("length", 1.0)

        feather_new_model = FEATHER_new(
            theta_max=theta_max,
            eval_points=eval_pts,
            order=order,
        )

        #Get the WEIGHTED  intermediate matrices (feather_new)
        A_mat_w   = nx.adjacency_matrix(feather_new_graph,
                        nodelist=range(feather_new_graph.number_of_nodes()), weight="weight")
        D_inv_w   = feather_new_model._create_D_inverse(feather_new_graph)
        A_tilde_w = feather_new_model._create_A_tilde(feather_new_graph)

        feather_new_model.fit(feather_new_graph, X_raw)
        feather_new_embedding = feather_new_model.get_embedding()

        feather_new_df = pd.DataFrame(
            feather_new_embedding,
            columns=_make_columns(order, eval_pts),
        )
        feather_new_df["osmid"] = original_nodes

        cos_cols_new = [f"cos_r{vis_r}_theta{t}" for t in range(eval_pts)]
        feather_new_df[f"cos_mean_r{vis_r}"] = feather_new_df[cos_cols_new].mean(axis=1)

        feather_new_map_nodes = amenity_nodes.loc[original_nodes].copy()
        feather_new_map_nodes[f"cos_mean_r{vis_r}"] = feather_new_df[f"cos_mean_r{vis_r}"].to_numpy()

        feather_new_fig, feather_new_ax = ox.plot_graph(
            city_graph_projected,
            node_size=0,
            edge_color="lightgray",
            edge_linewidth=0.05,
            bgcolor="black",
            show=False,
            close=False,
        )
        feather_new_map_nodes.plot(
            ax=feather_new_ax,
            column=f"cos_mean_r{vis_r}",
            cmap="plasma",
            markersize=0.5,
            alpha=0.9,
            legend=True,
            zorder=3,
            vmin=-1,
            vmax=1,
            legend_kwds={
                "shrink": 0.5,
                "label": f"FEATHER_new {amenity} cosine mean, r={vis_r}",
                "orientation": "vertical",
            },
        )
        feather_new_ax.set_title(f"FEATHER_new — {amenity}", fontsize=14)

        #------------  8  -------------------
        comparison_nodes = amenity_nodes.loc[original_nodes].copy()
        comparison_nodes["difference"] = (
            feather_new_df[f"cos_mean_r{vis_r}"].to_numpy()
            - feather_df[f"cos_mean_r{vis_r}"].to_numpy()
        )

        difference_fig, difference_ax = ox.plot_graph(
            city_graph_projected,
            node_size=0,
            edge_color="lightgray",
            edge_linewidth=0.05,
            bgcolor="black",
            show=False,
            close=False,
        )
        comparison_nodes.plot(
            ax=difference_ax,
            column="difference",
            cmap="coolwarm",
            markersize=0.5,
            alpha=0.9,
            legend=True,
            zorder=3,
            legend_kwds={
                "shrink": 0.5,
                "label": f"FEATHER_new − FEATHER ({amenity})",
                "orientation": "vertical",
            },
        )
        difference_ax.set_title(
            f"Difference (FEATHER_new − FEATHER) — {amenity}", fontsize=14
        )

        return {
            "poi_fig":         poi_fig,
            "distance_fig":    distance_fig,
            "feather_fig":     feather_fig,
            "feather_new_fig": feather_new_fig,
            "difference_fig":  difference_fig,
            #embeddings
            "feather_df": feather_df,
            "feather_new_df": feather_new_df,
            #intermediate matrices (FEATHER unweighted - original)
            "A": A_mat,
            "D_inv": D_inv,
            "A_tilde": A_tilde,
            #intermediate matrice (FEATHER_new weighted)
            "A_w": A_mat_w,
            "D_inv_w": D_inv_w,
            "A_tilde_w": A_tilde_w,

        }

    return (amenity_pipeline,)


@app.cell
def _(
    amenity_dropdown,
    amenity_pipeline,
    cache_dir,
    city_edges,
    city_graph,
    city_graph_projected,
    city_nodes,
    config,
    place,
):

    results = amenity_pipeline(
        amenity=amenity_dropdown.value,
        config=config,
        city_graph=city_graph,
        city_graph_projected=city_graph_projected,
        city_nodes=city_nodes,
        city_edges=city_edges,
        place=place,
        cache_dir=cache_dir,
    )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Figures
    """)
    return


@app.cell
def _(results):
    results["poi_fig"]
    return


@app.cell
def _(results):
    results["distance_fig"]
    return


@app.cell
def _(results):
    results["feather_fig"]
    return


@app.cell
def _(results):
    results["feather_new_fig"]
    return


@app.cell
def _(results):
    results["difference_fig"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrices
    """)
    return


@app.cell
def _(results):
    # Quick peek at the top-left corner of each matrix
    # Sparse matrices: use .toarray() to see values
    print("=== A (unweighted adjacency) ===")
    print(results["A"].toarray()[:6, :6])

    print("\n=== D_inv (unweighted) ===")
    print(results["D_inv"].toarray()[:6, :6])

    print("\n=== Â = D⁻¹A (unweighted) ===")
    print(results["A_tilde"].toarray()[:6, :6])

    print("\n=== A_w (weighted adjacency) ===")
    print(results["A_w"].toarray()[:6, :6])

    print("\n=== D_inv_w (weighted) ===")
    print(results["D_inv_w"].toarray()[:6, :6])

    print("\n=== Â_w = D⁻¹A (weighted) ===")
    print(results["A_tilde_w"].toarray()[:6, :6])
    return


@app.cell
def _(results):
    # Full embedding DataFrames
    results["feather_df"]
    return


@app.cell
def _(results):
    results["feather_new_df"]
    return


if __name__ == "__main__":
    app.run()
