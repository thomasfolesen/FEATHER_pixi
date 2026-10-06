import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import osmnx as ox
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from src.feather import FEATHER, FEATHER_new

    return FEATHER_new, np, nx, ox


@app.cell
def _(G, nx):
    AG = nx.Graph()

    AG.add_edge(0, 1, weight=2.0)
    AG.add_edge(1, 2, weight=3.0)
    AG.add_edge(2, 3, weight=1.5)

    print(type(G))
    print(G.nodes())
    print(G.edges(data=True))
    return


@app.cell
def _(FEATHER_new, np, nx):
    #test FEATHER_new independently of CSV
    G_two = nx.Graph()

    G_two.add_edge(0, 1, weight=2.0)
    G_two.add_edge(1, 2, weight=3.0)
    G_two.add_edge(2, 3, weight=1.5)

    X = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ])

    model = FEATHER_new(
        theta_max=2.5,
        eval_points=5,
        order=2
    )

    model.fit(G_two, X)

    embedding = model.get_embedding()

    print(embedding.shape)
    print(embedding)
    return


@app.cell
def _():
    copenhagen = [
    "Københavns Kommune, Denmark",
    "Frederiksberg Kommune, Denmark",
    ]
    return (copenhagen,)


@app.cell
def _(copenhagen, ox):
    gdf = ox.geocoder.geocode_to_gdf(copenhagen)
    return (gdf,)


@app.cell
def _(gdf):
    gdf[["display_name", "osm_type", "osm_id"]]
    return


@app.cell
def _(copenhagen, ox):
    G = ox.graph_from_place(copenhagen, network_type="walk", simplify=True, retain_all=True,)
    return (G,)


@app.cell
def _(G, ox):

    fig,ax = ox.plot_graph(
        G, node_size=0, edge_linewidth=0.25,
    )
    return


@app.cell
def _(G):
    type(G)
    return


@app.cell
def _(G):
    print(f"Nodes: {len(G.nodes)}")
    print(f"Edges: {len(G.edges)}")
    return


@app.cell
def _(G):
    list(G.nodes())[:5]
    return


@app.cell
def _(G):
    u, v, k = list(G.edges(keys=True))[0]
    G.edges[u, v, k]
    return


@app.cell
def _(G, ox):
    nodes, edges = ox.graph_to_gdfs(G)
    return edges, nodes


@app.cell
def _(nodes):
    nodes.head(30)
    return


@app.cell
def _(edges):
    edges.head(30)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Experiments
    """)
    return


@app.cell
def _():
    # #Reindex nodes
    # G_new = nx.convert_node_labels_to_integers(G)

    # #Create weight attribute. Change the weight to length
    # for u, v, k, data in G.edges(keys=True, data=True):
    #     data["weight"] = data["length"]

    # # Degree feature
    # X = np.array([
    # G.degree(n)
    # for n in range(G.number_of_nodes())
    # ])

    # # 5. FEATHER
    # model = FEATHER_new()
    # model.fit(G, X)
 
    # embedding = model.get_embedding()
 
    # print(embedding.shape)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
