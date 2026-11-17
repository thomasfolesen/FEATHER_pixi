import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FEATHER vs FEATHER_new — step by step on Copenhagen

    Both models compute the same object: an empirical characteristic function
    of a node feature, evaluated over the distribution reached by an $r$-step
    random walk.

    $$\varphi_u^{(r)}(\theta)=\mathbb{E}_{v\sim P^{r}(u,\cdot)}\!\left[e^{i\theta x_v}\right]
    =\sum_v \tilde{A}^{r}_{uv}\cos(\theta x_v)+i\sum_v \tilde{A}^{r}_{uv}\sin(\theta x_v)$$

    The only difference is the transition matrix $\tilde{A}=D^{-1}A$:

    | | `FEATHER` | `FEATHER_new` |
    |---|---|---|
    | $A_{uv}$ | $1$ | $1/L_{uv}$ (conductance) |
    | $d_u$ | neighbour count | $\sum_v 1/L_{uv}$ |
    | $\tilde{A}_{uv}$ | $1/d_u$, **uniform** | $\dfrac{1/L_{uv}}{\sum_k 1/L_{uk}}$, **prefers short streets** |

    `FEATHER_new` now derives the weights itself from the `length` attribute,
    so both models take the **same graph**. No weight assignment is needed in
    this notebook.
    """)
    return


@app.cell
def _():
    import os
    import re
    import itertools

    import numpy as np
    import pandas as pd
    import networkx as nx
    import osmnx as ox
    import pandana as pdna
    import matplotlib.pyplot as plt
    from scipy import sparse

    from src.feather import FEATHER, FEATHER_new
    from src.config_loader import load_config, load_amenities

    return (
        FEATHER,
        FEATHER_new,
        itertools,
        load_amenities,
        load_config,
        np,
        nx,
        os,
        ox,
        pd,
        pdna,
        plt,
        re,
        sparse,
    )


@app.cell(hide_code=True)
def _(mo, np, pd, sparse):
    def to_dense(M):
        return np.asarray(M.todense()) if sparse.issparse(M) else np.asarray(M)

    def row_dense(M, i):
        """One dense row; works for both scipy sparse matrix and sparse array."""
        return np.asarray(M.tocsr()[i].todense()).ravel()

    def row_sums(M):
        return np.asarray(M.sum(axis=1)).ravel()

    def mat_df(M, k=8, decimals=3):
        dense = to_dense(M)[:k, :k]
        labels = [f"n{i}" for i in range(dense.shape[0])]
        return pd.DataFrame(dense, index=labels, columns=labels).round(decimals)

    def panel(title, *items):
        return mo.vstack([mo.md(f"**{title}**"), *items], gap=0.6)

    def duo(left, right):
        return mo.hstack([left, right], widths="equal", gap=2, align="start")

    def morans_i(values, A_binary):
        """Global Moran's I of `values` under binary adjacency `A_binary`."""
        z = np.asarray(values, dtype=float)
        z = z - z.mean()
        denom = (z**2).sum()
        if denom == 0:
            return float("nan")
        return (len(z) / float(A_binary.sum())) * float(z @ (A_binary @ z)) / denom

    return duo, mat_df, morans_i, panel, row_dense, row_sums, to_dense


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. The Copenhagen graph

    Loaded from `cache/<place_slug>/city_graph.graphml` — simplified,
    undirected, exactly the graph the pipeline uses.
    """)
    return


@app.cell
def _(load_amenities, load_config):
    cfg = load_config("input/geo/config.txt")
    amenity_list = load_amenities("input/geo/amenities.txt")

    PLACE = cfg["place"]
    THETA_MAX = cfg["theta_max"]
    EVAL_POINTS = cfg["eval_points"]
    ORDER = cfg["order"]
    return EVAL_POINTS, ORDER, PLACE, THETA_MAX, amenity_list


@app.cell
def _(PLACE, os, ox, re):
    def place_slug(place):
        """'Copenhagen Municipality, Denmark' -> 'copenhagen_municipality_denmark'"""
        return re.sub(r"[^a-z0-9]+", "_", place.lower()).strip("_")

    CACHE_DIR = "cache"
    PLACE_CACHE_DIR = os.path.join(CACHE_DIR, place_slug(PLACE))
    GRAPHML_SNAPSHOT = os.path.join(PLACE_CACHE_DIR, "city_graph.graphml")
    os.makedirs(PLACE_CACHE_DIR, exist_ok=True)

    # OSMnx HTTP cache, used by the separate POI query below
    ox.settings.use_cache = True
    ox.settings.cache_folder = CACHE_DIR
    ox.settings.log_console = False
    return (GRAPHML_SNAPSHOT,)


@app.cell
def _(amenity_list, mo):
    amenity_ui = mo.ui.dropdown(
        options=amenity_list,
        value=amenity_list[0],
        label="Amenity used as the node feature",
    )
    amenity_ui
    return (amenity_ui,)


@app.cell
def _(GRAPHML_SNAPSHOT, PLACE, os, ox):
    if os.path.exists(GRAPHML_SNAPSHOT):
        G_raw = ox.io.load_graphml(GRAPHML_SNAPSHOT)
        graph_source = f"loaded cached snapshot `{GRAPHML_SNAPSHOT}`"
    else:
        G_raw = ox.graph.graph_from_place(
            PLACE, network_type="walk", simplify=True, retain_all=False
        )
        G_raw = ox.convert.to_undirected(G_raw)
        ox.io.save_graphml(G_raw, GRAPHML_SNAPSHOT)
        graph_source = f"downloaded; snapshot written to `{GRAPHML_SNAPSHOT}`"

    if G_raw.is_directed():
        G_raw = ox.convert.to_undirected(G_raw)
    return G_raw, graph_source


@app.cell
def _(G_raw, nx):
    n_selfloops = nx.number_of_selfloops(G_raw)
    n_parallel = G_raw.number_of_edges() - len(
        {(min(u, v), max(u, v)) for u, v in G_raw.edges()}
    )
    return n_parallel, n_selfloops


@app.cell(hide_code=True)
def _(G_raw, PLACE, graph_source, mo, n_parallel, n_selfloops):
    mo.md(rf"""
    - place: **{PLACE}** &nbsp;·&nbsp; {graph_source}
    - nodes **{G_raw.number_of_nodes():,}** &nbsp;·&nbsp; edges **{G_raw.number_of_edges():,}**
    - directed **{G_raw.is_directed()}** &nbsp;·&nbsp; multigraph **{G_raw.is_multigraph()}**

    > **Self-loops: {n_selfloops}** — removed below. These are created by
    > `simplify=True` whenever a closed ring (park loop path, square
    > perimeter, turning circle) attaches to the network at a single junction.
    > NetworkX counts a self-loop **twice** in `degree` but stores it **once**
    > on the adjacency diagonal, so that row of $\tilde{{A}}$ sums to less
    > than 1 and probability leaks out of the walk.
    >
    > **Parallel edges: {n_parallel}** — kept. `ox.convert.to_undirected`
    > already collapsed the reciprocal pairs; what survives is two physically
    > distinct routes between the same junctions. `nx.adjacency_matrix` sums
    > them, which is correct for conductances: two routes in parallel really
    > do conduct more.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. The node feature $x$

    Walking-network distance from each node to the nearest amenity, in metres
    (Pandana, on the projected graph). Both models get the **identical**
    vector, so every difference downstream comes from $\tilde{A}$.
    """)
    return


@app.cell
def _(G_raw, ox, pdna):
    G_proj = ox.project_graph(G_raw)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj)
    edges_gdf = edges_gdf.reset_index()

    pdna_net = pdna.Network(
        nodes_gdf.geometry.x,
        nodes_gdf.geometry.y,
        edges_gdf["u"],
        edges_gdf["v"],
        edges_gdf[["length"]],
    )
    return nodes_gdf, pdna_net


@app.cell
def _(PLACE, amenity_ui, nodes_gdf, np, ox, pd, pdna_net):
    AMENITY = amenity_ui.value
    SEARCH_DIST = 50_000

    pois = ox.features.features_from_place(PLACE, {"amenity": [AMENITY]})
    pois = pois.to_crs(nodes_gdf.crs).copy()
    pois["geometry"] = pois.geometry.centroid

    pdna_net.set_pois(
        category=AMENITY,
        maxdist=SEARCH_DIST,
        maxitems=100,
        x_col=pois.geometry.x,
        y_col=pois.geometry.y,
    )
    nearest_raw = pdna_net.nearest_pois(
        distance=SEARCH_DIST, category=AMENITY, num_pois=1
    )

    feature_by_osmid = pd.Series(
        np.nan_to_num(nearest_raw.iloc[:, 0].to_numpy(dtype=float), nan=SEARCH_DIST),
        index=nodes_gdf.index,
        name=f"nearest_{AMENITY}",
    )
    return AMENITY, feature_by_osmid, pois


@app.cell(hide_code=True)
def _(AMENITY, feature_by_osmid, mo, pois):
    mo.md(f"""
    `nearest_{AMENITY}` over {len(feature_by_osmid):,} nodes from
    {len(pois):,} POIs — min **{feature_by_osmid.min():.0f} m** ·
    median **{feature_by_osmid.median():.0f} m** ·
    max **{feature_by_osmid.max():.0f} m**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. A small inspection subgraph

    An 8×8 corner of a 50,000×50,000 matrix is almost all zeros. Every matrix
    below is printed on a BFS ball around the city centre, self-loops removed
    and nodes relabelled $0\dots N-1$ as `nodelist=range(n)` requires.

    **One graph serves both models.** It carries `length` but no `weight`, so
    `FEATHER` sees a binary adjacency, while `FEATHER_new` copies it
    internally and derives `weight = 1/length` itself.
    """)
    return


@app.cell
def _(mo):
    n_sub_ui = mo.ui.slider(
        start=50, stop=1500, step=50, value=250,
        label="Subgraph size (nodes)", show_value=True,
    )
    k_show_ui = mo.ui.slider(
        start=4, stop=15, step=1, value=8,
        label="Matrix block to print (k × k)", show_value=True,
    )
    mo.hstack([n_sub_ui, k_show_ui], justify="start", gap=2)
    return k_show_ui, n_sub_ui


@app.cell
def _(G_raw, feature_by_osmid, itertools, n_sub_ui, np, nx, ox):
    centre_node = ox.distance.nearest_nodes(G_raw, X=12.5683, Y=55.6761)

    ball_nodes = list(
        itertools.islice(nx.bfs_tree(G_raw, source=centre_node), int(n_sub_ui.value))
    )

    sub_base = G_raw.subgraph(ball_nodes).copy()
    sub_base.remove_edges_from(nx.selfloop_edges(sub_base))
    sub_base = sub_base.subgraph(
        max(nx.connected_components(sub_base), key=len)
    ).copy()

    sub_osmids = list(sub_base.nodes())
    G_sub = nx.relabel_nodes(
        sub_base, {old: new for new, old in enumerate(sub_osmids)}, copy=True
    )
    N_SUB = G_sub.number_of_nodes()

    X_sub = feature_by_osmid.loc[sub_osmids].to_numpy(dtype=float)
    X_sub = np.nan_to_num(X_sub, nan=float(feature_by_osmid.max()))
    return G_sub, N_SUB, X_sub, sub_osmids


@app.cell
def _(G_sub):
    # Mirror of the conductance graph FEATHER_new builds internally.
    # Kept only so the intermediate matrices can be displayed; the model
    # itself is always called on G_sub.
    G_cond = G_sub.copy()
    for _u, _v, _d in G_cond.edges(data=True):
        _d["weight"] = 1.0 / max(float(_d.get("length", 1.0)), 1.0)

    has_weight_in_G_sub = any("weight" in d for _, _, d in G_sub.edges(data=True))
    return G_cond, has_weight_in_G_sub


@app.cell(hide_code=True)
def _(G_sub, N_SUB, X_sub, duo, has_weight_in_G_sub, mo, nx, panel, pd):
    def _edges_df(G, cols):
        return pd.DataFrame(
            [
                {"u": u, "v": v, **{c: d.get(c, "— absent —") for c in cols}}
                for u, v, d in list(G.edges(data=True))[:6]
            ]
        )

    mo.vstack(
        [
            mo.md(
                f"Subgraph **{N_SUB}** nodes, **{G_sub.number_of_edges()}** edges, "
                f"self-loops **{nx.number_of_selfloops(G_sub)}**. "
                f"Feature range **{X_sub.min():.0f} – {X_sub.max():.0f} m**. "
                f"`weight` present in G_sub: **{has_weight_in_G_sub}**."
            ),
            duo(
                panel(
                    "G_sub as given to both models",
                    _edges_df(G_sub, ["length", "weight"]),
                ),
                panel(
                    "What FEATHER_new builds internally",
                    mo.md(
                        "```python\n"
                        "G = graph.copy()\n"
                        "for u, v, d in G.edges(data=True):\n"
                        "    length = max(float(d.get('length', 1.0)), 1.0)\n"
                        "    d['weight'] = 1.0 / length\n"
                        "```"
                    ),
                ),
            ),
        ]
    )
    return


@app.cell
def _(EVAL_POINTS, FEATHER, FEATHER_new, ORDER, THETA_MAX):
    model_old = FEATHER(theta_max=THETA_MAX, eval_points=EVAL_POINTS, order=ORDER)
    model_new = FEATHER_new(theta_max=THETA_MAX, eval_points=EVAL_POINTS, order=ORDER)
    return model_new, model_old


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step A — calculating $A$

    Both models call the same line:

    ```python
    A = nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes()), weight="weight")
    ```

    The key point: **`weight` is a dictionary key, not a flag.** The build is
    `for u, v, wt in G.edges(data=weight, default=1)`, so an edge *without*
    the attribute contributes $1$. `FEATHER` passes a graph that has no
    `weight` key and gets a binary matrix; `FEATHER_new` passes its own copy
    where `weight = 1/length` and gets conductances. Same code, different
    graph.
    """)
    return


@app.cell
def _(G_cond, G_sub, N_SUB, nx):
    A_old = nx.adjacency_matrix(G_sub, nodelist=range(N_SUB))
    A_new = nx.adjacency_matrix(G_cond, nodelist=range(N_SUB), weight="weight")
    return A_new, A_old


@app.cell(hide_code=True)
def _(A_new, A_old, duo, k_show_ui, mat_df, mo, panel):
    _k = int(k_show_ui.value)

    duo(
        panel(
            "FEATHER — A (binary)",
            mat_df(A_old, k=_k),
            mo.md(
                f"non-zeros **{A_old.nnz}** · values "
                f"**{A_old.data.min():.0f} – {A_old.data.max():.0f}** (unitless)"
            ),
        ),
        panel(
            "FEATHER_new — A (conductance, 1/m)",
            mat_df(A_new, k=_k, decimals=4),
            mo.md(
                f"non-zeros **{A_new.nnz}** · values "
                f"**{A_new.data.min():.4f} – {A_new.data.max():.4f}**"
            ),
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Same sparsity pattern — the same streets exist in both. Only the
    magnitudes differ. Entries above 1 on the left are **parallel edges**
    being summed, not weights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step B — calculating $D$

    $D$ is the diagonal of row sums of $A$ — the normaliser that turns
    connection strengths into probabilities. **This is where the two classes
    differ in source code.**

    - `FEATHER`: `graph.degree[u]`, a *count*
    - `FEATHER_new`: `graph.degree(u, weight="weight")`, a *sum of conductances*
    """)
    return


@app.cell
def _(G_cond, G_sub, N_SUB, np):
    deg_old = np.array([G_sub.degree[n] for n in range(N_SUB)], dtype=float)
    deg_new = np.array(
        [G_cond.degree(n, weight="weight") for n in range(N_SUB)], dtype=float
    )
    return deg_new, deg_old


@app.cell(hide_code=True)
def _(deg_new, deg_old, duo, k_show_ui, mo, panel, pd):
    _k = int(k_show_ui.value)
    _idx = [f"n{i}" for i in range(_k)]

    duo(
        panel(
            "FEATHER — unweighted degree",
            pd.DataFrame({"d_u (count)": deg_old[:_k]}, index=_idx),
            mo.md(
                f"mean **{deg_old.mean():.2f}** · "
                f"min **{deg_old.min():.0f}** · max **{deg_old.max():.0f}**"
            ),
        ),
        panel(
            "FEATHER_new — total conductance",
            pd.DataFrame({"d_u^w (1/m)": deg_new[:_k].round(4)}, index=_idx),
            mo.md(
                f"mean **{deg_new.mean():.4f}** · "
                f"min **{deg_new.min():.4f}** · max **{deg_new.max():.4f}**"
            ),
        ),
    )
    return


@app.cell(hide_code=True)
def _(deg_new, deg_old, mo, np, pd):
    _corr = float(np.corrcoef(deg_old, deg_new)[0, 1])
    _cv = float(deg_new.std() / deg_new.mean())

    mo.vstack(
        [
            mo.md(
                f"""
                $d^w_u / d_u$ is the **mean conductance per street** at node $u$ —
                high in dense fine-grained blocks, low where the node sits on long
                arterial roads. Correlation between the two degree definitions
                **{_corr:.3f}**.

                Weighted-degree coefficient of variation **{_cv:.3f}**. A high
                value warns that a few nodes dominate the walk, which is the
                $1/L$ failure mode — watch it if short connector stubs are common.
                """
            ),
            pd.DataFrame(
                {
                    "d_u (count)": deg_old[:10],
                    "d_u^w (1/m)": deg_new[:10].round(4),
                    "mean conductance / street": (deg_new[:10] / deg_old[:10]).round(4),
                },
                index=[f"n{i}" for i in range(10)],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step C — calculating $D^{-1}$

    A diagonal matrix inverts entrywise, so this is $1/d_u$ down the diagonal
    as a sparse COO matrix.
    """)
    return


@app.cell
def _(G_cond, G_sub, model_new, model_old):
    D_inv_old = model_old._create_D_inverse(G_sub)
    # passed the conductance mirror, because the method reads edge_data["weight"]
    D_inv_new = model_new._create_D_inverse(G_cond)
    return D_inv_new, D_inv_old


@app.cell(hide_code=True)
def _(D_inv_new, D_inv_old, duo, k_show_ui, mat_df, mo, panel):
    _k = int(k_show_ui.value)

    duo(
        panel(
            "FEATHER — 1 / neighbour count",
            mat_df(D_inv_old, k=_k, decimals=4),
            mo.md("Typically 0.25 – 0.5."),
        ),
        panel(
            "FEATHER_new — 1 / total conductance",
            mat_df(D_inv_new, k=_k, decimals=2),
            mo.md("Metres, so order $10^1$ to $10^2$."),
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The magnitudes differ wildly, but $A$ differs by the same factor, so
    $D^{-1}A$ lands back in $[0,1]$ for both. The scale cancels; only the
    *relative* weighting among a node's neighbours survives.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step D — calculating $\tilde{A} = D^{-1}A$

    The random-walk transition matrix. $\tilde{A}_{uv}$ is the probability of
    stepping from $u$ to $v$, so every row sums to 1.

    $$\tilde{A}^{\text{FEATHER}}_{uv}=\frac{1}{d_u}
    \qquad
    \tilde{A}^{\text{FEATHER\_new}}_{uv}=\frac{1/L_{uv}}{\sum_k 1/L_{uk}}$$
    """)
    return


@app.cell
def _(G_sub, model_new, model_old):
    A_tilde_old = model_old._create_A_tilde(G_sub)
    # G_sub, not G_cond: the method derives the weights itself
    A_tilde_new = model_new._create_A_tilde(G_sub)
    return A_tilde_new, A_tilde_old


@app.cell(hide_code=True)
def _(
    A_new,
    A_tilde_new,
    A_tilde_old,
    D_inv_new,
    duo,
    k_show_ui,
    mat_df,
    mo,
    np,
    panel,
    row_sums,
    to_dense,
):
    _k = int(k_show_ui.value)
    _rs_old = row_sums(A_tilde_old)
    _rs_new = row_sums(A_tilde_new)

    # the model's internal copy must reproduce the mirror exactly
    _matches = bool(np.allclose(to_dense(A_tilde_new), to_dense(D_inv_new.dot(A_new))))

    duo(
        panel(
            "FEATHER — Ã",
            mat_df(A_tilde_old, k=_k),
            mo.md(
                f"row sums **{_rs_old.min():.6f} – {_rs_old.max():.6f}** → stochastic ✓"
            ),
        ),
        panel(
            "FEATHER_new — Ã",
            mat_df(A_tilde_new, k=_k),
            mo.md(
                f"row sums **{_rs_new.min():.6f} – {_rs_new.max():.6f}** → stochastic ✓"
                f"<br>matches `D_inv_new · A_new`: **{_matches}**"
            ),
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The one-step walk from a single node

    The clearest view of the difference: pick a node and print where the
    walker actually goes.
    """)
    return


@app.cell
def _(A_tilde_new, N_SUB, mo, row_dense):
    _spread = []
    for _i in range(N_SUB):
        _v = row_dense(A_tilde_new, _i)
        _nz = _v[_v > 0]
        if len(_nz) >= 3:
            _spread.append((float(_nz.max() - _nz.min()), _i))
    focus_node = max(_spread)[1] if _spread else 0

    node_ui = mo.ui.number(
        start=0, stop=N_SUB - 1, step=1, value=int(focus_node),
        label="Inspect the one-step walk from node",
    )
    node_ui
    return (node_ui,)


@app.cell(hide_code=True)
def _(A_tilde_new, A_tilde_old, duo, mo, node_ui, np, panel, pd, row_dense):
    _u = int(node_ui.value)
    _p_old = row_dense(A_tilde_old, _u)
    _p_new = row_dense(A_tilde_new, _u)
    _nb = np.where(_p_old > 0)[0]

    duo(
        panel(
            f"One-step transition probabilities from n{_u}",
            pd.DataFrame(
                {
                    "neighbour": [f"n{j}" for j in _nb],
                    "P FEATHER": _p_old[_nb].round(4),
                    "P FEATHER_new": _p_new[_nb].round(4),
                }
            ).set_index("neighbour"),
        ),
        panel(
            "What this shows",
            mo.md(
                r"""
                `FEATHER` spreads probability **uniformly** — every neighbour gets
                $1/d_u$ regardless of distance.

                `FEATHER_new` spreads it **inversely to street length** — the
                walker prefers the shortest street. For an accessibility study
                this is the right direction: a long street costs more to walk, so
                it should be a less likely step.

                That is the entire behavioural difference. Every power of
                $\tilde{A}$ compounds it.
                """
            ),
        ),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step E — the $\theta$ grid and the $\cos/\sin$ lift

    The scalar feature is lifted into $2T$ numbers by evaluating
    $e^{i\theta x_u}$ on a grid of $T$ values of $\theta$. **Identical in both
    classes**, so shown once.
    """)
    return


@app.cell
def _(EVAL_POINTS, N_SUB, THETA_MAX, X_sub, np):
    theta_grid = np.linspace(0.01, THETA_MAX, EVAL_POINTS)
    X_outer = np.outer(X_sub, theta_grid).reshape(N_SUB, -1)
    X_lift = np.concatenate([np.cos(X_outer), np.sin(X_outer)], axis=1)
    return X_lift, X_outer


@app.cell(hide_code=True)
def _(EVAL_POINTS, N_SUB, THETA_MAX, X_lift, X_outer, mo, pd):
    mo.vstack(
        [
            mo.md(
                "```python\n"
                "theta = np.linspace(0.01, self.theta_max, self.eval_points)\n"
                "X = np.outer(X, theta)\n"
                "X = X.reshape(graph.number_of_nodes(), -1)\n"
                "X = np.concatenate([np.cos(X), np.sin(X)], axis=1)\n"
                "```"
            ),
            mo.md(
                f"""
                - $\\theta$ grid **{EVAL_POINTS}** points, 0.01 → {THETA_MAX}
                - lifted shape **{X_lift.shape}** = ({N_SUB}, 2 × {EVAL_POINTS})
                - largest angle $\\theta x$ reached: **{X_outer.max():,.0f} rad**

                > With $x$ in metres that angle is far outside the range where
                > $\\cos$ varies smoothly, so the values are effectively
                > pseudo-random phases. Both models suffer it equally so the
                > comparison stays fair — but see Step H.
                """
            ),
            pd.DataFrame(
                X_lift[:6, :6].round(4),
                index=[f"n{i}" for i in range(6)],
                columns=[f"cos θ{j}" for j in range(6)],
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step F — propagation $\tilde{A}^{r}$

    Block $r$ holds $\tilde{A}^{r}X_{\text{lift}}$. There is **no $r=0$
    block**, so the raw feature never appears in the embedding. Identical code
    in both classes; only $\tilde{A}$ differs.
    """)
    return


@app.cell
def _(A_tilde_new, A_tilde_old, ORDER, X_lift):
    def propagate(A_tilde, X0, order):
        """Reproduces the loop in fit(), keeping each block separately."""
        blocks, Xr = [], X0
        for _ in range(order):
            Xr = A_tilde.dot(Xr)
            blocks.append(Xr)
        return blocks

    blocks_old = propagate(A_tilde_old, X_lift, ORDER)
    blocks_new = propagate(A_tilde_new, X_lift, ORDER)
    return blocks_new, blocks_old


@app.cell
def _(ORDER, mo):
    r_ui = mo.ui.slider(
        start=1, stop=ORDER, step=1, value=ORDER,
        label="Propagation order r to inspect", show_value=True,
    )
    r_ui
    return (r_ui,)


@app.cell(hide_code=True)
def _(blocks_new, blocks_old, duo, mo, panel, pd, r_ui):
    _r = int(r_ui.value) - 1

    def _prev(B):
        return pd.DataFrame(
            B[:6, :5].round(4),
            index=[f"n{i}" for i in range(6)],
            columns=[f"cos θ{j}" for j in range(5)],
        )

    _bo, _bn = blocks_old[_r], blocks_new[_r]

    duo(
        panel(
            f"FEATHER — Ã^{_r + 1} X",
            _prev(_bo),
            mo.md(
                f"std **{_bo.std():.4f}** · range **{_bo.min():.3f} … {_bo.max():.3f}**"
            ),
        ),
        panel(
            f"FEATHER_new — Ã^{_r + 1} X",
            _prev(_bn),
            mo.md(
                f"std **{_bn.std():.4f}** · range **{_bn.min():.3f} … {_bn.max():.3f}**"
            ),
        ),
    )
    return


@app.cell(hide_code=True)
def _(ORDER, blocks_new, blocks_old, mo, np, pd):
    _rows = [
        {
            "order r": _i + 1,
            "FEATHER std": round(float(blocks_old[_i].std()), 5),
            "FEATHER_new std": round(float(blocks_new[_i].std()), 5),
            "mean |difference|": round(
                float(np.abs(blocks_old[_i] - blocks_new[_i]).mean()), 5
            ),
            "max |difference|": round(
                float(np.abs(blocks_old[_i] - blocks_new[_i]).max()), 5
            ),
        }
        for _i in range(ORDER)
    ]

    mo.vstack(
        [
            mo.md(
                r"""
                ### How the two walks drift apart with $r$

                At $r=1$ they differ only by the reweighting of immediate
                neighbours. Each further power compounds it, while the shrinking
                std shows both walks averaging towards a stationary distribution —
                but *different* stationary distributions.
                """
            ),
            pd.DataFrame(_rows).set_index("order r"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step G — the final embedding

    Shape $(N,\ \text{order}\times 2\times T)$. Within each order block the
    column order is **all cos, then all sin**.
    """)
    return


@app.cell
def _(G_sub, X_sub, model_new, model_old):
    model_old.fit(G_sub, X_sub)
    model_new.fit(G_sub, X_sub)

    emb_old = model_old.get_embedding()
    emb_new = model_new.get_embedding()
    return emb_new, emb_old


@app.cell
def _(EVAL_POINTS, ORDER):
    def make_columns(order=ORDER, eval_pts=EVAL_POINTS):
        cols = []
        for r in range(1, order + 1):
            cols += [f"cos_r{r}_theta{t}" for t in range(eval_pts)]
            cols += [f"sin_r{r}_theta{t}" for t in range(eval_pts)]
        return cols

    EMB_COLUMNS = make_columns()
    return (EMB_COLUMNS,)


@app.cell(hide_code=True)
def _(EMB_COLUMNS, duo, emb_new, emb_old, panel, pd, sub_osmids):
    df_old = pd.DataFrame(emb_old, columns=EMB_COLUMNS)
    df_old["osmid"] = sub_osmids

    df_new = pd.DataFrame(emb_new, columns=EMB_COLUMNS)
    df_new["osmid"] = sub_osmids

    duo(
        panel(f"FEATHER {emb_old.shape}", df_old.head(8).round(4)),
        panel(f"FEATHER_new {emb_new.shape}", df_new.head(8).round(4)),
    )
    return df_new, df_old


@app.cell(hide_code=True)
def _(
    A_old,
    EVAL_POINTS,
    ORDER,
    df_new,
    df_old,
    emb_new,
    emb_old,
    mo,
    morans_i,
    np,
    pd,
):
    _dot = (emb_old * emb_new).sum(axis=1)
    _nrm = np.linalg.norm(emb_old, axis=1) * np.linalg.norm(emb_new, axis=1)
    _sim = _dot / np.where(_nrm == 0, 1, _nrm)

    _cols = [f"cos_r{ORDER}_theta{t}" for t in range(EVAL_POINTS)]
    _m_old = df_old[_cols].mean(axis=1)
    _m_new = df_new[_cols].mean(axis=1)

    _summary = pd.DataFrame(
        {
            "metric": [
                "per-node cosine similarity (mean)",
                "per-node cosine similarity (min)",
                f"corr of cos_mean_r{ORDER}",
                f"mean |cos_mean_r{ORDER}| difference",
                "Moran I — FEATHER",
                "Moran I — FEATHER_new",
            ],
            "value": [
                round(float(_sim.mean()), 4),
                round(float(_sim.min()), 4),
                round(float(np.corrcoef(_m_old, _m_new)[0, 1]), 4),
                round(float(np.abs(_m_old - _m_new).mean()), 4),
                round(float(morans_i(_m_old.to_numpy(), A_old)), 3),
                round(float(morans_i(_m_new.to_numpy(), A_old)), 3),
            ],
        }
    ).set_index("metric")

    mo.vstack(
        [
            mo.md(
                """
                ### How different are the two embeddings?

                If the correlation sits above ~0.95 and the two Moran values are
                close, the weighting is not earning its complexity on this
                network and plain FEATHER is the better scientific choice.
                """
            ),
            _summary,
        ]
    )
    return


@app.cell(hide_code=True)
def _(EVAL_POINTS, ORDER, df_new, df_old, mo, plt):
    _cols = [f"cos_r{ORDER}_theta{t}" for t in range(EVAL_POINTS)]

    _fig, _ax = plt.subplots(figsize=(5.5, 5.5))
    _ax.scatter(df_old[_cols].mean(axis=1), df_new[_cols].mean(axis=1), s=6, alpha=0.5)
    _ax.plot([-1, 1], [-1, 1], "k--", linewidth=0.8)
    _ax.set_xlim(-1, 1)
    _ax.set_ylim(-1, 1)
    _ax.set_xlabel(f"FEATHER   cos_mean_r{ORDER}")
    _ax.set_ylabel(f"FEATHER_new   cos_mean_r{ORDER}")
    _ax.set_title("Node-by-node agreement (dashed = identical)")
    plt.close(_fig)

    mo.vstack(
        [
            mo.md(
                "Points on the dashed line are nodes where the weighting made no "
                "difference. Scatter away from it is the effect of the "
                "conductance walk."
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Step H — feature scaling

    The characteristic function only carries information while $\theta x$
    stays in a range where $\cos$ varies smoothly across the spread of $x$.
    Two failure modes bracket the useful region:

    - **$\theta x \ll 1$ everywhere** → $\cos \approx 1$ for every node. The
      embedding is nearly constant and discriminates between nothing.
    - **$\theta x \gg 2\pi$ with spread $\gg 2\pi$** → the phase wraps many
      times, so two nodes a metre apart in feature space land on unrelated
      values. The embedding is deterministic noise.

    With $x$ in metres you are in the second regime. **Moran's $I$** of the
    mapped quantity separates them:

    $$I=\frac{N}{\sum_{ij}A_{ij}}\cdot
    \frac{\sum_{ij}A_{ij}(z_i-\bar z)(z_j-\bar z)}{\sum_i (z_i-\bar z)^2}$$

    $I$ near 1 means neighbouring nodes have similar values — real spatial
    structure. $I$ near 0 means phase-wrapped noise. `std` near 0 is the
    opposite failure: every node got the same value.
    """)
    return


@app.cell
def _():
    SCALINGS = {
        "x  (metres)": lambda x: x,
        "x / 10": lambda x: x / 10.0,
        "x / 100": lambda x: x / 100.0,
        "x / 1000  (km)": lambda x: x / 1000.0,
        "min-max [0,1]": lambda x: (x - x.min()) / max(x.max() - x.min(), 1e-12),
        "z-score": lambda x: (x - x.mean()) / max(x.std(), 1e-12),
    }
    return (SCALINGS,)


@app.cell
def _(
    A_old,
    EVAL_POINTS,
    FEATHER,
    FEATHER_new,
    G_sub,
    ORDER,
    SCALINGS,
    THETA_MAX,
    X_sub,
    morans_i,
    np,
    pd,
):
    def _cos_mean(E, order=ORDER, eval_pts=EVAL_POINTS):
        start = (order - 1) * 2 * eval_pts
        return E[:, start : start + eval_pts].mean(axis=1)

    scaling_curves = {}
    _rows = []

    for _name, _fn in SCALINGS.items():
        _x = _fn(X_sub.astype(float))

        _m1 = FEATHER(theta_max=THETA_MAX, eval_points=EVAL_POINTS, order=ORDER)
        _m1.fit(G_sub, _x)
        _eo = _m1.get_embedding()

        _m2 = FEATHER_new(theta_max=THETA_MAX, eval_points=EVAL_POINTS, order=ORDER)
        _m2.fit(G_sub, _x)
        _en = _m2.get_embedding()

        _co, _cn = _cos_mean(_eo), _cos_mean(_en)
        scaling_curves[_name] = (_x, _co, _cn)

        _dot = (_eo * _en).sum(axis=1)
        _nrm = np.linalg.norm(_eo, axis=1) * np.linalg.norm(_en, axis=1)

        _rows.append(
            {
                "scaling": _name,
                "max θx (rad)": round(float(np.abs(_x).max() * THETA_MAX), 2),
                "std FEATHER": round(float(_co.std()), 4),
                "Moran I FEATHER": round(float(morans_i(_co, A_old)), 3),
                "std FEATHER_new": round(float(_cn.std()), 4),
                "Moran I FEATHER_new": round(float(morans_i(_cn, A_old)), 3),
                "model agreement": round(
                    float((_dot / np.where(_nrm == 0, 1, _nrm)).mean()), 3
                ),
            }
        )

    scaling_table = pd.DataFrame(_rows).set_index("scaling")
    return scaling_curves, scaling_table


@app.cell(hide_code=True)
def _(mo, scaling_table):
    mo.vstack([mo.md("### Sweep results"), scaling_table])
    return


@app.cell(hide_code=True)
def _(ORDER, mo, plt, scaling_curves):
    _names = list(scaling_curves)
    _fig, _axes = plt.subplots(
        2, len(_names), figsize=(3.0 * len(_names), 6.2), sharey=True
    )

    for _j, _name in enumerate(_names):
        _x, _co, _cn = scaling_curves[_name]
        for _i, (_c, _lbl, _col) in enumerate(
            [(_co, "FEATHER", "#1f77b4"), (_cn, "FEATHER_new", "#d62728")]
        ):
            _ax = _axes[_i, _j]
            _ax.scatter(_x, _c, s=4, alpha=0.5, color=_col)
            _ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)
            _ax.set_ylim(-1.05, 1.05)
            if _i == 0:
                _ax.set_title(_name, fontsize=9)
            if _i == 1:
                _ax.set_xlabel("scaled feature", fontsize=8)
            if _j == 0:
                _ax.set_ylabel(f"{_lbl}\ncos_mean_r{ORDER}", fontsize=8)
            _ax.tick_params(labelsize=7)

    _fig.suptitle(
        "Embedding value vs scaled feature — structure collapses into noise as θx grows",
        fontsize=11,
    )
    _fig.tight_layout()
    plt.close(_fig)

    mo.vstack(
        [
            mo.md(
                """
                Top row FEATHER, bottom row FEATHER_new, one column per scaling.
                A **smooth curve** means the characteristic function is resolving
                the feature; a **filled band** means the phase has wrapped and the
                feature-to-embedding mapping is effectively random. The left-most
                column is the current pipeline setting.
                """
            ),
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def _(THETA_MAX, mo, scaling_table):
    _best = scaling_table["Moran I FEATHER_new"].astype(float).idxmax()
    _worst = scaling_table["Moran I FEATHER_new"].astype(float).idxmin()

    mo.md(
        f"""
        > Strongest spatial structure at **{_best}**, weakest at **{_worst}**.
        >
        > `min-max` and `z-score` compress $x$ into roughly $[0,1]$ or $[-3,3]$,
        > so with $\\theta \\le {THETA_MAX}$ the largest phase is only a few
        > radians — inside the smooth regime, hence the clean curves.
        >
        > One caveat before standardising: min-max and z-score are **global**
        > transforms, so the embedding becomes dependent on the extent of the
        > study area. Rerun on another city and the same street gets a different
        > embedding. A fixed divisor like `x / 1000` stays comparable across
        > cities.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    | Step | `FEATHER` | `FEATHER_new` | Same code? |
    |---|---|---|---|
    | $A$ | $A_{uv}=1$ | $A_{uv}=1/L_{uv}$ | yes — different graph |
    | $D$ | $d_u=\deg(u)$ | $d^w_u=\sum_v 1/L_{uv}$ | **no** |
    | $D^{-1}$ | $1/\deg(u)$ | $1/\sum_v 1/L_{uv}$, zeros guarded | **no** |
    | $\tilde{A}=D^{-1}A$ | — | — | yes |
    | $\theta$ grid, $\cos/\sin$ lift | — | — | yes |
    | $\tilde{A}^{r}X$ loop | — | — | yes |

    Both models estimate
    $\mathbb{E}_{v\sim P^{r}(u,\cdot)}\!\left[e^{i\theta x_v}\right]$.
    `FEATHER` uses the uniform random walk on the street graph, `FEATHER_new`
    the conductance walk where $w = 1/L$. Every numerical difference follows
    from that one substitution.

    ### Two things to carry forward

    1. **Self-loops.** This notebook strips them in §3. `FEATHER_new` does
       not, so add `G.remove_edges_from(nx.selfloop_edges(G))` after the copy
       in `_create_A_tilde`, or strip them in the pipeline — otherwise those
       rows of $\tilde{A}$ sum to less than 1 on the full city graph.
    2. **The 1 m vs 3 m problem** noted in the `_create_A_tilde` comment.
       Plain $1/L$ treats them as a 3:1 difference, which is rounding noise in
       OSM geometry. An offset fixes it — `1.0 / (length + 25.0)` brings the
       ratio to 1.08:1 while still decaying roughly like $1/L$ at longer
       distances. The offset reads as "crossing a junction costs ~25 m
       regardless of the street". Note it is not scale-free, so the constant
       must be held fixed when comparing cities.
    """)
    return


if __name__ == "__main__":
    app.run()
