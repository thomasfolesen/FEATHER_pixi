import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # FEATHER vs FEATHER_new — tiny synthetic A–E graph, by-hand math check

    No OSM / Overpass, no cached city graph, no amenity lookup — just a
    hand-built 5-node graph so every matrix and embedding value can be
    checked against pen-and-paper math.

    **Graph (edge: length):**

    - A-B: 1
    - B-E: 3
    - E-D: 3
    - A-C: 2
    - C-D: 1
    - A-D: 5

    `D` is treated as the "amenity" / destination node.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt

    from src.feather import FEATHER, FEATHER_new

    return FEATHER, FEATHER_new, mo, np, nx, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the synthetic graph

    Nodes are relabeled `0..n-1` (FEATHER / FEATHER_new both expect
    `nx.adjacency_matrix(graph, nodelist=range(n))`), but every table
    below is shown with the original A–E letters as row/column labels
    so it stays easy to read.
    """)
    return


@app.cell
def _(nx):
    edges = [
        ("A", "B", 1),
        ("B", "E", 3),
        ("E", "D", 3),
        ("A", "C", 2),
        ("C", "D", 1),
        ("A", "D", 5),
    ]

    G_named = nx.Graph()
    G_named.add_weighted_edges_from(edges, weight="length")

    node_order = ["A", "B", "C", "D", "E"]
    node_mapping = {name: i for i, name in enumerate(node_order)}
    inv_mapping = {i: name for name, i in node_mapping.items()}

    graph = nx.relabel_nodes(G_named, node_mapping, copy=True)
    n_nodes = graph.number_of_nodes()
    labels = [inv_mapping[i] for i in range(n_nodes)]
    return graph, inv_mapping, labels, n_nodes, node_mapping


@app.cell
def _(graph, inv_mapping, mo, pd):
    edge_rows = [
        {"u": inv_mapping[u], "v": inv_mapping[v], "length": d["length"]}
        for u, v, d in graph.edges(data=True)
    ]
    edge_table = pd.DataFrame(edge_rows)
    mo.vstack([mo.md(f"**{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges:**"), edge_table])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature vector X

    `D` is the amenity / end node — a natural, easy-to-hand-check feature
    is each node's shortest-path distance to `D` (weighted by `length`).
    Swap this out for whatever feature you're actually verifying.
    """)
    return


@app.cell
def _(graph, n_nodes, node_mapping, np, nx):
    dist_to_D = nx.shortest_path_length(graph, target=node_mapping["D"], weight="length")
    X = np.array([dist_to_D[i] for i in range(n_nodes)], dtype=float)
    return (X,)


@app.cell
def _(X, inv_mapping, mo, n_nodes, pd):
    X_df = pd.DataFrame(
        {"distance_to_D": X},
        index=[inv_mapping[i] for i in range(n_nodes)],
    )
    mo.vstack([mo.md("**Feature vector X (shortest-path distance to D):**"), X_df])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameters
    """)
    return


@app.cell
def _(mo):
    # Defaults (eval_points=5, order=2, theta_max=2.5) match the values used
    # in the uploaded FEATHER_new_step_by_step.xlsx, so the numbers and
    # charts below line up exactly with that spreadsheet out of the box.
    eval_points_slider = mo.ui.slider(1, 25, value=5, label="eval_points")
    order_slider = mo.ui.slider(1, 6, value=2, label="order")
    theta_max_slider = mo.ui.slider(0.5, 5.0, value=2.5, step=0.5, label="theta_max")
    mo.hstack([eval_points_slider, order_slider, theta_max_slider])
    return eval_points_slider, order_slider, theta_max_slider


@app.cell
def _(eval_points_slider, order_slider, theta_max_slider):
    eval_points = eval_points_slider.value
    order = order_slider.value
    theta_max = theta_max_slider.value
    return eval_points, order, theta_max


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FEATHER (unweighted — every edge counts equally, length ignored)
    """)
    return


@app.cell
def _(
    FEATHER,
    X,
    eval_points,
    graph,
    labels,
    n_nodes,
    nx,
    order,
    pd,
    theta_max,
):
    feather_model = FEATHER(theta_max=theta_max, eval_points=eval_points, order=order)

    A = nx.adjacency_matrix(graph, nodelist=range(n_nodes))
    D_inv = feather_model._create_D_inverse(graph)
    A_tilde = feather_model._create_A_tilde(graph)

    feather_model.fit(graph, X)
    feather_embedding = feather_model.get_embedding()

    A_df = pd.DataFrame(A.toarray(), index=labels, columns=labels)
    D_inv_df = pd.DataFrame(D_inv.toarray(), index=labels, columns=labels).round(4)
    A_tilde_df = pd.DataFrame(A_tilde.toarray(), index=labels, columns=labels).round(4)
    feather_embedding_df = pd.DataFrame(feather_embedding, index=labels).round(4)
    return A_tilde_df, D_inv_df, feather_embedding, feather_embedding_df


@app.cell
def _(A_tilde_df, D_inv_df, mo):
    mo.vstack(
        [
            mo.md("**A_tilde = D_inverse @ A** (random-walk transition probabilities, length ignored):"),
            A_tilde_df,
            mo.md("**D_inverse** (1 / node degree):"),
            D_inv_df,
        ]
    )
    return


@app.cell
def _(feather_embedding_df, mo):
    mo.vstack([mo.md("**FEATHER embedding** (cos blocks then sin blocks, per order r=1..order):"), feather_embedding_df])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## FEATHER_new (weighted, w = 1/length)

    **A quirk worth knowing about:** `FEATHER_new._create_D_inverse` reads
    the edge attribute literally named `"weight"`. Our graph only carries
    `"length"`, so calling `_create_D_inverse(graph)` directly on the raw
    graph silently falls back to unweighted degree (every edge counts as
    1) instead of summing `1/length`. `_create_A_tilde` avoids this
    internally by building a local copy with `weight = 1/length` set
    *before* calling `_create_D_inverse` on it — so `A_tilde_w` itself is
    correct, but a standalone `D_inverse_w` printed from the raw graph
    would **not** match it. To show the `D_inverse_w` that is actually
    consistent with `A_tilde_w` (and with hand math), we replicate that
    same `weight = 1/length` assignment below before calling
    `_create_D_inverse`.
    """)
    return


@app.cell
def _(FEATHER_new, X, eval_points, graph, labels, order, pd, theta_max):
    feather_new_model = FEATHER_new(theta_max=theta_max, eval_points=eval_points, order=order)

    graph_w = graph.copy()
    for _, _, _d in graph_w.edges(data=True):
        _length = max(float(_d.get("length", 1.0)), 1.0)  # same floor as _create_A_tilde
        _d["weight"] = 1.0 / _length

    D_inv_w = feather_new_model._create_D_inverse(graph_w)
    A_tilde_w = feather_new_model._create_A_tilde(graph)  # builds its own weighted copy internally

    feather_new_model.fit(graph, X)
    feather_new_embedding = feather_new_model.get_embedding()

    D_inv_w_df = pd.DataFrame(D_inv_w.toarray(), index=labels, columns=labels).round(4)
    A_tilde_w_df = pd.DataFrame(A_tilde_w.toarray(), index=labels, columns=labels).round(4)
    feather_new_embedding_df = pd.DataFrame(feather_new_embedding, index=labels).round(4)
    return (
        A_tilde_w,
        A_tilde_w_df,
        D_inv_w_df,
        feather_new_embedding,
        feather_new_embedding_df,
    )


@app.cell
def _(A_tilde_w_df, D_inv_w_df, mo):
    mo.vstack(
        [
            mo.md("**A_tilde_w = D_inverse_w @ A(1/length)** (shorter edges pull visibly more weight):"),
            A_tilde_w_df,
            mo.md("**D_inverse_w** (1 / sum of 1/length at each node):"),
            D_inv_w_df,
        ]
    )
    return


@app.cell
def _(feather_new_embedding_df, mo):
    mo.vstack([mo.md("**FEATHER_new embedding:**"), feather_new_embedding_df])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing FEATHER_new across theta

    This reproduces the chart your partner built in Excel: x-axis is the
    characteristic function evaluation point (theta), y-axis is the
    embedding value at that point. Each node gets its own curve set — one
    line per (scale, real/imaginary) combination — plus one more chart
    for the unweighted average across all five nodes.

    Blue = real part, orange = imaginary part. Line style marks the
    random-walk scale: solid = scale 1, dashed = scale 2, dotted = scale
    3, and so on if you push `order` higher with the sliders above.
    """)
    return


@app.cell
def _(eval_points, np, theta_max):
    # Same formula feather.py uses internally in `fit()`, so this lines up
    # exactly with the theta values each embedding column was evaluated at.
    theta = np.linspace(0.01, theta_max, eval_points)
    return (theta,)


@app.cell
def _(eval_points, order):
    _linestyles = ["-", "--", ":", "-."]

    def curve_blocks(embedding_row):
        """
        Split one embedding row into (scale, real_values, imag_values,
        linestyle) tuples, matching the column layout feather.py produces:
        for each scale r = 1..order, `eval_points` real (cos) columns
        followed by `eval_points` imaginary (sin) columns, scales
        concatenated in order.
        """
        blocks = []
        for r in range(1, order + 1):
            start = (r - 1) * 2 * eval_points
            real_vals = embedding_row[start : start + eval_points]
            imag_vals = embedding_row[start + eval_points : start + 2 * eval_points]
            linestyle = _linestyles[(r - 1) % len(_linestyles)]
            blocks.append((r, real_vals, imag_vals, linestyle))
        return blocks

    return (curve_blocks,)


@app.cell
def _(curve_blocks, theta):
    def plot_node_curves(ax, embedding_row, title):
        for r, real_vals, imag_vals, linestyle in curve_blocks(embedding_row):
            ax.plot(theta, real_vals, color="tab:blue", linestyle=linestyle, marker="o", markersize=3, label=f"s{r} Re")
            ax.plot(theta, imag_vals, color="tab:orange", linestyle=linestyle, marker="o", markersize=3, label=f"s{r} Im")
        ax.set_title(title)
        ax.set_xlabel("theta")
        ax.set_ylabel("value")
        ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
        ax.legend(fontsize=7)

    return (plot_node_curves,)


@app.cell
def _(feather_new_embedding, labels, plot_node_curves, plt):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for _i, _label in enumerate(labels):
        plot_node_curves(axes[_i], feather_new_embedding[_i], f"Node {_label}")

    # Hide any unused subplot slots, in case there are ever fewer than 5 nodes
    for _j in range(len(labels), len(axes)):
        axes[_j].axis("off")

    fig.suptitle("FEATHER_new embedding values vs. theta, per node")
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Average chart, styled after Figure 1 of the FEATHER paper

    This redoes just the "Average across nodes" chart to look like the
    paper's Figure 1 — "the real part of ... mean characteristic
    functions with standard deviations around the mean": one line per
    scale (the real-part mean across the 5 nodes) with a shaded ± 1
    standard deviation band around it, boxed axes, dashed gridlines, and
    a bordered legend underneath the plot. The per-node panels above keep
    their original look — only this chart is restyled.

    The curve itself is **not** just the 5 stored evaluation points
    joined by straight lines. `θ` is a continuous real-valued input to
    cos/sin, so the characteristic function is a genuine smooth wave —
    we re-evaluate the same scale-r random-walk weighted formula
    (`A_tilde^r · cos(x ⊗ θ)`) at hundreds of closely spaced θ values to
    draw the actual curve, then mark the handful of θ values that are
    actually stored in the embedding / Excel file as dots sitting right
    on top of it.
    """)
    return


@app.cell
def _(np):
    def dense_characteristic_curves(A_tilde, X, order, thetas):
        """
        Evaluate the real (cos) and imaginary (sin) parts of the r-scale
        random-walk weighted characteristic function at an arbitrary,
        densely sampled set of theta values. Same recursion feather.py
        uses internally (apply A_tilde once per scale, cos/sin kept
        separate instead of concatenated -- equivalent since A_tilde
        acts column-wise), just evaluated at many thetas instead of only
        `eval_points` of them, so the curve comes out as a smooth wave
        instead of a few points joined by straight lines.
        """
        cos_x = np.cos(np.outer(X, thetas))  # (n_nodes, len(thetas))
        sin_x = np.sin(np.outer(X, thetas))
        reals, imags = [], []
        Hc, Hs = cos_x, sin_x
        for _ in range(order):
            Hc = A_tilde.dot(Hc)
            Hs = A_tilde.dot(Hs)
            reals.append(Hc)
            imags.append(Hs)
        return reals, imags

    return (dense_characteristic_curves,)


@app.cell
def _(A_tilde_w, X, dense_characteristic_curves, np, order, theta_max):
    # Hundreds of closely spaced theta values instead of just `eval_points`
    # of them -- this is what actually makes the curve look like a smooth
    # wave rather than a jagged line between a handful of dots.
    theta_dense = np.linspace(0.01, theta_max, 25)
    real_dense_blocks, imag_dense_blocks = dense_characteristic_curves(
        A_tilde_w, X, order, theta_dense
    )
    return real_dense_blocks, theta_dense


@app.cell
def _(eval_points, feather_new_embedding, order, real_dense_blocks):
    # Real part only (matches what the paper figure actually plots), one
    # mean + std curve per scale, computed across the 5 nodes.
    real_mean_dense = [block.mean(axis=0) for block in real_dense_blocks]
    real_std_dense = [block.std(axis=0) for block in real_dense_blocks]

    # Same thing at just the handful of theta values actually stored in
    # the embedding / Excel file, so we can mark them on top of the
    # smooth curve as a sanity check that both agree.
    real_mean_sampled = []
    for _r in range(1, order + 1):
        _start = (_r - 1) * 2 * eval_points
        _real_block = feather_new_embedding[:, _start : _start + eval_points]  # (n_nodes, eval_points)
        real_mean_sampled.append(_real_block.mean(axis=0))
    return real_mean_dense, real_mean_sampled, real_std_dense


@app.cell
def _(
    order,
    plt,
    real_mean_dense,
    real_mean_sampled,
    real_std_dense,
    theta,
    theta_dense,
):
    _colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:cyan"]

    fig_avg, ax_avg = plt.subplots(figsize=(7, 5))

    for _r in range(1, order + 1):
        _color = _colors[(_r - 1) % len(_colors)]
        _mean = real_mean_dense[_r - 1]
        _std = real_std_dense[_r - 1]
        ax_avg.plot(theta_dense, _mean, color=_color, linewidth=1.5, label=f"Walk {_r}")
        ax_avg.fill_between(theta_dense, _mean - _std, _mean + _std, color=_color, alpha=0.25)
        # The handful of points actually stored in the embedding / Excel,
        # shown sitting exactly on the smooth curve.
        ax_avg.scatter(
            theta, real_mean_sampled[_r - 1], color=_color, s=20, zorder=5,
            edgecolor="white", linewidth=0.6,
        )

    ax_avg.set_xlabel("Evaluation point")
    ax_avg.set_ylabel("Characteristic function value")
    ax_avg.set_ylim(-1.05, 1.05)
    ax_avg.set_xlim(theta_dense[0], theta_dense[-1])
    ax_avg.grid(True, linestyle="--", alpha=0.5)
    ax_avg.axhline(0, color="black", linewidth=0.8)
    ax_avg.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=order, frameon=True)

    fig_avg.tight_layout()
    fig_avg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Difference (FEATHER_new − FEATHER)
    """)
    return


@app.cell
def _(
    feather_embedding,
    feather_embedding_df,
    feather_new_embedding,
    mo,
    np,
    pd,
):
    diff = feather_new_embedding - feather_embedding
    diff_df = pd.DataFrame(
        diff, index=feather_embedding_df.index, columns=feather_embedding_df.columns
    ).round(4)

    per_node_mean_abs_diff = pd.DataFrame(
        {"mean_abs_diff": np.abs(diff).mean(axis=1)},
        index=feather_embedding_df.index,
    ).round(4)

    mo.vstack(
        [
            mo.md(f"**Max abs difference:** {round(float(np.abs(diff).max()), 4)}"),
            mo.md("**Per-node mean abs difference:**"),
            per_node_mean_abs_diff,
            mo.md("**Full difference matrix:**"),
            diff_df,
        ]
    )
    return


if __name__ == "__main__":
    app.run()