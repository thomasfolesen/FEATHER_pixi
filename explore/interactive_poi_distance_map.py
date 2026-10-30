import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pickle
    from pathlib import Path
    import plotly.graph_objects as go

    return Path, go, mo, pickle


@app.cell
def _(Path):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MAP_DATA_DIR = PROJECT_ROOT / "output" / "interactive_map_data"

    map_files = {
        path.stem: path
        for path in sorted(MAP_DATA_DIR.glob("*.pkl"))
    }

    return MAP_DATA_DIR, map_files


@app.cell
def _(MAP_DATA_DIR, map_files, mo):
    mo.stop(
        not map_files,
        mo.md(
            f"No saved map data found in `{MAP_DATA_DIR}`.\n\n"
            "Run the map-data saving cell in your pipeline first."
        ),
    )

    city_selector = mo.ui.dropdown(
        options=list(map_files),
        value=next(iter(map_files)),
        label="City",
    )

    mo.vstack([
        mo.md("# Interactive POI distance map"),
        city_selector,
    ])

    return (city_selector,)


@app.cell
def _(city_selector, map_files, pickle):
    with map_files[city_selector.value].open("rb") as _file:
        city_map = pickle.load(_file)

    return (city_map,)


@app.cell
def _(city_map, mo):
    category_selector = mo.ui.dropdown(
        options=list(city_map["distances"].columns),
        value=city_map["distances"].columns[0],
        label="POI category",
    )

    category_selector
    return (category_selector,)


@app.cell
def _(city_map):
    # Prepare roads only when the selected city changes.
    road_x = []
    road_y = []

    for _geometry in city_map["edges"].geometry:
        if _geometry is None or _geometry.is_empty:
            continue

        _lines = (
            list(_geometry.geoms)
            if _geometry.geom_type == "MultiLineString"
            else [_geometry]
        )

        for _line in _lines:
            if _line.geom_type != "LineString":
                continue

            _x, _y = _line.xy
            road_x.extend(list(_x) + [None])
            road_y.extend(list(_y) + [None])

    return road_x, road_y


@app.cell
def _(category_selector, city_map, go, mo, road_x, road_y):
    _place = city_map["place"]
    _category = category_selector.value
    _category_name = _category.replace("_", " ")
    _limit = city_map["CLIP_BUFFER_M"]
    _nearest = city_map["NEAREST_POIS"]

    _nodes = city_map["nodes"].copy()
    _nodes["distance"] = city_map["distances"][_category].reindex(
        _nodes.index
    )

    _figure = go.Figure()

    _figure.add_trace(
        go.Scattergl(
            x=road_x,
            y=road_y,
            mode="lines",
            line=dict(color="#dddddd", width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    _missing = _nodes[_nodes["distance"].isna()]
    _available = _nodes[_nodes["distance"].notna()]

    if not _missing.empty:
        _figure.add_trace(
            go.Scattergl(
                x=_missing.geometry.x,
                y=_missing.geometry.y,
                mode="markers",
                marker=dict(size=5, color="#999999"),
                name="No reachable POI",
                hovertemplate=(
                    f"<b>{_place}</b><br>"
                    f"Category: {_category_name}<br>"
                    f"No reachable POI within {_limit:,} m"
                    "<extra></extra>"
                ),
            )
        )

    if not _available.empty:
        _figure.add_trace(
            go.Scattergl(
                x=_available.geometry.x,
                y=_available.geometry.y,
                mode="markers",
                customdata=_available["distance"].to_numpy(),
                marker=dict(
                    size=5,
                    color=_available["distance"],
                    colorscale="Viridis",
                    cmin=0,
                    cmax=_limit,
                    colorbar=dict(
                        title=dict(text="Mean walking<br>distance (m)")
                    ),
                ),
                showlegend=False,
                hovertemplate=(
                    f"<b>{_place}</b><br>"
                    f"Category: {_category_name}<br>"
                    "Mean walking distance: %{customdata:.1f} m"
                    "<extra></extra>"
                ),
            )
        )

    _figure.update_layout(
        title=dict(
            text=(
                f"{_place}<br>"
                f"<sup>Mean distance to up to {_nearest} nearest "
                f"{_category_name} POIs</sup>"
            )
        ),
        template="plotly_white",
        height=750,
        margin=dict(l=10, r=20, t=90, b=50),
        dragmode="pan",
        hovermode="closest",
        uirevision=_place,
        xaxis=dict(visible=False),
        yaxis=dict(
            visible=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(
            orientation="h",
            y=-0.02,
            x=0,
        ),
    )

    mo.vstack([
        mo.ui.plotly(
            _figure,
            config={
                "scrollZoom": True,
                "displaylogo": False,
            },
        ),
        mo.md(
            f"Mean over up to **{_nearest} POIs** reachable within "
            f"**{_limit:,} metres**. Unavailable neighbours are excluded. "
            "Grey nodes have no reachable POI within this limit.\n\n"
            "**Hover** for distances, **scroll** to zoom, "
            "and **drag** to pan."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()