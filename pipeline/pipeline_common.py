"""Shared configuration, output paths and CSV schema."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "output"


def parse_config_list(value):
    """Parse nested config values such as ``[[item], [item]]``."""
    return [
        item.strip()
        for item in re.findall(r"\[([^\[\]]+)\]", value)
    ]


def parse_order_list(value):
    """Parse selected FEATHER orders, for example ``[[1], [7]]``."""
    entries = parse_config_list(value)

    # Also tolerate a single bracketed comma-separated value such as [1, 7].
    if len(entries) == 1 and "," in entries[0]:
        entries = [item.strip() for item in entries[0].split(",")]

    if not entries:
        raise ValueError(
            "orders must contain at least one integer, for example "
            "orders = [[1], [7]]"
        )

    try:
        orders = [int(item) for item in entries]
    except ValueError as error:
        raise ValueError(
            "orders must contain integers, for example orders = [[1], [7]]"
        ) from error

    if any(order <= 0 for order in orders):
        raise ValueError("Every FEATHER order must be a positive integer.")

    # FEATHER naturally produces orders in ascending order. Remove duplicates.
    return sorted(set(orders))


def load_config(path):
    """Load parameters and amenity categories from the config file."""
    config = {
        "resolved_categories": {},
        "outlier_iqr_multiplier": 1.5,
    }

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Places
        if key == "place":
            config["places"] = parse_config_list(value)

        # Selected FEATHER orders, e.g. orders = [[1], [7]]
        elif key == "orders":
            config["orders"] = parse_order_list(value)

        # Float values
        elif key in (
            "theta_max",
            "high_low_percent",
            "outlier_iqr_multiplier",
        ):
            config[key] = float(value)

        # Integer values. ``order`` remains accepted for old config files.
        elif key in (
            "eval_points",
            "order",
            "CLIP_BUFFER_M",
            "NEAREST_POIS",
        ):
            config[key] = int(value)

        # Amenity categories
        else:
            tags = []
            for entry in parse_config_list(value):
                osm_key, osm_tag = entry.split("=", 1)
                tags.append(
                    {
                        "osm_key": osm_key.strip(),
                        "osm_tag": osm_tag.strip(),
                    }
                )

            config["resolved_categories"][key] = tags

    if "orders" in config and "order" in config:
        raise ValueError(
            "Use either 'orders = [[1], [7]]' or the legacy 'order = 7', "
            "not both."
        )

    # Backwards compatibility: old `order = 7` meant save r1 through r7.
    if "orders" not in config:
        if "order" not in config:
            raise ValueError(
                "Missing FEATHER orders. Add a line such as "
                "orders = [[1], [7]] to input/config.txt."
            )
        if config["order"] <= 0:
            raise ValueError("order must be a positive integer.")
        config["orders"] = list(range(1, config["order"] + 1))

    config["orders"] = sorted(set(config["orders"]))
    config["max_order"] = max(config["orders"])

    # Backwards-compatible alias for code that only needs the largest r value.
    # New code that decides which orders to save should use config["orders"].
    config["order"] = config["max_order"]

    multiplier = config["outlier_iqr_multiplier"]
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(
            "outlier_iqr_multiplier must be finite and positive."
        )

    return config


def safe_filename(place):
    """Make sure names of files and folders are safe to use."""
    return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()


def embedding_columns(categories, orders, eval_points):
    """Return FEATHER column names for the requested random-walk orders.

    ``orders`` may be an iterable such as ``[1, 7]``. An integer remains
    supported for older callers and means every order from 1 through that value.
    """
    if isinstance(orders, (int, np.integer)):
        order_values = list(range(1, int(orders) + 1))
    else:
        order_values = list(orders)

    return [
        f"r{r}_{category}_eval{point}_{component}"
        for r in order_values
        for component in ("real", "imaginary")
        for category in categories
        for point in range(1, eval_points + 1)
    ]


def output_paths(place):
    name = safe_filename(place)
    return (
        OUTPUT_ROOT
        / "FEATHER_embeddings"
        / name
        / f"{name}_feather_embeddings.csv",
        OUTPUT_ROOT
        / "POI_distances"
        / name
        / f"{name}_poi_distances.csv",
    )


def read_node_csv(path):
    table = pd.read_csv(path, dtype={"node_id": str}, index_col="node_id")
    if not table.index.is_unique:
        raise ValueError(f"Duplicate node IDs: {path}")
    return table


def read_metadata(embedding_path):
    path = embedding_path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run the refactored pipeline first."
        )
    return json.loads(path.read_text(encoding="utf-8"))
