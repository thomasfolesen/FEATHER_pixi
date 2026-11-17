"""Shared configuration, output paths and CSV schema"""
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
    """parse items like places and ammenitie OSM types in the form of

        place = [[Copenhagen Municipality, Denmark], [Polla, Salerno, Campania, Italy]]

    to something more useable like

        ["Copenhagen Municipality, Denmark", "Polla, Salerno, Campania, Italy",]

    """
    return [
        item.strip()
        for item in re.findall(r"\[([^\[\]]+)\]", value)
    ]

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

        # Float values
        elif key in (
            "theta_max",
            "high_low_percent",
            "outlier_iqr_multiplier",
        ):
            config[key] = float(value)

        # Integer values
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
                tags.append({
                    "osm_key": osm_key.strip(),
                    "osm_tag": osm_tag.strip(),
                })

            config["resolved_categories"][key] = tags

    multiplier = config["outlier_iqr_multiplier"]
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(
            "outlier_iqr_multiplier must be finite and positive."
        )

    return config

def safe_filename(place):
    """

    make sure names of file and folders are safe to use

    """
    return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()

def embedding_columns(categories, order, eval_points):
    # Same order as FEATHER: r, component, category, evaluation point.
    return [
        f"r{r}_{category}_eval{point}_{component}"
        for r in range(1, order + 1)
        for component in ("real", "imaginary")
        for category in categories
        for point in range(1, eval_points + 1)
    ]

def output_paths(place):
    name = safe_filename(place)
    return (
        OUTPUT_ROOT / "FEATHER_embeddings" / name / f"{name}_feather_embeddings.csv",
        OUTPUT_ROOT / "POI_distances" / name / f"{name}_poi_distances.csv",
    )


def read_node_csv(path):
    table = pd.read_csv(path, dtype={"node_id": str}, index_col="node_id")
    if not table.index.is_unique:
        raise ValueError(f"Duplicate node IDs: {path}")
    return table


def read_metadata(embedding_path):
    path = embedding_path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run the refactored pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


