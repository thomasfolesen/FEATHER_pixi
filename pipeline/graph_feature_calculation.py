"""Calculate graph features: mean and std of every embedding column per city."""
# Save as pipeline/graph_feature_calculation.py and run from the project root:
# pixi run python pipeline/graph_feature_calculation.py
# Run pipeline/geofabrik_final.py first if embedding files are missing.
# Reads every city folder in output/FEATHER_embeddings/ and writes
# output/FEATHER_patterns/graph_features.csv (one row per place) plus
# output/FEATHER_patterns/graph_features_info.json (vertex counts, skipped files).

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
EMBEDDING_ROOT = PROJECT_ROOT / "output" / "FEATHER_embeddings"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "FEATHER_patterns"
GRAPH_FEATURES_PATH = OUTPUT_ROOT / "graph_features.csv"
GRAPH_FEATURES_INFO_PATH = OUTPUT_ROOT / "graph_features_info.json"


def safe_filename(place):
    return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()


def load_config():
    settings = {}
    for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    places = [value.strip() for value in re.findall(r"\[([^\[\]]+)\]", settings["place"])]
    categories = [key for key, value in settings.items() if key != "place" and value.startswith("[[")]
    order, points = int(settings["order"]), int(settings["eval_points"])
    if not places or not categories or min(order, points) < 1:
        raise ValueError("Configure places, categories, and positive order/eval_points.")
    names = [safe_filename(place) for place in places]
    if len(set(names)) != len(names):
        raise ValueError("Some places produce the same filename; remove duplicate places.")
    columns = [
        f"r{r}_{category}_eval{point}_{component}"
        for r in range(1, order + 1)
        for component in ("real", "imaginary")
        for category in categories
        for point in range(1, points + 1)
    ]
    return settings, places, columns


def feature_names(columns):
    return [f"{stat}__{column}" for stat in ("mean", "std") for column in columns]


def summarise_graph(path, columns):
    """Read one chunk at a time and combine its mean and variance."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if set(header) != {"node_id", *columns} or len(header) != len(columns) + 1:
        raise ValueError(f"{path.name}: embedding columns do not match config.txt. Rerun the pipeline.")
    count = 0
    mean = np.zeros(len(columns))
    squared_deviations = np.zeros(len(columns))
    # Allow space for several temporary arrays, even with many embedding columns.
    chunk_size = max(1, min(2000, 16_000_000 // (8 * len(columns) * 4)))
    for chunk in pd.read_csv(path, usecols=columns, dtype=float, chunksize=chunk_size):
        values = chunk[columns].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: embeddings contain missing or infinite values.")
        size = len(values)
        if not size:
            continue
        chunk_mean = values.mean(axis=0)
        chunk_squared = ((values - chunk_mean) ** 2).sum(axis=0)
        delta = chunk_mean - mean
        total = count + size
        squared_deviations += chunk_squared + delta ** 2 * (count * size / total)
        mean += delta * (size / total)
        count = total
    if not count:
        raise ValueError(f"{path.name}: no vertices found.")
    std = np.sqrt(np.maximum(squared_deviations / count, 0))
    return np.concatenate([mean, std]), count


def summarise_graphs(paths, place_names, columns):
    """Summarise every embedding file, skipping files that fail validation."""
    summaries, counts, skipped = {}, {}, []
    for path in paths:
        folder_name = path.parent.name
        place = place_names.get(folder_name, folder_name.replace("_", " ").title())
        try:
            print(f"Reading {place}...", flush=True)
            summaries[place], counts[place] = summarise_graph(path, columns)
        except ValueError as error:
            print(f"Skipping {place}: {error}")
            skipped.append({"place": place, "file": str(path), "reason": str(error)})
    return summaries, counts, skipped


def build_graph_features(summaries, columns):
    """Turn per-place summaries into the graph_features table (one row per place)."""
    features = pd.DataFrame.from_dict(summaries, orient="index", columns=feature_names(columns))
    features.index.name = "place"
    return features


def main():
    settings, places, columns = load_config()
    paths = sorted(EMBEDDING_ROOT.glob("*/*_feather_embeddings.csv"))
    if not paths:
        raise ValueError(f"No saved embeddings found in {EMBEDDING_ROOT}. Run the pipeline first.")

    place_names = {safe_filename(place): place for place in places}
    print(f"Found {len(paths)} saved embedding files.", flush=True)
    summaries, counts, skipped = summarise_graphs(paths, place_names, columns)
    if not summaries:
        raise ValueError("No usable embedding files; graph_features.csv was not written.")

    features = build_graph_features(summaries, columns)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    features.to_csv(GRAPH_FEATURES_PATH)
    info = {"vertices": counts, "skipped_files": skipped, "embedding_columns": columns}
    GRAPH_FEATURES_INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved graph features for {len(features)} places to {GRAPH_FEATURES_PATH}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
"""Graph feature calculation: mean and std of every embedding column per city."""

import numpy as np
import pandas as pd


def summarise_graph(path, columns):
    """Read one chunk at a time and combine its mean and variance."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if set(header) != {"node_id", *columns} or len(header) != len(columns) + 1:
        raise ValueError(f"{path.name}: embedding columns do not match config.txt. Rerun the pipeline.")
    count = 0
    mean = np.zeros(len(columns))
    squared_deviations = np.zeros(len(columns))
    # Allow space for several temporary arrays, even with many embedding columns.
    chunk_size = max(1, min(2000, 16_000_000 // (8 * len(columns) * 4)))
    for chunk in pd.read_csv(path, usecols=columns, dtype=float, chunksize=chunk_size):
        values = chunk[columns].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: embeddings contain missing or infinite values.")
        size = len(values)
        if not size:
            continue
        chunk_mean = values.mean(axis=0)
        chunk_squared = ((values - chunk_mean) ** 2).sum(axis=0)
        delta = chunk_mean - mean
        total = count + size
        squared_deviations += chunk_squared + delta ** 2 * (count * size / total)
        mean += delta * (size / total)
        count = total
    if not count:
        raise ValueError(f"{path.name}: no vertices found.")
    std = np.sqrt(np.maximum(squared_deviations / count, 0))
    return np.concatenate([mean, std]), count


def summarise_graphs(paths, place_names, columns):
    """Summarise every embedding file, skipping files that fail validation."""
    summaries, counts, skipped = {}, {}, []
    for path in paths:
        folder_name = path.parent.name
        place = place_names.get(folder_name, folder_name.replace("_", " ").title())
        try:
            print(f"Reading {place}...", flush=True)
            summaries[place], counts[place] = summarise_graph(path, columns)
        except ValueError as error:
            print(f"Skipping {place}: {error}")
            skipped.append({"place": place, "file": str(path), "reason": str(error)})
    return summaries, counts, skipped


def build_graph_features(summaries, columns):
    """Turn per-place summaries into the graph_features table (one row per place)."""
    names = [f"{stat}__{column}" for stat in ("mean", "std") for column in columns]
    features = pd.DataFrame.from_dict(summaries, orient="index", columns=names)
    features.index.name = "place"
    return features