"""Pool node-level FEATHER embeddings into one graph vector per city.

The script intentionally does not select r values. Whatever r columns exist in each
embedding CSV are pooled. Therefore input/config.txt is the single source of truth
for r selection via pipeline/FEATHER_embeddings.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline_common import embedding_variant_path, load_config

CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "graph_features"
CHUNK_SIZE = 2_000

EMBEDDING_VARIANTS = (
    "r_categoryAll_evalAll",
    "r_categoryMean_evalAll",
    "r_categoryAll_evalMean",
    "r_categoryMean_evalMean",
)

print("Starting pipeline/graph_feature_calculation.py")


def pool_city(path):
    """Return feature names, population mean and population std for one city."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding CSV: {path}")

    header = pd.read_csv(path, nrows=0)
    if len(header.columns) < 2:
        raise ValueError(f"Embedding CSV has no feature columns: {path}")

    node_id_column = header.columns[0]
    feature_columns = list(header.columns[1:])

    count = 0
    mean = None
    m2 = None

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        numeric = chunk[feature_columns].apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in {path}")
        if len(values) == 0:
            continue

        chunk_count = len(values)
        chunk_mean = values.mean(axis=0)
        centered = values - chunk_mean
        chunk_m2 = np.square(centered).sum(axis=0)

        if count == 0:
            count = chunk_count
            mean = chunk_mean
            m2 = chunk_m2
        else:
            total = count + chunk_count
            delta = chunk_mean - mean
            mean = mean + delta * (chunk_count / total)
            m2 = m2 + chunk_m2 + np.square(delta) * count * chunk_count / total
            count = total

    if count == 0:
        raise ValueError(f"Embedding CSV contains no rows: {path}")

    std = np.sqrt(np.maximum(m2 / count, 0.0))
    return feature_columns, mean, std, count


def calculate_variant(variant, places):
    mean_rows = []
    std_rows = []
    mean_std_rows = []
    reference_columns = None

    for place in places:
        path = embedding_variant_path(place, variant)
        columns, mean, std, node_count = pool_city(path)

        if reference_columns is None:
            reference_columns = columns
        elif columns != reference_columns:
            raise ValueError(
                f"Embedding columns differ between cities for {variant}: {path}"
            )

        mean_row = {"place": place}
        mean_row.update(dict(zip(columns, mean)))
        mean_rows.append(mean_row)

        std_row = {"place": place}
        std_row.update(dict(zip(columns, std)))
        std_rows.append(std_row)

        combined = {"place": place}
        combined.update({f"mean__{column}": value for column, value in zip(columns, mean)})
        combined.update({f"std__{column}": value for column, value in zip(columns, std)})
        mean_std_rows.append(combined)

        print(f"{variant}: pooled {place} ({node_count:,} nodes).", flush=True)

    destination = OUTPUT_ROOT / variant
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mean_rows).to_csv(destination / "mean.csv", index=False)
    pd.DataFrame(std_rows).to_csv(destination / "std.csv", index=False)
    pd.DataFrame(mean_std_rows).to_csv(destination / "mean_std.csv", index=False)

    print(f"Saved graph features: {destination}", flush=True)


def run_pipeline():
    config = load_config(CONFIG_PATH)
    places = list(config["places"])
    if not places:
        raise ValueError("No places configured in input/config.txt.")

    for variant in EMBEDDING_VARIANTS:
        calculate_variant(variant, places)

    print("\nAll graph-level feature files are ready.", flush=True)


if __name__ == "__main__":
    run_pipeline()
