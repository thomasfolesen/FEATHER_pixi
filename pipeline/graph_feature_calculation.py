"""Reduce node-level FEATHER embeddings to one feature vector per city.

Run from the project root after the node-level embedding files exist:

    pixi run python pipeline/graph_feature_calculation.py

The node-level embedding representations and graph reductions are configured in
``input/variants.txt``. For each embedding representation this script reads every
city embedding once and calculates, for every selected embedding column:

    mean      population mean across nodes
    std       population standard deviation across nodes (ddof=0)
    mean_std  mean and std concatenated

All FEATHER orders configured in ``input/config.txt`` are used together. Extra
orders in an old embedding CSV are ignored, but every configured order must be
present.

Outputs:

    output/graph_features/<embedding>/mean.csv
    output/graph_features/<embedding>/std.csv
    output/graph_features/<embedding>/mean_std.csv

Each output has one row per city/graph.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline_common import (
    CONFIG_PATH,
    EMBEDDING_VARIANTS,
    embedding_variant_path,
    load_config,
)

VARIANTS_PATH = PROJECT_ROOT / "input" / "variants.txt"
GRAPH_FEATURE_ROOT = PROJECT_ROOT / "output" / "graph_features"
GRAPH_REDUCTIONS = ("mean", "std", "mean_std")
CHUNK_SIZE = 2_000
R_COLUMN = re.compile(r"^r(\d+)_")


def comma_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_variants_file(path=VARIANTS_PATH):
    """Read embedding representations and graph reductions from variants.txt."""
    values = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"Invalid line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    embeddings = comma_list(
        values.get("embeddings", ", ".join(EMBEDDING_VARIANTS))
    )
    reductions = comma_list(
        values.get("graph_reductions", ", ".join(GRAPH_REDUCTIONS))
    )

    unknown_embeddings = [
        value for value in embeddings if value not in EMBEDDING_VARIANTS
    ]
    if unknown_embeddings:
        raise ValueError(
            "Unknown embedding variant(s): "
            + ", ".join(unknown_embeddings)
            + ". Expected: "
            + ", ".join(EMBEDDING_VARIANTS)
        )

    unknown_reductions = [
        value for value in reductions if value not in GRAPH_REDUCTIONS
    ]
    if unknown_reductions:
        raise ValueError(
            "Unknown graph_reductions value(s): "
            + ", ".join(unknown_reductions)
            + ". Expected: mean, std, mean_std."
        )

    if not embeddings:
        raise ValueError("Configure at least one embedding in input/variants.txt.")
    if not reductions:
        raise ValueError(
            "Configure at least one graph_reduction in input/variants.txt."
        )

    return embeddings, reductions


def r_value(column):
    match = R_COLUMN.match(column)
    return int(match.group(1)) if match else None


def selected_header(path, configured_orders):
    """Return node-ID column and embedding columns for configured r-values."""
    if not path.is_file():
        raise FileNotFoundError(path)

    header = pd.read_csv(path, nrows=0)
    if header.columns.empty:
        raise ValueError(f"Embedding CSV has no columns: {path}")

    node_id_column = header.columns[0]
    allowed = set(configured_orders)
    columns = [
        column
        for column in header.columns[1:]
        if r_value(column) in allowed
    ]
    present_orders = sorted({r_value(column) for column in columns})

    if not columns:
        raise ValueError(
            f"No embedding columns in {path} match configured orders "
            f"{list(configured_orders)}."
        )

    missing_orders = sorted(allowed.difference(present_orders))
    if missing_orders:
        raise ValueError(
            f"Embedding CSV is missing configured r-values {missing_orders}: {path}\n"
            "Regenerate the embedding from the current input/config.txt."
        )

    if len(columns) != len(set(columns)):
        raise ValueError(f"Duplicate embedding columns: {path}")

    return node_id_column, columns, present_orders


def summarize_city(path, configured_orders, expected_columns=None):
    """Calculate population mean/std for one city without loading all nodes."""
    node_id_column, columns, present_orders = selected_header(
        path,
        configured_orders,
    )

    if expected_columns is not None and columns != expected_columns:
        raise ValueError(
            "Embedding columns differ between cities for the same variant.\n"
            f"File: {path}\n"
            "Regenerate all city embeddings/variants from the same config."
        )

    count = 0
    mean = np.zeros(len(columns), dtype=np.float64)
    m2 = np.zeros(len(columns), dtype=np.float64)

    usecols = [node_id_column, *columns]
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        numeric = chunk[columns].apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=np.float64, copy=False)

        if values.shape[0] == 0:
            continue
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite embedding value in chunk {chunk_number}: {path}"
            )

        chunk_count = values.shape[0]
        chunk_mean = values.mean(axis=0)
        chunk_m2 = ((values - chunk_mean) ** 2).sum(axis=0)

        if count == 0:
            mean = chunk_mean
            m2 = chunk_m2
            count = chunk_count
            continue

        combined_count = count + chunk_count
        delta = chunk_mean - mean
        mean = mean + delta * (chunk_count / combined_count)
        m2 = (
            m2
            + chunk_m2
            + (delta ** 2) * count * chunk_count / combined_count
        )
        count = combined_count

    if count == 0:
        raise ValueError(f"Embedding CSV contains no node rows: {path}")

    variance = np.maximum(m2 / count, 0.0)
    std = np.sqrt(variance)

    return columns, present_orders, count, mean, std


def feature_names(columns, reduction):
    if reduction == "mean":
        return [f"{column}__nodeMean" for column in columns]
    if reduction == "std":
        return [f"{column}__nodeStd" for column in columns]
    if reduction == "mean_std":
        return (
            [f"{column}__nodeMean" for column in columns]
            + [f"{column}__nodeStd" for column in columns]
        )
    raise ValueError(f"Unknown graph reduction: {reduction}")


def feature_values(mean, std, reduction):
    if reduction == "mean":
        return mean
    if reduction == "std":
        return std
    if reduction == "mean_std":
        return np.concatenate([mean, std])
    raise ValueError(f"Unknown graph reduction: {reduction}")


def output_path(embedding, reduction):
    return GRAPH_FEATURE_ROOT / embedding / f"{reduction}.csv"


def calculate_embedding_variant(embedding, reductions, config):
    """Build all requested city-level reductions for one embedding variant."""
    print(f"\n=== Graph features: {embedding} ===", flush=True)

    configured_orders = list(config["orders"])
    expected_columns = None
    rows = {reduction: [] for reduction in reductions}
    used_orders = None
    skipped = []

    for place in config["places"]:
        path = embedding_variant_path(place, embedding)
        if not path.is_file():
            print(f"Missing embedding for {place} (skipped): {path}", flush=True)
            skipped.append(place)
            continue

        columns, present_orders, node_count, mean, std = summarize_city(
            path,
            configured_orders,
            expected_columns,
        )

        if expected_columns is None:
            expected_columns = columns
            used_orders = present_orders
            print(
                f"Using r-values {used_orders}; "
                f"{len(columns):,} node-embedding columns.",
                flush=True,
            )

        for reduction in reductions:
            values = feature_values(mean, std, reduction)
            row = {"place": place}
            row.update(zip(feature_names(columns, reduction), values))
            rows[reduction].append(row)

        print(
            f"{place}: summarized {node_count:,} nodes.",
            flush=True,
        )

    if expected_columns is None:
        raise ValueError(
            f"No usable city embeddings found for {embedding}. "
            "Run FEATHER_embeddings.py and embedding_variants.py first."
        )

    for reduction in reductions:
        table = pd.DataFrame(rows[reduction])
        if table.empty:
            raise ValueError(f"No graph features created for {embedding} / {reduction}.")
        if not table["place"].is_unique:
            raise ValueError(
                f"Duplicate city rows while creating {embedding} / {reduction}."
            )

        path = output_path(embedding, reduction)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")
        table.to_csv(temporary, index=False)
        temporary.replace(path)

        print(
            f"Saved {len(table):,} city vectors x {len(table.columns) - 1:,} "
            f"features: {path}",
            flush=True,
        )

    if skipped:
        print(
            f"Skipped {len(skipped)} configured city/cities with missing embeddings.",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List embedding variants configured in input/variants.txt.",
    )
    parser.add_argument(
        "--embedding",
        help="Create graph features for one configured embedding variant only.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(CONFIG_PATH)
    embeddings, reductions = read_variants_file()

    if args.list:
        for index, embedding in enumerate(embeddings):
            print(f"{index}: {embedding}")
        return

    if args.embedding is not None:
        if args.embedding not in embeddings:
            raise ValueError(
                f"{args.embedding!r} is not configured in input/variants.txt. "
                "Run with --list to see configured embeddings."
            )
        embeddings = [args.embedding]

    for embedding in embeddings:
        calculate_embedding_variant(embedding, reductions, config)

    print("\nGraph-level feature files are ready.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
