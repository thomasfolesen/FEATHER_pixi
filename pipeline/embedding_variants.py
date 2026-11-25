"""Create mean-based node-level FEATHER embedding variants.

The source CSV contains exactly the r values selected by input/config.txt.
This script never adds, removes, or selects r values. It only averages across
amenity categories and/or evaluation points while keeping every r present in
that source file.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline_common import (
    FULL_EMBEDDING_VARIANT,
    embedding_variant_path,
    load_config,
)

CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
CHUNK_SIZE = 2_000

CATEGORY_MEAN_EVAL_ALL = "r_categoryMean_evalAll"
CATEGORY_ALL_EVAL_MEAN = "r_categoryAll_evalMean"
CATEGORY_MEAN_EVAL_MEAN = "r_categoryMean_evalMean"

COLUMN_RE = re.compile(
    r"^r(?P<order>\d+)_(?P<category>.+)_eval(?P<eval>\d+)_(?P<component>real|imaginary)$"
)

print("Starting pipeline/embedding_variants.py")


def configured_orders(config):
    """Return the configured orders as a sorted unique list of ints."""
    raw = config["orders"]
    values = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            values.extend(item)
        else:
            values.append(item)
    orders = sorted({int(value) for value in values})
    if not orders or min(orders) <= 0:
        raise ValueError("orders must contain positive integers.")
    return orders


def parse_source_columns(columns):
    """Parse full embedding columns and return metadata keyed by column name."""
    parsed = {}
    for column in columns:
        match = COLUMN_RE.fullmatch(column)
        if match is None:
            raise ValueError(f"Unexpected FEATHER embedding column: {column}")
        parsed[column] = {
            "order": int(match.group("order")),
            "category": match.group("category"),
            "eval": int(match.group("eval")),
            "component": match.group("component"),
        }
    return parsed


def validate_source_header(source_path, config):
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing full embedding CSV: {source_path}\n"
            "Run pipeline/FEATHER_embeddings.py first."
        )

    header = pd.read_csv(source_path, nrows=0)
    if len(header.columns) < 2:
        raise ValueError(f"Embedding CSV has no feature columns: {source_path}")

    node_id_column = header.columns[0]
    feature_columns = list(header.columns[1:])
    parsed = parse_source_columns(feature_columns)

    actual_orders = sorted({meta["order"] for meta in parsed.values()})
    expected_orders = configured_orders(config)
    if actual_orders != expected_orders:
        raise ValueError(
            f"{source_path} contains r values {actual_orders}, but config.txt "
            f"requests {expected_orders}. Delete the embedding output and rerun "
            "pipeline/FEATHER_embeddings.py."
        )

    expected_categories = set(config["resolved_categories"])
    actual_categories = {meta["category"] for meta in parsed.values()}
    if actual_categories != expected_categories:
        raise ValueError(
            f"{source_path} categories do not match input/config.txt."
        )

    expected_eval_points = set(range(1, int(config["eval_points"]) + 1))
    actual_eval_points = {meta["eval"] for meta in parsed.values()}
    if actual_eval_points != expected_eval_points:
        raise ValueError(
            f"{source_path} evaluation points do not match input/config.txt."
        )

    return node_id_column, feature_columns, parsed


def build_groups(feature_columns, parsed):
    """Build aggregation groups directly from the columns that actually exist."""
    orders = sorted({meta["order"] for meta in parsed.values()})
    categories = sorted({meta["category"] for meta in parsed.values()})
    eval_points = sorted({meta["eval"] for meta in parsed.values()})
    components = ("real", "imaginary")

    by_key = {
        (meta["order"], meta["category"], meta["eval"], meta["component"]): column
        for column, meta in parsed.items()
    }

    category_mean_eval_all = []
    category_all_eval_mean = []
    category_mean_eval_mean = []

    for order in orders:
        for component in components:
            for point in eval_points:
                source_columns = [
                    by_key[(order, category, point, component)]
                    for category in categories
                ]
                category_mean_eval_all.append((
                    f"r{order}_categoryMean_eval{point}_{component}",
                    source_columns,
                ))

            for category in categories:
                source_columns = [
                    by_key[(order, category, point, component)]
                    for point in eval_points
                ]
                category_all_eval_mean.append((
                    f"r{order}_{category}_evalMean_{component}",
                    source_columns,
                ))

            source_columns = [
                by_key[(order, category, point, component)]
                for category in categories
                for point in eval_points
            ]
            category_mean_eval_mean.append((
                f"r{order}_categoryMean_evalMean_{component}",
                source_columns,
            ))

    return {
        CATEGORY_MEAN_EVAL_ALL: category_mean_eval_all,
        CATEGORY_ALL_EVAL_MEAN: category_all_eval_mean,
        CATEGORY_MEAN_EVAL_MEAN: category_mean_eval_mean,
    }


def aggregate_chunk(chunk, node_id_column, groups):
    node_ids = chunk[node_id_column]
    numeric = chunk.drop(columns=[node_id_column]).apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("The full embedding CSV contains non-finite values.")

    outputs = {}
    for variant, definitions in groups.items():
        data = {node_id_column: node_ids.to_numpy(copy=False)}
        for output_column, source_columns in definitions:
            data[output_column] = numeric[source_columns].mean(axis=1).to_numpy(copy=False)
        outputs[variant] = pd.DataFrame(data)
    return outputs


def create_variants_for_city(place, config):
    source_path = embedding_variant_path(place, FULL_EMBEDDING_VARIANT)
    node_id_column, feature_columns, parsed = validate_source_header(source_path, config)
    groups = build_groups(feature_columns, parsed)

    final_paths = {
        variant: embedding_variant_path(place, variant)
        for variant in groups
    }
    temporary_paths = {
        variant: path.with_name(f"{path.name}.tmp")
        for variant, path in final_paths.items()
    }

    if all(path.is_file() for path in final_paths.values()):
        print(f"\nSkipping {place}: all embedding variants already exist.", flush=True)
        return

    for path in final_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    print(f"\nCreating embedding variants for {place}...", flush=True)
    print(f"Source: {source_path}", flush=True)

    wrote_rows = False
    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(source_path, dtype={node_id_column: str}, chunksize=CHUNK_SIZE),
            start=1,
        ):
            outputs = aggregate_chunk(chunk, node_id_column, groups)
            for variant, table in outputs.items():
                table.to_csv(
                    temporary_paths[variant],
                    mode="w" if chunk_number == 1 else "a",
                    header=chunk_number == 1,
                    index=False,
                )
            wrote_rows = True
            print(f"Processed chunk {chunk_number} ({len(chunk):,} nodes).", flush=True)

        if not wrote_rows:
            raise ValueError(f"Source embedding CSV contains no rows: {source_path}")

        for variant, final_path in final_paths.items():
            temporary_paths[variant].replace(final_path)
            print(f"Saved {variant}: {final_path}", flush=True)
    except Exception:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        raise


def run_pipeline():
    config = load_config(CONFIG_PATH)
    if not config["places"] or not config["resolved_categories"]:
        raise ValueError("Configure at least one place and amenity category.")

    for place in config["places"]:
        create_variants_for_city(place, config)

    print("\nAll embedding variants are ready.", flush=True)


if __name__ == "__main__":
    run_pipeline()
