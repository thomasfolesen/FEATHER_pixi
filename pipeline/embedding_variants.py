"""Create mean-based node-level FEATHER embedding variants.

The source is the full selected-order embedding produced by
pipeline/FEATHER_embeddings.py:

    r_categoryAll_evalAll

This script writes three derived variants without running FEATHER again:

    r_categoryMean_evalAll
    r_categoryAll_evalMean
    r_categoryMean_evalMean

Large CSVs are processed in chunks so the complete embedding matrix is never
loaded into memory at once. Existing derived CSVs are overwritten from the
current full source CSV. No JSON sidecar files are created.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.pipeline_common import (
    FULL_EMBEDDING_VARIANT,
    embedding_columns,
    embedding_variant_path,
    load_config,
)


CONFIG_PATH = PROJECT_ROOT / "input" / "config.txt"
CHUNK_SIZE = 2_000

CATEGORY_MEAN_EVAL_ALL = "r_categoryMean_evalAll"
CATEGORY_ALL_EVAL_MEAN = "r_categoryAll_evalMean"
CATEGORY_MEAN_EVAL_MEAN = "r_categoryMean_evalMean"

print("Starting pipeline/embedding_variants.py")


def build_groups(categories, orders, eval_points):
    """Return ordered source-column groups for each derived variant."""
    categories = list(categories)
    orders = list(orders)

    category_mean_eval_all = []
    category_all_eval_mean = []
    category_mean_eval_mean = []

    for order in orders:
        for component in ("real", "imaginary"):
            for point in range(1, eval_points + 1):
                output_column = (
                    f"r{order}_categoryMean_eval{point}_{component}"
                )
                source_columns = [
                    f"r{order}_{category}_eval{point}_{component}"
                    for category in categories
                ]
                category_mean_eval_all.append(
                    (output_column, source_columns)
                )

            for category in categories:
                output_column = (
                    f"r{order}_{category}_evalMean_{component}"
                )
                source_columns = [
                    f"r{order}_{category}_eval{point}_{component}"
                    for point in range(1, eval_points + 1)
                ]
                category_all_eval_mean.append(
                    (output_column, source_columns)
                )

            output_column = (
                f"r{order}_categoryMean_evalMean_{component}"
            )
            source_columns = [
                f"r{order}_{category}_eval{point}_{component}"
                for category in categories
                for point in range(1, eval_points + 1)
            ]
            category_mean_eval_mean.append(
                (output_column, source_columns)
            )

    return {
        CATEGORY_MEAN_EVAL_ALL: category_mean_eval_all,
        CATEGORY_ALL_EVAL_MEAN: category_all_eval_mean,
        CATEGORY_MEAN_EVAL_MEAN: category_mean_eval_mean,
    }


def validate_source_header(source_path, expected_columns):
    """Validate the source CSV and return its node-ID column name."""
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing full embedding CSV: {source_path}\n"
            "Run: pixi run python pipeline/FEATHER_embeddings.py"
        )

    header = pd.read_csv(source_path, nrows=0)
    if header.columns.empty:
        raise ValueError(f"Source embedding CSV has no columns: {source_path}")

    node_id_column = header.columns[0]
    actual_columns = list(header.columns[1:])
    if actual_columns != list(expected_columns):
        raise ValueError(
            "The full embedding CSV columns do not match the current "
            "categories, orders, or eval_points in input/config.txt:\n"
            f"  {source_path}\n"
            "Delete or move the full CSV and rerun FEATHER_embeddings.py."
        )

    return node_id_column


def aggregate_chunk(chunk, node_id_column, groups):
    """Create all three derived DataFrames for one source chunk."""
    node_ids = chunk[node_id_column]
    numeric = chunk.drop(columns=[node_id_column]).apply(
        pd.to_numeric,
        errors="raise",
    )

    if not np.isfinite(numeric.to_numpy(dtype=float, copy=False)).all():
        raise ValueError("The full embedding CSV contains non-finite values.")

    outputs = {}
    for variant, definitions in groups.items():
        data = {node_id_column: node_ids.to_numpy(copy=False)}
        for output_column, source_columns in definitions:
            data[output_column] = numeric[source_columns].mean(
                axis=1,
                skipna=False,
            ).to_numpy(copy=False)
        outputs[variant] = pd.DataFrame(data, index=chunk.index)

    return outputs


def create_variants_for_city(place, config):
    """Create the three mean-based variants for one city's full embedding."""
    categories = list(config["resolved_categories"])
    orders = list(config["orders"])
    eval_points = config["eval_points"]

    source_path = embedding_variant_path(place, FULL_EMBEDDING_VARIANT)
    expected_columns = list(
        embedding_columns(categories, orders, eval_points)
    )
    node_id_column = validate_source_header(source_path, expected_columns)
    groups = build_groups(categories, orders, eval_points)

    final_paths = {
        variant: embedding_variant_path(place, variant)
        for variant in groups
    }
    temporary_paths = {
        variant: path.with_name(f"{path.name}.tmp")
        for variant, path in final_paths.items()
    }

    for path in final_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    print(f"\nCreating embedding variants for {place}...", flush=True)
    print(f"Source: {source_path}", flush=True)

    wrote_any_rows = False
    seen_node_ids = set()
    try:
        for chunk_number, chunk in enumerate(
            pd.read_csv(
                source_path,
                dtype={node_id_column: str},
                chunksize=CHUNK_SIZE,
            ),
            start=1,
        ):
            node_ids = chunk[node_id_column]
            if node_ids.duplicated().any():
                raise ValueError(
                    f"Duplicate node IDs found inside chunk {chunk_number}: "
                    f"{source_path}"
                )

            current_ids = set(node_ids)
            repeated_ids = current_ids.intersection(seen_node_ids)
            if repeated_ids:
                example = next(iter(repeated_ids))
                raise ValueError(
                    "Duplicate node IDs found across source chunks; "
                    f"for example {example!r}: {source_path}"
                )
            seen_node_ids.update(current_ids)

            outputs = aggregate_chunk(chunk, node_id_column, groups)
            for variant, table in outputs.items():
                table.to_csv(
                    temporary_paths[variant],
                    mode="w" if chunk_number == 1 else "a",
                    header=chunk_number == 1,
                    index=False,
                )

            wrote_any_rows = True
            print(
                f"Processed chunk {chunk_number} "
                f"({len(chunk):,} nodes).",
                flush=True,
            )

        if not wrote_any_rows:
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
        raise ValueError(
            "Configure at least one place and amenity category in input/config.txt."
        )

    for place in config["places"]:
        create_variants_for_city(place, config)

    print("\nAll embedding variants are ready.", flush=True)


if __name__ == "__main__":
    run_pipeline()
