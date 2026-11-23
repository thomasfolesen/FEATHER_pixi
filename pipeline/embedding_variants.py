"""Build reduced FEATHER files and select the embedding variants to test."""
# Save as pipeline/embedding_variants.py. Run from the project root, after
# pipeline/graph_feature_calculation.py:
#   pixi run python pipeline/embedding_variants.py           (build files)
#   pixi run python pipeline/embedding_variants.py --list    (show variants)
# Writes:
#   output/FEATHER_embeddings_reduced/<city>/<city>_feather_embeddings_reduced.csv (+ .json)
#   output/FEATHER_patterns/graph_features_reduced.csv (+ _info.json)
# Run it once before the HPC jobs. clusters.py and classification.py only read.

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.graph_feature_calculation import (
    EMBEDDING_ROOT, GRAPH_FEATURES_INFO_PATH, GRAPH_FEATURES_PATH, OUTPUT_ROOT as PATTERN_ROOT,
    feature_names, load_config as load_base_config, safe_filename, summarise_graph,
)

VARIANTS_PATH = PROJECT_ROOT / "input" / "variants.txt"
REDUCED_ROOT = PROJECT_ROOT / "output" / "FEATHER_embeddings_reduced"
REDUCED_VECTORS_PATH = PATTERN_ROOT / "graph_features_reduced.csv"
REDUCED_VECTORS_INFO_PATH = PATTERN_ROOT / "graph_features_reduced_info.json"
EXPERIMENT_ROOT = PROJECT_ROOT / "output" / "experiments"

CATEGORY_MODES = ("as_is", "mean", "std", "mean_std")
LEVELS = ("node", "city")
R_MODES = ("all", "min_max", "single", "interval")
STAT_NAMES = {"mean": "amenityMean", "std": "amenityStd"}
R_IN_COLUMN = re.compile(r"(?:^|__)r(\d+)_")


# ---------- config ----------

@dataclass
class Setup:
    settings: dict
    places: list
    categories: list
    order: int
    eval_points: int
    columns: list  # as-is embedding columns

    def signature(self):
        return {"categories": self.categories, "order": self.order, "eval_points": self.eval_points}


def load_setup():
    settings, places, columns = load_base_config()
    categories = [key for key, value in settings.items() if key != "place" and value.startswith("[[")]
    return Setup(settings, places, categories, int(settings["order"]), int(settings["eval_points"]), columns)


def reduced_columns(order, eval_points):
    """Mean columns first, then std columns, both in (r, component, point) order."""
    return [
        f"r{r}_{STAT_NAMES[stat]}_eval{point}_{component}"
        for stat in STAT_NAMES
        for r in range(1, order + 1)
        for component in ("real", "imaginary")
        for point in range(1, eval_points + 1)
    ]


def category_groups(setup):
    """Index matrix (groups x categories) into the as-is columns."""
    position = {column: index for index, column in enumerate(setup.columns)}
    return np.array([
        [position[f"r{r}_{category}_eval{point}_{component}"] for category in setup.categories]
        for r in range(1, setup.order + 1)
        for component in ("real", "imaginary")
        for point in range(1, setup.eval_points + 1)
    ])


def master_path(place):
    name = safe_filename(place)
    return EMBEDDING_ROOT / name / f"{name}_feather_embeddings.csv"


def reduced_path(place):
    name = safe_filename(place)
    return REDUCED_ROOT / name / f"{name}_feather_embeddings_reduced.csv"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- build reduced files ----------

def reduce_embeddings_file(source, destination, setup):
    header = pd.read_csv(source, nrows=0).columns.tolist()
    if header[0] != "node_id" or sorted(header[1:]) != sorted(setup.columns):
        raise ValueError(f"{source.name}: embedding columns do not match config.txt.")
    groups = category_groups(setup)
    names = reduced_columns(setup.order, setup.eval_points)
    chunk_size = max(1, min(5000, 16_000_000 // (8 * len(setup.columns) * 4)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".partial")  # no broken file if stopped
    rows = 0
    try:
        for chunk in pd.read_csv(source, dtype={"node_id": str}, chunksize=chunk_size):
            values = chunk[setup.columns].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{source.name}: embeddings contain missing or infinite values.")
            stacked = values[:, groups]  # nodes x groups x categories
            reduced = pd.DataFrame(np.hstack([stacked.mean(axis=2), stacked.std(axis=2)]), columns=names)
            reduced.insert(0, "node_id", chunk["node_id"].to_numpy())
            reduced.to_csv(temp, mode="w" if rows == 0 else "a", header=rows == 0, index=False)
            rows += len(chunk)
        if not rows:
            raise ValueError(f"{source.name}: no rows found.")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return rows


def ensure_reduced_embeddings(place, setup, rebuild=False):
    source, destination = master_path(place), reduced_path(place)
    if not source.is_file():
        raise FileNotFoundError(f"missing embeddings: {source}")
    meta_path = destination.with_suffix(".json")
    expected = {**setup.signature(), "source_size": source.stat().st_size}
    meta = read_json(meta_path)
    if not rebuild and destination.is_file() and meta and {k: meta.get(k) for k in expected} == expected:
        print(f"Using cached reduced embeddings for {place}.", flush=True)
        return destination
    print(f"Reducing embeddings for {place} ({source.stat().st_size / 1024**3:.2f} GB)...", flush=True)
    vertices = reduce_embeddings_file(source, destination, setup)
    write_json(meta_path, {**expected, "vertices": vertices, "reduced_size": destination.stat().st_size})
    return destination


def reduced_vectors_fresh(setup, places):
    info = read_json(REDUCED_VECTORS_INFO_PATH)
    if not info or not REDUCED_VECTORS_PATH.is_file():
        return False
    if info["settings"] != setup.signature() or info["places"] != places:
        return False
    for place in places:
        meta = read_json(reduced_path(place).with_suffix(".json"))
        if not meta or meta.get("reduced_size") != info["reduced_sizes"].get(place):
            return False
    return True


def build_reduced_vectors(places, setup, rebuild=False):
    """graph_features_reduced.csv: mean and std over nodes of the reduced per-node files."""
    if not rebuild and reduced_vectors_fresh(setup, places):
        print("Using cached graph_features_reduced.csv.", flush=True)
        return
    names = reduced_columns(setup.order, setup.eval_points)
    summaries, counts, sizes = {}, {}, {}
    for place in places:
        path = reduced_path(place)
        summaries[place], counts[place] = summarise_graph(path, names)
        sizes[place] = read_json(path.with_suffix(".json"))["reduced_size"]
    features = pd.DataFrame.from_dict(summaries, orient="index", columns=feature_names(names))
    features.index.name = "place"
    PATTERN_ROOT.mkdir(parents=True, exist_ok=True)
    temp = REDUCED_VECTORS_PATH.with_name(REDUCED_VECTORS_PATH.name + ".partial")
    try:
        features.to_csv(temp)
        temp.replace(REDUCED_VECTORS_PATH)
    finally:
        temp.unlink(missing_ok=True)
    write_json(REDUCED_VECTORS_INFO_PATH, {
        "settings": setup.signature(), "places": places, "vertices": counts, "reduced_sizes": sizes,
    })
    print(f"Saved {REDUCED_VECTORS_PATH} ({len(features)} places).", flush=True)


# ---------- variants ----------

@dataclass(frozen=True)
class Variant:
    categories: str
    level: str
    r_mode: str
    r_values: tuple
    r_label: str

    @property
    def name(self):
        return f"{self.categories}__{self.level}__{self.r_label}"

    @property
    def reduced(self):
        return self.categories != "as_is"


def split_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def read_variant_settings(path=VARIANTS_PATH):
    settings = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()
    return settings


def resolve_r(token, order):
    """single:3 | min_max | interval:3-7 | all  ->  (mode, r values, label)."""
    mode, _, argument = token.partition(":")
    if mode == "all":
        return mode, tuple(range(1, order + 1)), "all"
    if mode == "min_max":
        return mode, tuple(sorted({1, order})), "minmax"
    if mode == "single":
        r = int(argument)
        if not 1 <= r <= order:
            raise ValueError(f"{token}: r must be between 1 and {order}.")
        return mode, (r,), f"r{r}"
    if mode == "interval":
        start, end = (int(part) for part in argument.split("-"))
        if not 1 <= start <= end <= order:
            raise ValueError(f"{token}: use interval:START-END with 1 <= START <= END <= {order}.")
        return mode, tuple(range(start, end + 1)), f"r{start}-{end}"
    raise ValueError(f"Unknown r selection {token!r}; use {', '.join(R_MODES)}.")


def list_variants(setup, levels=LEVELS, path=VARIANTS_PATH):
    settings = read_variant_settings(path)
    categories, chosen_levels = split_list(settings["categories"]), split_list(settings["levels"])
    for value, allowed in [(c, CATEGORY_MODES) for c in categories] + [(l, LEVELS) for l in chosen_levels]:
        if value not in allowed:
            raise ValueError(f"{path.name}: unknown value {value!r}; use one of {allowed}.")
    r_options = [resolve_r(token, setup.order) for token in split_list(settings["r_values"])]
    variants = [
        Variant(category, level, *r_option)
        for category in categories for level in chosen_levels if level in levels for r_option in r_options
    ]
    only = split_list(settings.get("only", ""))
    if only:
        unknown = set(only) - {variant.name for variant in variants}
        if unknown:
            raise ValueError(f"{path.name}: unknown names in only: {sorted(unknown)}")
        variants = [variant for variant in variants if variant.name in only]
    return variants


def variant_columns(variant, available):
    keep = [column for column in available if int(R_IN_COLUMN.search(column).group(1)) in variant.r_values]
    if variant.categories in ("mean", "std"):
        keep = [column for column in keep if f"_{STAT_NAMES[variant.categories]}_" in column]
    if not keep:
        raise ValueError(f"{variant.name}: no columns selected.")
    return keep

# ---------- ONLY USED FOR SMALL TESTING ----------

def cv_folds(path=VARIANTS_PATH):
    folds = int(read_variant_settings(path).get("folds", 5))
    if folds < 2:
        raise ValueError(f"{path.name}: folds must be 2 or more.")
    return folds
# ---------- ONLY USED FOR SMALL TESTING ----------

def node_sample_per_city(path=VARIANTS_PATH):
    return int(read_variant_settings(path).get("node_sample_per_city", 1000))


# ---------- read variants (used by clusters.py and classification.py) ----------

RERUN = "Run pipeline/graph_feature_calculation.py and pipeline/embedding_variants.py first."


def load_city_table(variant, setup):
    """One row per city with the variant's columns. Returns (table, vertices)."""
    if variant.reduced:
        path, info = REDUCED_VECTORS_PATH, read_json(REDUCED_VECTORS_INFO_PATH)
        names = feature_names(reduced_columns(setup.order, setup.eval_points))
        if not info or info["settings"] != setup.signature():
            raise ValueError(f"{path.name} is missing or does not match config.txt. {RERUN}")
    else:
        path, info = GRAPH_FEATURES_PATH, read_json(GRAPH_FEATURES_INFO_PATH)
        names = feature_names(setup.columns)
        if not info or info["embedding_columns"] != setup.columns:
            raise ValueError(f"{path.name} is missing or does not match config.txt. {RERUN}")
    table = pd.read_csv(path, index_col="place", dtype={"place": str}, keep_default_na=False,
                        float_precision="round_trip")
    if table.columns.tolist() != names:
        raise ValueError(f"{path.name} does not match config.txt. {RERUN}")
    missing = [place for place in setup.places if place not in table.index]
    if missing:
        print(f"Not in {path.name} (skipped): {missing}", flush=True)
    table = table.loc[[place for place in setup.places if place in table.index]]
    return table[variant_columns(variant, names)], {p: info["vertices"][p] for p in table.index}


def node_sources(variant, setup):
    """(place, csv path, columns, vertices) for each city with a node-level file."""
    sources = []
    if variant.reduced:
        names = reduced_columns(setup.order, setup.eval_points)
        for place in setup.places:
            path = reduced_path(place)
            meta = read_json(path.with_suffix(".json"))
            if not path.is_file() or not meta:
                print(f"No reduced embeddings for {place} (skipped).", flush=True)
                continue
            if {k: meta[k] for k in setup.signature()} != setup.signature():
                raise ValueError(f"{path.name} does not match config.txt. {RERUN}")
            sources.append((place, path, variant_columns(variant, names), meta["vertices"]))
    else:
        info = read_json(GRAPH_FEATURES_INFO_PATH)
        if not info or info["embedding_columns"] != setup.columns:
            raise ValueError(f"{GRAPH_FEATURES_INFO_PATH.name} is missing or does not match config.txt. {RERUN}")
        for place in setup.places:
            path = master_path(place)
            if not path.is_file() or place not in info["vertices"]:
                print(f"No embeddings for {place} (skipped).", flush=True)
                continue
            sources.append((place, path, variant_columns(variant, setup.columns), info["vertices"][place]))
    if not sources:
        raise ValueError(f"{variant.name}: no node-level files found. {RERUN}")
    return sources


def add_variant_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--variant", help="Run one variant by name.")
    group.add_argument("--variant-index", type=int, help="Run one variant by index (SLURM array).")
    group.add_argument("--all", action="store_true", help="Run every variant.")
    group.add_argument("--list", action="store_true", help="Print the variants with their index.")
    group.add_argument("--count", action="store_true", help="Print the number of variants.")


def chosen_variants(args, variants):
    """Returns the variants to run, or None after --list/--count."""
    if args.list:
        for index, variant in enumerate(variants):
            print(index, variant.name)
        return None
    if args.count:
        print(len(variants))
        return None
    if args.all:
        return variants
    if args.variant_index is not None:
        if not 0 <= args.variant_index < len(variants):
            raise ValueError(f"--variant-index must be between 0 and {len(variants) - 1}.")
        return [variants[args.variant_index]]
    matches = [variant for variant in variants if variant.name == args.variant]
    if not matches:
        raise ValueError(f"Unknown variant {args.variant!r}. Use --list to see the names.")
    return matches


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print every variant and stop.")
    parser.add_argument("--rebuild", action="store_true", help="Build the reduced files again.")
    args = parser.parse_args()
    setup = load_setup()
    variants = list_variants(setup)
    if args.list:
        for index, variant in enumerate(variants):
            print(index, variant.name, "r =", list(variant.r_values))
        return

    if not GRAPH_FEATURES_PATH.is_file():
        print(f"Warning: {GRAPH_FEATURES_PATH} is missing. Run pipeline/graph_feature_calculation.py "
              "for the as-is city variants.", flush=True)
    if not any(variant.reduced for variant in variants):
        print("No reduced variants in variants.txt; nothing to build.")
        return

    built = []
    for place in setup.places:
        try:
            ensure_reduced_embeddings(place, setup, rebuild=args.rebuild)
            built.append(place)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipping {place}: {error}", flush=True)
    if not built:
        raise ValueError("No usable embedding files. Run FEATHER_embeddings.py first.")
    build_reduced_vectors(built, setup, rebuild=args.rebuild)
    print(f"\nReady: {len(variants)} variants, {len(built)} cities.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
