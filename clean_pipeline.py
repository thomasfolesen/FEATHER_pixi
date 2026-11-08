#!/usr/bin/env python3
# Save this script in the project root, beside input/, cache/ and output/.
# Run the commands below in a terminal from that root, while the pipeline is stopped.
# Copy the command after '# ' when pasting into your terminal.
#
# Preview cleanup for places listed under 'place' in input/config.txt:
# python3 clean_pipeline.py --config
#
# Delete those places' cache and output, including shared city-comparison images:
# python3 clean_pipeline.py --config --delete
# Other cities' individual files and shared Geofabrik downloads are kept.
#
# Also delete the shared Geofabrik index and regional downloads:
# python3 clean_pipeline.py --config --shared-cache --delete
# These downloads will be fetched again when the pipeline needs them.
#
# Preview a full cleanup of cache/ and output/ for every place:
# python3 clean_pipeline.py --all
#
# Delete the entire cache/ and output/ directories, including their contents:
# python3 clean_pipeline.py --all --delete
#
# Commands without --delete only show what would be removed.
# Deletion is permanent. The input/config.txt file is kept.
"""Remove pipeline cache and output for all places or configured places."""

import argparse
import re
import shutil
from pathlib import Path


def safe_filename(place):
    # Same place names as the pipeline.
    return re.sub(r"[^a-zA-Z]+", "_", place).strip("_").lower()


def configured_places(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "place":
            places = [item.strip() for item in re.findall(r"\[([^\[\]]+)\]", value)]
            if not places or any(not safe_filename(place) for place in places):
                raise ValueError("The place setting must contain valid place names.")
            return places
    raise ValueError(f"No place setting found in {path}")


def cleanup_paths(root, places, shared_cache=False):
    paths = []
    for place in places:
        name = safe_filename(place)
        paths.extend([
            root / "cache" / name,
            root / "cache" / "interactive_map_data" / f"{name}.pkl",
        ])
        for folder in (
            "POI_distances", "graph_POis", "FEATHER_embeddings",
            "FEATHER_curves/FEATHER_curves_mean",
            "FEATHER_curves/FEATHER_curves_high_low",
        ):
            paths.append(root / "output" / folder / name)

    # City comparisons also contain the selected places, so they become stale.
    for folder in ("FEATHER_curves_mean", "FEATHER_curves_high_low"):
        parent = root / "output" / "FEATHER_curves" / folder
        check_path(parent, root)
        if parent.is_symlink():
            raise ValueError(f"Refusing to scan a linked directory: {parent}")
        if parent.exists():
            paths.extend(parent.glob("feather_*.png"))
    if shared_cache:
        # Regional downloads and the index can be shared by several cities.
        paths.append(root / "cache" / "geofabrik")
    return list(dict.fromkeys(paths))


def check_path(path, root):
    # Never follow a parent link outside the cache or output directory.
    base = root / path.relative_to(root).parts[0]
    if path != base and not path.parent.resolve().is_relative_to(base):
        raise ValueError(f"Path leaves the cleanup directory: {path}")


def remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Clear the entire cache and output directories.")
    mode.add_argument("--config", action="store_true", help="Clear places listed in input/config.txt.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Project root; defaults to the directory containing this script.")
    parser.add_argument("--shared-cache", action="store_true",
                        help="Also clear shared Geofabrik downloads in --config mode.")
    parser.add_argument("--delete", action="store_true", help="Delete the listed paths; otherwise only preview.")
    args = parser.parse_args()
    root = args.root.resolve()

    try:
        # Check the project location before preparing any deletion.
        config = root / "input" / "config.txt"
        if not config.is_file():
            raise ValueError(f"Cannot find {config}; put this script in the project root or use --root.")
        for folder in ("cache", "output"):
            if (root / folder).is_symlink():
                raise ValueError(f"Refusing to clean a linked {folder} directory.")

        if args.all:
            paths = [root / "cache", root / "output"]
        else:
            places = configured_places(config)
            print("Places: " + "; ".join(places))
            print("Includes shared cross-city comparison images.")
            if not args.shared_cache:
                print("Shared Geofabrik downloads are kept; add --shared-cache to clear them.")
            paths = cleanup_paths(root, places, args.shared_cache)

        paths = [path for path in paths if path.exists() or path.is_symlink()]
        # Validate the whole plan before deleting anything.
        for path in paths:
            check_path(path, root)
        if not paths:
            print("Nothing to delete.")
            return
        print("\nPaths to delete (directories include all their contents):")
        for path in paths:
            print(f"  {path}")
        if not args.delete:
            print("\nPreview only. Add --delete to remove these paths.")
            return
        for path in paths:
            remove_path(path)
        print(f"\nDeleted {len(paths)} paths.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cleanup failed: {error}\n")


if __name__ == "__main__":
    main()
