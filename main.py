"""
Run program with:
pixi run python -u main.py

# Delete the entire cache/ and output/ directories, including their contents:
# python3 clean_pipeline.py --all --delete
"""

from pipeline.geofabrik_final import run_pipeline
from classification.clusters import main as run_clustering


if __name__ == "__main__":
    # Create city data and FEATHER embeddings.
    run_pipeline()

    # Cluster the graphs using the saved embeddings.
    run_clustering()