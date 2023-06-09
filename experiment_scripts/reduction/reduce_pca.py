import json
from argparse import ArgumentParser

from paths import DATA_DIR
from src.dimensionality_reduction.pca_reducer import PCAReducer

parser = ArgumentParser()
parser.add_argument("--feature-path", dest="feature_path")
args = parser.parse_args()

features_dir = f"{DATA_DIR}/{args.feature_path}"
reductions_dir = f"{features_dir}/reductions/pca"

n_components_space = [10, 50, 100, 200]

with open(f"{features_dir}/feature_config.json", mode="r") as f:
    output_dim = json.load(f)["feature_dim"]

for i, n_components in enumerate(n_components_space):
    reducer = PCAReducer(n_components)
    reducer.reduce_dimensions(features_dir=features_dir)
    reducer.save_reduced_features(f"{reductions_dir}/run_{i}")
