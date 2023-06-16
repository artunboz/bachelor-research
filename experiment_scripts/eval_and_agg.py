import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from paths import DATA_DIR
from src.evaluation.evaluator import Evaluator

parser = ArgumentParser()
parser.add_argument("--feature-path", dest="feature_path")
parser.add_argument("--clustering-type", dest="clustering_type")
args = parser.parse_args()

feature_folder_path = f"{DATA_DIR}/{args.feature_path}"
clustering_folders_path = (
    f"{DATA_DIR}/{args.feature_path}/clustering/{args.clustering_type}"
)

# Compute results.
for folder in tqdm(clustering_folders_path, desc="Evaluated Folders"):
    folder_path = f"{clustering_folders_path}/{folder}"
    evaluator = Evaluator(
        features_path=f"{feature_folder_path}/features.npy",
        cluster_labels_folder_path=folder_path,
        image_names_path=f"{feature_folder_path}/image_names.pickle",
        ground_truth_path=f"{DATA_DIR}/labelled_faces/clean_labels.csv",
    )
    evaluator.compute_metrics()
    evaluator.save_metrics()

# Aggregate results.
all_results_df = pd.DataFrame(
    columns=[
        "name",
        "image_count",
        "silhouette",
        "davies_bouldin",
        "precision",
        "recall",
        "f1",
    ]
)
for folder in clustering_folders_path:
    folder_path = f"{clustering_folders_path}/{folder}"
    with open(f"{folder_path}/metrics.json", mode="r") as f:
        d = json.load(f)
        d["name"] = folder
        d_df = pd.DataFrame([d])
        all_results_df = pd.concat([all_results_df, d_df], ignore_index=True)

results_path = f"{feature_folder_path}/results"
if not os.path.isdir(results_path):
    os.mkdir(results_path)
all_results_df.to_csv(f"{results_path}/{args.clustering_type}_results.csv", index=False)
