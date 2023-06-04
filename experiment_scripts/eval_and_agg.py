import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from paths import DATA_DIR
from src.evaluation.evaluator import Evaluator

parser = ArgumentParser()
parser.add_argument("feature")
parser.add_argument("n_runs", type=int)
args = parser.parse_args()

clustering_type = "kmeans"
root_folder = f"{DATA_DIR}/{args.feature}"
eval_folders = sorted(os.listdir(root_folder))
if "results" in eval_folders:
    eval_folders.remove("results")
if "configs.csv" in eval_folders:
    eval_folders.remove("configs.csv")

for i in range(args.n_runs):
    # Compute results.
    for folder in tqdm(eval_folders, desc="Evaluated Folders"):
        folder_path = f"{root_folder}/{folder}"
        evaluator = Evaluator(
            features_path=f"{folder_path}/features.npy",
            cluster_labels_folder_path=f"{folder_path}/clustering/{clustering_type}/run_{i}",
            image_names_path=f"{root_folder}/{folder}/image_names.pickle",
            ground_truth_path=f"{DATA_DIR}/labelled_faces/clean_labels.csv",
        )
        evaluator.compute_metrics()
        evaluator.save_metrics()

    all_results_df = pd.DataFrame(
        columns=[
            "name",
            "image_count",
            "silhouette",
            # "calinski_harabasz",
            "davies_bouldin",
            # "precision",
            # "recall",
            "f1",
        ]
    )
    # Aggregate results.
    for folder in eval_folders:
        folder_path = f"{root_folder}/{folder}"
        with open(
                f"{folder_path}/clustering/{clustering_type}/run_{i}/metrics.json",
                mode="r"
        ) as f:
            d = json.load(f)
            d["name"] = folder
            d_df = pd.DataFrame([d])
            all_results_df = pd.concat([all_results_df, d_df], ignore_index=True)

    results_path = f"{root_folder}/results"
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    all_results_df.to_csv(f"{results_path}/run_{i}_results.csv", index=False)
