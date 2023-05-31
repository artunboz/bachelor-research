import json
import os

import pandas as pd

from paths import DATA_DIR
from src.evaluation.evaluator import Evaluator

clustering_type = "kmeans"
run = "run_0"

root_folder = f"{DATA_DIR}/lbp"
eval_folders = sorted(os.listdir(root_folder))
for folder in eval_folders:
    folder_path = f"{root_folder}/{folder}"
    evaluator = Evaluator(
        features_path=f"{folder_path}/features.npy",
        cluster_labels_folder_path=f"{folder_path}/clustering/{clustering_type}/{run}",
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
        "calinski_harabasz",
        "davies_bouldin",
        "precision",
        "recall",
        "f1",
    ]
)
for i, folder in enumerate(eval_folders):
    folder_path = f"{root_folder}/{folder}"
    with open(
        f"{folder_path}/clustering/{clustering_type}/{run}/metrics.json", mode="r"
    ) as f:
        d = json.load(f)
        d["name"] = folder
        d_df = pd.DataFrame([d])
        all_results_df = pd.concat([all_results_df, d_df], ignore_index=True)

results_path = f"{root_folder}/results"
if not os.path.isdir(results_path):
    os.mkdir(results_path)
all_results_df.to_csv(f"{results_path}/{run}_results.csv", index=False)
