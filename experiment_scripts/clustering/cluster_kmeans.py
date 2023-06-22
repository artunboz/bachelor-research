import os
from argparse import ArgumentParser

from tqdm import tqdm

from paths import DATA_DIR
from src.clustering.kmeans_clustering import KMeansClustering

parser = ArgumentParser()
parser.add_argument("--feature-path", dest="feature_path")
args = parser.parse_args()

n_clusters = 12

root_folder = f"{DATA_DIR}/{args.feature_path}"
eval_folders = sorted(os.listdir(root_folder))

for folder in tqdm(eval_folders, desc="Configurations"):
    folder_path = f"{root_folder}/{folder}"
    clustering_folder_path = f"{folder_path}/clustering"
    kmeans_folder_path = f"{clustering_folder_path}/kmeans/run_0"

    if not os.path.isdir(clustering_folder_path):
        os.mkdir(clustering_folder_path)
    if not os.path.isdir(kmeans_folder_path):
        os.mkdir(kmeans_folder_path)

    clustering = KMeansClustering(n_clusters=n_clusters)
    clustering.cluster(folder_path)
    clustering.save_cluster_labels(f"{kmeans_folder_path}")
