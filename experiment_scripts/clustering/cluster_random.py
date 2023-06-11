import os
from argparse import ArgumentParser

from tqdm import tqdm

from paths import DATA_DIR
from src.clustering.random_clustering import RandomClustering

parser = ArgumentParser()
parser.add_argument("--feature-path", dest="feature_path")
args = parser.parse_args()

n_clusters = 12

root_folder = f"{DATA_DIR}/{args.feature_path}"
eval_folders = sorted(os.listdir(root_folder))

for folder in tqdm(eval_folders, desc="Configurations"):
    folder_path = f"{root_folder}/{folder}"
    clustering_folder_path = f"{folder_path}/clustering"
    random_folder_path = f"{clustering_folder_path}/random"

    if not os.path.isdir(clustering_folder_path):
        os.mkdir(clustering_folder_path)
    if not os.path.isdir(random_folder_path):
        os.mkdir(random_folder_path)

    clustering = RandomClustering(n_clusters=n_clusters)
    clustering.cluster(folder_path)
    clustering.save_cluster_labels(f"{random_folder_path}")
