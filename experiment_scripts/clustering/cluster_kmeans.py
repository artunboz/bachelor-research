import os

from paths import DATA_DIR
from src.clustering.kmeans_clustering import KMeansClustering

n_clusters_space = [10, 20]

root_folder = f"{DATA_DIR}/lbp"
eval_folders = sorted(os.listdir(root_folder))

for i, n_clusters in enumerate(n_clusters_space):
    for folder in eval_folders:
        folder_path = f"{root_folder}/{folder}"
        clustering_folder_path = f"{folder_path}/clustering"
        kmeans_folder_path = f"{clustering_folder_path}/kmeans"

        if not os.path.isdir(clustering_folder_path):
            os.mkdir(clustering_folder_path)
        if not os.path.isdir(kmeans_folder_path):
            os.mkdir(kmeans_folder_path)

        clustering = KMeansClustering(n_clusters=n_clusters)
        clustering.cluster(folder_path)
        clustering.save_cluster_labels(f"{kmeans_folder_path}/run_{i}")
