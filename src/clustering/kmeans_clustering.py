from typing import cast

import numpy as np
from sklearn.cluster import KMeans

from src.clustering.abstract_clustering import AbstractClustering


class KMeansClustering(AbstractClustering):
    def __init__(self, n_clusters: int) -> None:
        """Inits a KMeansClustering instance.

        :param n_clusters: An integer indicating the number of clusters.
        """
        super().__init__()
        self.n_clusters: int = n_clusters

    def cluster(self, features_dir: str) -> np.ndarray:
        """Clusters the given samples.

        :param features_dir: A string indicating the file containing the features.
        :return: A 1-d numpy array of shape containing the cluster label for each sample
            in the same order as the input array.
        """
        features: np.ndarray = np.load(f"{features_dir}/features.npy")
        kmeans: KMeans = cast(
            KMeans, KMeans(n_clusters=self.n_clusters, n_init="auto").fit(features)
        )
        self.cluster_labels = kmeans.labels_

        return self.cluster_labels
