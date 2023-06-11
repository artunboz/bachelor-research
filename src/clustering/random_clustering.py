import numpy as np

from src.clustering.abstract_clustering import AbstractClustering


class RandomClustering(AbstractClustering):
    def __init__(self, n_clusters: int) -> None:
        """Inits a RandomClustering instance.

        :param n_clusters: An integer indicating the number of clusters.
        """
        super().__init__()
        self.n_clusters: int = n_clusters
        self.rng: np.random.Generator = np.random.default_rng()

    def cluster(self, features_dir: str) -> np.ndarray:
        """Clusters the given samples.

        :param features_dir: A string indicating the file containing the features.
        :return: A 1-d numpy array of shape containing the cluster label for each sample
            in the same order as the input array.
        """
        features: np.ndarray = np.load(f"{features_dir}/features.npy")
        self.cluster_labels = self.rng.integers(self.n_clusters, size=features.shape[0])

        return self.cluster_labels
