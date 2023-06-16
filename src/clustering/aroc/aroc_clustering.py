import numpy as np

from src.clustering.abstract_clustering import AbstractClustering
from src.clustering.aroc.aroc import cluster_aroc


class AROClustering(AbstractClustering):
    def __init__(
        self, n_neighbours: int, threshold: float, min_samples: int, num_proc: int = 20
    ):
        """Inits an AROClustering instance.

        :param n_neighbours: An integer indicating the number of neighbors to use.
        :param threshold: A float indicating the merging threshold.
        :param min_samples: An integer indicating the minimum number of samples in a
            cluster such that the samples are not considered fuzzy.
        :param num_proc: An integer indicating the number of cores to use. Defaults to
            20.
        """
        super().__init__()
        self.n_neighbours: int = n_neighbours
        self.threshold: float = threshold
        self.min_samples: int = min_samples
        self.num_proc: int = num_proc

    def cluster(self, features_dir: str) -> np.ndarray:
        """Clusters the given samples.

        :param features_dir: A string indicating the file containing the features.
        :return: A 1-d numpy array of shape containing the cluster label for each sample
            in the same order as the input array.
        """
        features: np.ndarray = np.load(f"{features_dir}/features.npy").astype("int32")
        self.cluster_labels = cluster_aroc(
            features, self.n_neighbours, self.threshold, self.min_samples, self.num_proc
        )

        return self.cluster_labels
