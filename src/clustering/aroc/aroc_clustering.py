import numpy as np

from src.clustering.abstract_clustering import AbstractClustering
from src.clustering.aroc.aroc import aroc


class AROClustering(AbstractClustering):
    def __init__(self, n_neighbours: int, threshold: float, num_proc: int = 8):
        self.n_neighbours: int = n_neighbours
        self.threshold: float = threshold
        self.num_proc: int = num_proc

    def cluster(self, samples: np.ndarray) -> np.ndarray:
        return aroc(samples, self.n_neighbours, self.threshold, self.num_proc)
