from abc import ABC, abstractmethod

import numpy as np


class AbstractClustering(ABC):
    @abstractmethod
    def cluster(self, samples: np.ndarray) -> np.ndarray:
        """Clusters the given samples.

        :param samples: A 2-d numpy array of shape (n_samples, n_features) containing
            the samples.
        :return: A 1-d numpy array of shape containing the cluster label for each sample
            in the same order as the input array.
        """
