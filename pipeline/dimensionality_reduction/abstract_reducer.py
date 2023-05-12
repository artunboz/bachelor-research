from abc import ABC, abstractmethod

import numpy as np


class AbstractReducer(ABC):
    @abstractmethod
    def reduce_dimensions(self, samples: np.ndarray) -> np.ndarray:
        """Reduces the dimensions of the given samples.

        :param samples: A 2-d numpy array of shape (n_samples, n_features) containing
            the samples.
        :return: A 2-d numpy array of shape (n_samples, n_reduced_features) containing
            the samples in a latent space with a lower dimensionality.
        """
