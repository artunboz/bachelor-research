import numpy as np
from sklearn.decomposition import PCA

from pipeline.dimensionality_reduction.abstract_reducer import AbstractReducer


class PCAReducer(AbstractReducer):
    def __init__(self, n_components: int) -> None:
        """Inits a PCAReducer instance.

        :param n_components: An integer indicating the number of features to retain.
        """
        self.pca: PCA = PCA(n_components=n_components)

    def reduce_dimensions(self, samples: np.ndarray) -> np.ndarray:
        """Reduces the dimensions of the given samples using Singular Value
        Decomposition (SVD).

        :param samples: A 2-d numpy array of shape (n_samples, n_features) containing
            the samples.
        :return: A 2-d numpy array of shape (n_samples, n_reduced_features) containing
            the samples in a latent space with a lower dimensionality.
        """
        self.pca: PCA = self.pca.fit(samples)
        return self.pca.transform(samples)

    def get_explained_variance(self) -> float:
        """Returns the variance explained by the retained components.

        :return: A float indicating the explained variance.
        """
        return sum(self.pca.explained_variance_ratio_)
