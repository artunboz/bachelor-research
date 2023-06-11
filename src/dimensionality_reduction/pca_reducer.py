from typing import cast

import numpy as np
from sklearn.decomposition import PCA

from src.dimensionality_reduction.abstract_reducer import AbstractReducer


class PCAReducer(AbstractReducer):
    def __init__(self, n_components: int) -> None:
        """Inits a PCAReducer instance.

        :param n_components: An integer indicating the number of features to retain.
        """
        super().__init__()
        self.n_components: int = n_components
        self.pca: PCA = PCA(n_components=self.n_components)

    def reduce_dimensions(self, features_dir: str) -> np.ndarray:
        """Reduces the dimensions of the given samples using Singular Value
        Decomposition (SVD).

        :param features_dir: A string indicating the file containing the features.
        :return: A 2-d numpy array of shape (n_samples, n_reduced_features) containing
            the samples in a latent space with a lower dimensionality.
        """
        features: np.ndarray = np.load(f"{features_dir}/features.npy")
        self.pca: PCA = cast(PCA, self.pca.fit(features))
        self.reduced_features = self.pca.transform(features)

        return self.reduced_features

    def get_explained_variance(self) -> float:
        """Returns the variance explained by the retained components.

        :return: A float indicating the explained variance.
        """
        return sum(self.pca.explained_variance_ratio_)
