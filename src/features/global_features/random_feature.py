import numpy as np

from src.features.global_features.abstract_global_feature import (
    AbstractGlobalFeature,
)


class RandomFeature(AbstractGlobalFeature):
    def __init__(
        self, resize_size: tuple[int, int] = (48, 48), n_dims: int = 10
    ) -> None:
        """Inits a RandomFeature instance.

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param n_dims: An integer indicating the number of dimensions of the random
            feature.
        """
        super().__init__(resize_size)
        self.n_dims: int = n_dims
        self.rng: np.random.Generator = np.random.default_rng()

    def read_image(self, image_path: str) -> np.ndarray:
        """Returns an empty numpy array as this method is not needed for this feature.

        :param image_path: A string indicating the path to the image.
        :return: A numpy array containing the image.
        """
        return np.array([])

    def compute_image_features(self, image: np.ndarray) -> np.ndarray:
        """Returns a numpy array of shape (self.n_dims,) containing random floats
        between -1 and 1.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the computed features.
        """
        return self.rng.standard_normal(self.n_dims)
