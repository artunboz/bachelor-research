from abc import ABC, abstractmethod

import numpy as np


class AbstractFeature(ABC):
    # @abstractmethod
    # def compute_features_image(self, image: np.ndarray) -> np.ndarray:
    #     """Computes features for the given image.
    #
    #     :param image: A numpy ndarray representing the image.
    #     :return: A numpy ndarray containing the computed features.
    #     """

    @abstractmethod
    def compute_features(self, image_folder_path: str) -> np.ndarray:
        """Computes features for all the images found in the folder located at the given
        path and return them in a 2-d numpy array of shape (n_images, n_features).

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :return: A 2-d numpy array of shape (n_images, n_features).
        """
