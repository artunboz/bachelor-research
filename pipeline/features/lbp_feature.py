import os

import numpy as np
from skimage import feature
from skimage import io
from skimage import transform
from tqdm import tqdm

from pipeline.features.abstract_feature import AbstractFeature


class LBPFeature(AbstractFeature):
    def __init__(
        self,
        p: int = 8,
        r: int = 1,
        method: str = "uniform",
        resize_size: tuple[int, int] = (96, 96),
    ) -> None:
        """

        :param p:
        :param r:
        :param method:
        :param resize_size:
        """
        self.p: int = p
        self.r: int = r
        self.method: str = method
        self.resize_size: tuple[int, int] = resize_size

    def compute_features_image(self, image: np.ndarray) -> np.ndarray:
        """Computes LBP features for the given image.

        :param image: A numpy ndarray representing the image.
        :return: A numpy ndarray containing the computed features.
        """
        image = transform.resize(image, self.resize_size)
        return feature.local_binary_pattern(image, self.p, self.r, self.method).ravel()

    def compute_features(self, image_folder_path: str) -> np.ndarray:
        """Computes LBP features for all the images found in the folder located at the
        given path and return them in a 2-d numpy array of shape (n_images, n_features).

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :return: A 2-d numpy array of shape (n_images, n_features).
        """
        features_list: list[np.ndarray] = []
        for image_file in tqdm(
            sorted(os.listdir(image_folder_path)),
            desc="LBP features extracted from images",
        ):
            image: np.ndarray = io.imread(
                f"{image_folder_path}/{image_file}", as_gray=True
            )
            features_list.append(self.compute_features_image(image))
        return np.stack(features_list, axis=0)
