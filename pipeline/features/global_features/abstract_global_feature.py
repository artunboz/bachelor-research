import os
from abc import abstractmethod

import numpy as np
from tqdm import tqdm

from pipeline.features.abstract_feature import AbstractFeature


class AbstractGlobalFeature(AbstractFeature):
    def extract_features(self, image_folder_path: str) -> np.ndarray:
        """Extracts features from all the images found in the folder located at the
        given path and returns them in a 2-d numpy array of shape
        (n_images, n_features).

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :return: A 2-d numpy array of shape (n_images, n_features).
        """
        features_list: list[np.ndarray] = []
        for image_file in tqdm(
            sorted(os.listdir(image_folder_path)),
            desc="Extracting features from the images",
        ):
            image: np.ndarray = self.read_image(f"{image_folder_path}/{image_file}")
            features_list.append(self.compute_image_features(image))
        return np.stack(features_list, axis=0)

    @abstractmethod
    def compute_image_features(self, image: np.ndarray) -> np.ndarray:
        """Computes the features for the given image.

        :param image: A numpy array containing the image.
        :return: A numppy array containing the
        """
