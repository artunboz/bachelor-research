import os
from abc import abstractmethod
from typing import Optional

import numpy as np
from tqdm import tqdm

from pipeline.features.abstract_feature import AbstractFeature
from pipeline.features.local_features.bag_of_visual_words import compute_bovw_features


class AbstractLocalFeature(AbstractFeature):
    def __init__(
        self, resize_size: tuple[int, int], bovw_n_clusters_space: list[int]
    ) -> None:
        """Inits and AbstractLocalFeature instance.

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param bovw_n_clusters_space: A list of integers representing the search space
            for the optimal n_clusters value based on the silhouette score.
        """
        super().__init__(resize_size)
        self.bovw_n_clusters_space: list[int] = bovw_n_clusters_space
        self.image_names: Optional[list[str]] = None
        self.image_features: Optional[np.ndarray] = None

    def extract_features(self, image_folder_path: str) -> np.ndarray:
        """Extracts features from all the images found in the folder located at the
        given path and returns them in a 2-d numpy array of shape
        (n_images, n_features).

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :return: A 2-d numpy array of shape (n_images, n_features).
        """
        sorted_image_names: list[str] = sorted(os.listdir(image_folder_path))
        image_name_descriptor_list: list[tuple[str, np.ndarray]] = []
        for image_name in tqdm(
            sorted_image_names,
            desc="Extracting features from the images",
        ):
            image: np.ndarray = self.read_image(f"{image_folder_path}/{image_name}")
            descriptors: np.ndarray = self.get_descriptors(image)
            if descriptors is not None:
                image_name_descriptor_list.append((image_name, descriptors))

        self.image_names, descriptor_list = list(zip(*image_name_descriptor_list))

        print("Computing bag of features...")
        self.image_features = compute_bovw_features(
            descriptor_list, self.bovw_n_clusters_space
        )
        print("Bag of features computed.")

        return self.image_features

    @abstractmethod
    def get_descriptors(self, image: np.ndarray) -> np.ndarray:
        """Computes the descriptors for the given image.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the descriptors.
        """
