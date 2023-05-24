from abc import ABC, abstractmethod

import numpy as np


class AbstractFeature(ABC):
    def __init__(self, resize_size: tuple[int, int]) -> None:
        """Inits an AbstractFeature instance.

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        """
        self.resize_size: tuple[int, int] = resize_size

    @abstractmethod
    def extract_features(self, image_folder_path: str) -> np.ndarray:
        """Extracts features from all the images found in the folder located at the
        given path and returns them in a 2-d numpy array of shape
        (n_images, n_features).

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :return: A 2-d numpy array of shape (n_images, n_features).
        """

    @abstractmethod
    def read_image(self, image_path: str) -> np.ndarray:
        """Reads the image found in the given path and returns the image as a numpy
        array.

        :param image_path: A string indicating the path to the image.
        :return: A numpy array containing the image.
        """

    @abstractmethod
    def get_config(self) -> str:
        """Returns the configuration of the feature as a string.

        :return: A string indicating the configuration of the feature.
        """
