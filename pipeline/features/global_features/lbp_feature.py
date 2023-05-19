import numpy as np
from skimage import feature, io, transform, util

from pipeline.features.global_features.abstract_global_feature import (
    AbstractGlobalFeature,
)


class LBPFeature(AbstractGlobalFeature):
    def __init__(
        self,
        resize_size: tuple[int, int],
        p: int = 8,
        r: int = 1,
        method: str = "uniform",
    ) -> None:
        """Inits a LBPFeature instance.

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param p: An integer indicating the number of circularly symmetric neighbor set
            points.
        :param r: An integer indicating the radius of circle.
        :param method: A string indicating the method to determine the pattern.
        """
        super().__init__(resize_size)
        self.p: int = p
        self.r: int = r
        self.method: str = method

    def read_image(self, image_path: str) -> np.ndarray:
        """Reads the image found in the given path as grayscale, resizes it based on the
        self.resize_size attribute and returns the image as a numpy array.

        :param image_path: A string indicating the path to the image.
        :return: A numpy array containing the image.
        """
        image: np.ndarray = io.imread(image_path, as_gray=True)
        image = transform.resize(image, self.resize_size[::-1])
        return util.img_as_uint(image)

    def compute_image_features(self, image: np.ndarray) -> np.ndarray:
        """Computes LBP features for the given image.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the computed features.
        """
        return feature.local_binary_pattern(image, self.p, self.r, self.method).ravel()
