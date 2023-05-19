import numpy as np
from skimage import feature
from skimage import io
from skimage import transform

from pipeline.features.global_features.abstract_global_feature import (
    AbstractGlobalFeature,
)


class HOGFeature(AbstractGlobalFeature):
    def __init__(
        self,
        resize_size: tuple[int, int],
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        channel_axis: int = -1,
    ) -> None:
        """Inits a HOGFeature instance. The default parameters are taken from the
        original paper (https://ieeexplore.ieee.org/document/1467360).

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param orientations: An integer indicating the number of orientation bins.
            Defaults to 9.
        :param pixels_per_cell: A 2-tuple of integers indicating the size (in pixels) of
            a cell. Defaults to (8, 8)
        :param cells_per_block: A 2-tuple of integers indicating the number of cells in
            each block. Defaults to (2, 2)
        :param block_norm: A string indicating the block normalization method. Options
            are L1, L1-sqrt, L2, and L2-Hys. Defaults to L2-Hys.
        :param channel_axis: If None, the image is assumed to be a grayscale (single
            channel) image. Otherwise, this parameter indicates which axis of the array
            corresponds to channels. Defaults to -1.
        """
        super().__init__(resize_size)
        self.orientations: int = orientations
        self.pixels_per_cell: tuple[int, int] = pixels_per_cell
        self.cells_per_block: tuple[int, int] = cells_per_block
        self.block_norm: str = block_norm
        self.channel_axis: int = channel_axis

    def read_image(self, image_path: str) -> np.ndarray:
        """Reads the image found in the given path, resizes it based on the
        self.resize_size attribute and returns the image as a numpy array.

        :param image_path: A string indicating the path to the image.
        :return: A numpy array containing the image.
        """
        image: np.ndarray = io.imread(image_path)
        return transform.resize(image, self.resize_size[::-1])

    def compute_image_features(self, image: np.ndarray) -> np.ndarray:
        """Computes HOG features for the given image.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the computed features.
        """
        return feature.hog(
            image=image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            channel_axis=self.channel_axis,
        )
