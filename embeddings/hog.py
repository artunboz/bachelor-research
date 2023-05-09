import os

import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize


class HOGFeature:
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        channel_axis: int = -1,
        resize: bool = True,
    ) -> None:
        """Inits a HOGFeature instance. The default parameters are taken from the
            original paper (https://ieeexplore.ieee.org/document/1467360).

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
        :param resize: A boolean indicating whether any given image shall be resized to
            (64, 128) pixels, which is the size used in the original paper. Defaults to
            True.
        """
        # Check for square cells and blocks.
        assert pixels_per_cell[0] == pixels_per_cell[1]
        assert cells_per_block[0] == cells_per_block[1]

        self.orientations: int = orientations
        self.pixels_per_cell: tuple[int, int] = pixels_per_cell
        self.cells_per_block: tuple[int, int] = cells_per_block
        self.block_norm: str = block_norm
        self.channel_axis: int = channel_axis
        self.resize: bool = resize

    def compute_features_image_folder(self, image_folder_path: str) -> list[np.ndarray]:
        """Computes HOG features for every image found in the given folder. Returns
        the features in a list.

        :param image_folder_path: A string indicating the path to the folder containing
            the images.
        :return: A list of numpy ndarrays containing the features of images.
        """
        features_list: list[np.ndarray] = []
        for image_file in os.listdir(image_folder_path):
            image: np.ndarray = imread(f"{image_folder_path}/{image_file}")
            features_list.append(self.compute_features_image(image))
        return features_list

    def compute_features_image(
        self, image: np.ndarray, visualize=False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Computes HOG features for the given image. Also returns the hog image if the
        visualize parameter is set to True.

        :param image: A numpy ndarray representing the image.
        :param visualize: A boolean indicating whether hog image shall be returned.
        :return: A numpy ndarray containing the computed features and optionally a numpy
            ndarray representing the hog image.
        """
        if self.resize:
            resized_image: np.ndarray = resize(image, (128, 64))
        else:
            resized_image: np.ndarray = image

        if visualize:
            fd, hog_image = hog(
                image=resized_image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=visualize,
                channel_axis=self.channel_axis,
            )
            return fd, hog_image
        else:
            return hog(
                image=resized_image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=visualize,
                channel_axis=self.channel_axis,
            )
