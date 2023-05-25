from typing import Optional

import cv2 as cv
import numpy as np

from src.features.local_features.abstract_local_feature import AbstractLocalFeature


class ORBFeature(AbstractLocalFeature):
    def __init__(
        self,
        resize_size: tuple[int, int],
        quantization_method: str,
        n_components_space: list[int],
        n_features: int = 20,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        first_level: int = 0,
        wta_k: int = 2,
        patch_size: int = 31,
        fast_threshold: int = 20,
    ) -> None:
        """Inits a ORBFeature instance. The underlying implementation relies on OpenCV's
        implementation of the ORB feature. The parameter descriptions are taken from the
        documentation of OpenCV. For more details, please refer to
        https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param quantization_method: A sting indicating the quantization method for
            converting the local descriptors of an image to a single feature vector.
            Available options are "fisher" for fisher vectors and "bovw" for
            bag-of-visual-words.
        :param n_components_space: A list of integers containing either the number of
            components to use or multiple numbers that form the options to choose the
            best number from for vector quantization.
        :param n_features: An integer indicating the maximum number of features to
            retain.
        :param scale_factor: A float indicating the pyramid decimation ratio, greater
            than 1.
        :param n_levels: An integer indicating the number of pyramid levels.
        :param first_level: An integer indicating the level of pyramid to put source
            image to. Previous layers are filled with upscaled source image.
        :param wta_k: An integer indicating the number of points that produce each
            element of the oriented BRIEF descriptor.
        :param patch_size: An integer indicating the size of the patch used by the
            oriented BRIEF descriptor.
        :param fast_threshold: An integer indicating the fast threshold
        """
        super().__init__(resize_size, quantization_method, n_components_space)
        self.n_features: int = n_features
        self.scale_factor: float = scale_factor
        self.n_levels: int = n_levels
        self.first_level: int = first_level
        self.wta_k: int = wta_k
        self.patch_size: int = patch_size
        self.fast_threshold: int = fast_threshold
        self.orb: cv.ORB = cv.ORB_create(
            nfeatures=self.n_features,
            scaleFactor=self.scale_factor,
            nlevels=self.n_levels,
            edgeThreshold=self.patch_size,
            firstLevel=self.first_level,
            WTA_K=self.wta_k,
            patchSize=self.patch_size,
            fastThreshold=self.fast_threshold,
        )

    def read_image(self, image_path: str) -> np.ndarray:
        """Reads the image found in the given path as grayscale and resizes the image
        using the self.resize_size attribute.

        :param image_path: A string indicating the path to the image.
        :return: A numpy array containing the image.
        """
        image: np.ndarray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        return cv.resize(image, self.resize_size)

    def get_descriptors(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Computes the descriptors for the given image.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the descriptors. None is returned if the
            OpenCV ORB object returns None or the number of returned keypoints is fewer
            than self.n_features.
        """
        _, des = self.orb.detectAndCompute(image, None)
        if des is None or des.shape[0] != self.n_features:
            return None
        else:
            return des
