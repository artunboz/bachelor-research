import cv2 as cv
import numpy as np

from pipeline.features.local_features.abstract_local_feature import AbstractLocalFeature


class ORBFeature(AbstractLocalFeature):
    def __init__(
        self,
        resize_size: tuple[int, int],
        bovw_n_clusters_range: tuple[int, int],
        n_keypoints: int,
        patch_size: int,
    ) -> None:
        """Inits a ORBFeature instance. The underlying implementation relies on OpenCV's
        implementation of the ORB feature.

        :param resize_size: A 2-tuple of integers indicating the pixel width and height
            of the resized image.
        :param bovw_n_clusters_range: A 2-tuple of integers indicating the start
            (inclusive) and stop (exclusive) for the range of values to try for the
            number of clusters used for bag-of-visual-words.
        :param n_keypoints: An integer indicating the maximum number of keypoints to
            detect in an image.
        :param patch_size: An integer indicating the value to use for the edgeThreshold
            and patchSize parameters of OpenCV's ORB implementation.
        """
        super().__init__(resize_size, bovw_n_clusters_range)
        self.orb: cv.ORB = cv.ORB_create(
            nfeatures=n_keypoints, edgeThreshold=patch_size, patchSize=patch_size
        )

    def read_image(self, image_path: str) -> np.ndarray:
        """Reads the image found in the given path as grayscale and resizes the image
        using the self.resize_size attribute.

        :param image_path: A string indicating the path to the image.
        :return: A numpy array containing the image.
        """
        image: np.ndarray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        return cv.resize(image, self.resize_size)

    def get_descriptors(self, image: np.ndarray) -> np.ndarray:
        """Computes the descriptors for the given image.

        :param image: A numpy array containing the image.
        :return: A numpy array containing the descriptors.
        """
        _, des = self.orb.detectAndCompute(image, None)
        return des
