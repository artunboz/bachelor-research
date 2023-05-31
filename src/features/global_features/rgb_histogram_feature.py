import cv2 as cv
import numpy as np

from src.features.global_features.abstract_global_feature import (
    AbstractGlobalFeature,
)


class RGBHistogramFeature(AbstractGlobalFeature):
    def __init__(
        self,
        resize_size: tuple[int, int],
    ) -> None:
        super().__init__(resize_size)

    def read_image(self, image_path: str) -> np.ndarray:
        image: np.ndarray = cv.imread(image_path)
        return cv.resize(image, self.resize_size)

    def compute_image_features(self, image: np.ndarray) -> np.ndarray:
        """Computes the RGB histogram feature with (256 * 3,) dimensions.

        :param image: A numpy array containing the image.
        :return: A 1-d numpy feature vector.
        """
        bgr_planes: tuple[np.ndarray, np.ndarray, np.ndarray] = cv.split(image)
        hist_size: int = 256
        hist_range: tuple[int, int] = (0, 256)  # the upper boundary is exclusive
        accumulate: bool = False
        b_hist: np.ndarray = cv.calcHist(
            bgr_planes, [0], None, [hist_size], hist_range, accumulate=accumulate
        ).ravel()
        g_hist: np.ndarray = cv.calcHist(
            bgr_planes, [1], None, [hist_size], hist_range, accumulate=accumulate
        ).ravel()
        r_hist: np.ndarray = cv.calcHist(
            bgr_planes, [2], None, [hist_size], hist_range, accumulate=accumulate
        ).ravel()
        return np.concatenate((b_hist, g_hist, r_hist)).astype(int)
